# Install required packages
import subprocess
subprocess.check_call(['pip', 'install', 'transformers==4.46.1', 'datasets', 'scikit-learn', 'sagemaker', 'boto3', 'torch'])

!pip install -U sagemaker

import os
import sys
import json
import torch
import logging
import numpy as np
import pandas as pd
import tarfile
import boto3
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LongformerTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from sagemaker.huggingface import HuggingFace
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner
)
import sagemaker
from sklearn.model_selection import train_test_split
import time
import random
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set target model
target_model = 'mortality'  # Change this to 'mortality', 'readmission', etc.

# Testing mode flag
IS_TESTING = True  # Set to False for production

# Model type to label mapping
LABEL_MAPPING = {
    'readmission': 'has_readmission',
    'mortality': 'inpt_mortality',
    'discharge_disposition': 'dischargedisposition_group',
    'care_escalation': 'care_escalation',
    'length-of-stay': 'keck_los'
}

# Model type to problem type mapping
PROBLEM_TYPE_MAPPING = {
    'readmission': 'single_label_classification',
    'mortality': 'single_label_classification',
    'discharge_disposition': 'multi_class',
    'care_escalation': 'single_label_classification',
    'length-of-stay': 'regression'
}

# Hyperparameter ranges for random search
HYPERPARAMETER_RANGES = {
    'learning_rate': ContinuousParameter(1e-5, 2e-5),
    'per_device_train_batch_size': CategoricalParameter([2, 4]),
    'num_train_epochs': IntegerParameter(3, 5),
    'weight_decay': ContinuousParameter(0.01, 0.1),
    'warmup_steps': IntegerParameter(100, 500),
    'gradient_accumulation_steps': CategoricalParameter([4, 8])
}

# Define model types and their configurations
MODEL_CONFIGS = {
    'length-of-stay': {
        'num_labels': 1,
        'problem_type': 'regression',
        'metrics': ['eval_mse', 'eval_mae'],
        'metric_for_best': 'eval_mse',
        'greater_is_better': False
    },
    'mortality': {
        'num_labels': 2,
        'problem_type': 'single_label_classification',
        'metrics': ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall'],
        'metric_for_best': 'eval_f1',
        'greater_is_better': True
    },
    'readmission': {
        'num_labels': 2,
        'problem_type': 'single_label_classification',
        'metrics': ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall'],
        'metric_for_best': 'eval_f1',
        'greater_is_better': True
    },
    'discharge_disposition': {
        'num_labels': 4,
        'problem_type': 'single_label_classification',
        'metrics': ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall'],
        'metric_for_best': 'eval_f1',
        'greater_is_better': True
    }
}

def get_s3_paths(target_model):
    """Get the S3 paths for the model training process"""
    return {
        'train_data': f's3://keck-dev-transfercenter-data/length-of-stay/raw/2024_compiled_intake',
        'features_file': f's3://keck-dev-transfercenter-data/length-of-stay/raw/2024_features.csv',
        'original_model': f's3://keck-dev-transfercenter-models/{target_model}/{target_model}.tar.gz',
        'output_model': f's3://keck-dev-transfercenter-models/{target_model}/{target_model}_2024.tar.gz',
        'train_manifest': f's3://keck-dev-transfercenter-data/{target_model}/train_manifest.json',
        'val_manifest': f's3://keck-dev-transfercenter-data/{target_model}/validation_manifest.json'
    }

# SageMaker setup
sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::248877539307:role/DS-ExecutionRole'
default_bucket = sagemaker_session.default_bucket()

# Local paths
LOCAL_MODEL_DIR = '/tmp/model'
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

def download_and_extract_model(model_path):
    """Download and extract the model from S3, ensuring clean directory and valid files."""
    s3 = boto3.client('s3')
    model_bucket, model_key = model_path[5:].split('/', 1)
    temp_tar = os.path.join(LOCAL_MODEL_DIR, "model.tar.gz")
    
    try:
        # Clear LOCAL_MODEL_DIR to prevent residual files
        if os.path.exists(LOCAL_MODEL_DIR):
            import shutil
            logger.info(f"Clearing existing model directory: {LOCAL_MODEL_DIR}")
            shutil.rmtree(LOCAL_MODEL_DIR)
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        # Download model
        s3.download_file(model_bucket, model_key, temp_tar)
        logger.info(f"Downloaded model from {model_path}")
        
        # Extract files
        with tarfile.open(temp_tar, "r:gz") as tar:
            for member in tar.getmembers():
                member.name = os.path.basename(member.name)  # Remove any path structure
                tar.extract(member, LOCAL_MODEL_DIR)
        
        os.remove(temp_tar)
        
        # Validate extracted files
        expected_files = {
            'config.json', 'model.safetensors', 'merges.txt', 'special_tokens_map.json',
            'tokenizer_config.json', 'tokenizer.json', 'vocab.json'
        }
        files = set(os.listdir(LOCAL_MODEL_DIR))
        logger.info(f"Extracted files: {files}")
        
        if not expected_files.issubset(files):
            logger.error(f"Missing expected files: {expected_files - files}")
            raise FileNotFoundError(f"Missing expected files: {expected_files - files}")
        
        # Remove unexpected files
        for file in files - expected_files:
            logger.warning(f"Removing unexpected file: {file}")
            os.remove(os.path.join(LOCAL_MODEL_DIR, file))
        
        return True
    except Exception as e:
        logger.error(f"Failed to download/extract model: {str(e)}")
        return False

def list_s3_objects(s3_path):
    """List objects in an S3 path."""
    s3 = boto3.client('s3')
    bucket, prefix = s3_path[5:].split('/', 1)  # Remove 's3://' and split
    
    logging.info(f"Listing objects in bucket: {bucket}, prefix: {prefix}")
    
    objects = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                objects.append(f"s3://{bucket}/{obj['Key']}")
    
    logging.info(f"Found {len(objects)} objects in {s3_path}")
    if len(objects) == 0:
        logging.warning(f"No files found in {s3_path}")
    
    return objects

def download_from_s3(s3_path):
    """Download a file from S3 to a temporary location."""
    s3 = boto3.client('s3')
    bucket, key = s3_path[5:].split('/', 1)  # Remove 's3://' and split
    
    # Create a temporary file
    local_path = os.path.join('/tmp', os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path

def save_to_s3(s3_path, content):
    """Save content to an S3 path."""
    s3 = boto3.client('s3')
    bucket, key = s3_path[5:].split('/', 1)  # Remove 's3://' and split
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        s3.upload_file(tmp.name, bucket, key)
    os.unlink(tmp.name)

def prepare_training_data(s3_paths, target_model):
    """
    Prepare training data by matching FINs between compiled intake files and features CSV
    """
    # Load features file and create mapping from FIN to label
    logging.info(f"Loading features from {s3_paths['features_file']}")
    features_df = pd.read_csv(s3_paths['features_file'])
    
    # Get the correct label column based on target model
    label_column = LABEL_MAPPING[target_model]
    logging.info(f"Using label column: {label_column}")
    
    # Convert FINs to strings and convert label to appropriate type based on problem type
    problem_type = PROBLEM_TYPE_MAPPING[target_model]
    if problem_type == 'regression':
        # For regression tasks, use float labels
        fin_to_label = {str(fin): float(label) for fin, label in zip(features_df['fin'], features_df[label_column])}
    else:
        # For classification tasks, use integer labels
        if target_model == 'readmission':
            # Define has_readmission based on multiple readmission counter columns
            features_df['has_readmission'] = (
                (features_df['all_readm_6h_30d_counter'] == 1) |
                (features_df['readm_6h_30d_counter_same_drg'] == 1) |
                (features_df['readm_6h_30d_counter_diff_drg'] == 1) |
                (features_df['all_readm_gt_30d_60d_counter'] == 1) |
                (features_df['readm_gt_30d_60d_counter_same_drg'] == 1) |
                (features_df['readm_gt_30d_60d_counter_diff_drg'] == 1) |
                (features_df['all_readm_gt_60d_90d_counter'] == 1) |
                (features_df['readm_gt_60d_90d_counter_same_drg'] == 1) |
                (features_df['readm_gt_60d_90d_counter_diff_drg'] == 1) |
                (features_df['all_readm_6h_60d_counter'] == 1) |
                (features_df['all_readm_6h_90d_counter'] == 1) |
                (features_df['readm_6h_60d_counter_same_drg'] == 1) |
                (features_df['readm_6h_90d_counter_same_drg'] == 1) |
                (features_df['readm_6h_60d_counter_diff_drg'] == 1) |
                (features_df['readm_6h_90d_counter_diff_drg'] == 1)
            ).astype(int)
            fin_to_label = {str(fin): int(label) for fin, label in zip(features_df['fin'], features_df['has_readmission'])}
        else:
            fin_to_label = {str(fin): int(label) for fin, label in zip(features_df['fin'], features_df[label_column])}
    
    logging.info(f"Loaded {len(fin_to_label)} FIN-to-label mappings")
    
    # Print sample of FINs from features file
    sample_fins_features = list(fin_to_label.keys())[:5]
    logging.info(f"Sample FINs from features file: {sample_fins_features}")
    
    # Get list of text files from S3
    text_files = list_s3_objects(s3_paths['train_data'])
    
    # If in testing mode, use a larger subset to increase chances of valid data
    if IS_TESTING:
        logging.info("Testing mode: Using subset of data")
        random.seed(42)  # For reproducibility
        text_files = random.sample(text_files, min(2400, len(text_files)))
    
    # Initialize counters and lists for tracking
    total_files = len(text_files)
    if total_files == 0:
        raise ValueError(f"No files found in {s3_paths['train_data']}")
        
    logging.info(f"Processing {total_files} files...")
    
    files_with_fin = 0
    files_with_matching_fin = 0
    sample_fins_intake = []  # Store sample FINs from intake files
    failed_files = []  # Track files that fail JSON decoding
    
    # Process each text file
    train_manifest = []
    val_manifest = []
    
    for i, text_file in enumerate(text_files, 1):
        if i % 100 == 0:
            logging.info(f"Processed {i}/{total_files} files...")
        
        # Download and read the text file
        local_path = download_from_s3(text_file)
        with open(local_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            # Find the case attributes JSON at the end of the file
            lines = content.split('\n')
            for line in reversed(lines):
                if line.strip():
                    case_attrs = json.loads(line)
                    break
            else:
                logging.warning(f"No valid JSON found in {text_file}")
                failed_files.append((text_file, "No valid JSON"))
                continue
            
            # Extract FIN__c from case attributes and convert to string
            fin = str(case_attrs.get('FIN__c')) if case_attrs.get('FIN__c') is not None else None
            if fin:
                files_with_fin += 1
                # Collect sample FINs from intake files
                if len(sample_fins_intake) < 5:
                    sample_fins_intake.append(fin)
                
                # Check if FIN exists in features file
                if fin in fin_to_label:
                    files_with_matching_fin += 1
                    label = fin_to_label[fin]
                    
                    # Add to manifest
                    manifest_entry = {
                        'source': text_file,
                        'label': label
                    }
                    
                    # Split 80/20 for train/val
                    if random.random() < 0.8:
                        train_manifest.append(manifest_entry)
                    else:
                        val_manifest.append(manifest_entry)
                    
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to decode JSON from file {text_file}: {str(e)}")
            failed_files.append((text_file, str(e)))
            continue
        finally:
            os.unlink(local_path)
    
    # Log statistics
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Files with FIN: {files_with_fin}")
    logging.info(f"Files with matching FIN: {files_with_matching_fin}")
    logging.info(f"Files failed JSON decoding: {len(failed_files)}")
    if total_files > 0:
        logging.info(f"FIN match rate: {files_with_matching_fin/total_files*100:.2f}%")
    
    # Print sample FINs from intake files for comparison
    logging.info(f"Sample FINs from intake files: {sample_fins_intake}")
    
    # Log a sample of failed files
    if failed_files:
        logging.info(f"Sample of failed files (first 5): {failed_files[:5]}")
    
    # Save manifests to S3
    save_to_s3(s3_paths['train_manifest'], json.dumps(train_manifest))
    save_to_s3(s3_paths['val_manifest'], json.dumps(val_manifest))
    
    if not train_manifest:
        raise ValueError("No valid training data found after FIN matching")
    
    return len(train_manifest), len(val_manifest)

def tokenize_function(examples, tokenizer):
    """Tokenize examples."""
    tokenized = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=4096)
    tokenized['labels'] = examples['label']
    return tokenized

def compute_metrics(eval_pred, problem_type):
    """Compute metrics based on problem type."""
    predictions, labels = eval_pred
    
    if problem_type == 'regression':
        predictions = predictions.squeeze()
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        # Print metrics in the exact format expected by SageMaker
        print(f'{{"eval_loss": {mse}, "eval_mse": {mse}, "eval_mae": {mae}}}')
        
        return {
            'eval_loss': mse,
            'eval_mse': mse,
            'eval_mae': mae
        }
    else:
        if problem_type == 'binary':
            predictions = (predictions.squeeze() > 0.5).astype(int)
        else:  # multi_class
            predictions = predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        
        metrics = {
            'eval_accuracy': acc,
            'eval_f1': f1,
            'eval_precision': precision,
            'eval_recall': recall
        }
        
        # Print metrics in the format expected by SageMaker
        print(json.dumps(metrics))
        
        return metrics

def create_training_script():
    """Create the training script for the model."""
    os.makedirs('code', exist_ok=True)
    
    training_script = '''
import os
import sys
import json
import torch
import logging
import numpy as np
import pandas as pd
import tarfile
import boto3
import tempfile
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LongformerTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from torch.nn import MSELoss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify safetensors is installed
try:
    import safetensors
except ImportError:
    logger.error("safetensors library is required for loading model.safetensors")
    raise ImportError("Please install safetensors: pip install safetensors")

# Load model configurations
MODEL_CONFIGS = {
    'length-of-stay': {
        'num_labels': 1,
        'problem_type': 'regression',
        'metrics': ['eval_mse', 'eval_mae'],
        'metric_for_best': 'eval_mse',
        'greater_is_better': False
    },
    'mortality': {
        'num_labels': 2,
        'problem_type': 'single_label_classification',
        'metrics': ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall'],
        'metric_for_best': 'eval_f1',
        'greater_is_better': True
    },
    'readmission': {
        'num_labels': 2,
        'problem_type': 'single_label_classification',
        'metrics': ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall'],
        'metric_for_best': 'eval_f1',
        'greater_is_better': True
    },
    'discharge_disposition': {
        'num_labels': 4,
        'problem_type': 'single_label_classification',
        'metrics': ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall'],
        'metric_for_best': 'eval_f1',
        'greater_is_better': True
    }
}

def load_dataset(manifest_path):
    """Load dataset from manifest file."""
    logger.info(f"Loading dataset from {manifest_path}")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found at {manifest_path}")
        raise FileNotFoundError(f"No such file: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    texts = []
    labels = []
    sources = []  # Keep track of source files for FIN extraction
    
    for item in manifest:
        source = item['source']
        if source.startswith('s3://'):
            # Handle S3 paths
            s3 = boto3.client('s3')
            bucket_prefix = source[5:]
            bucket, key = bucket_prefix.split('/', 1)
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                try:
                    s3.download_file(bucket, key, tmp.name)
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        text = f.read()
                finally:
                    os.unlink(tmp.name)
        else:
            # Handle local files
            with open(source, 'r') as f:
                text = f.read()
        
        texts.append(text)
        labels.append(item['label'])
        sources.append(source)
    
    dataset = Dataset.from_dict({'text': texts, 'label': labels, 'source': sources})
    logger.info(f"Created dataset with {len(dataset)} samples")
    return dataset

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss

def compute_metrics_regression(eval_pred):
    """Compute metrics for regression tasks and log sample predictions."""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    
    # Print metrics in the exact format expected by SageMaker
    print(f'{{"eval_loss": {mse}, "eval_mse": {mse}, "eval_mae": {mae}}}')
    
    # Try to extract and log a sample of predictions with FINs
    try:
        try:
            val_dataset = eval_pred.data_loader.dataset
            sources = val_dataset['source']
            logger.info(f"Extracting sample predictions from {len(sources)} validation samples")
            
            # Just extract a few FINs for the sample (limit to 5)
            sample_size = min(5, len(sources))
            sample_indices = range(sample_size)
            sample_sources = [sources[i] for i in sample_indices]
            sample_labels = [labels[i] for i in sample_indices]
            sample_predictions = [predictions[i] for i in sample_indices]
            
            # Extract FINs from the sample sources
            sample_fins = []
            for source in sample_sources:
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        if source.startswith('s3://'):
                            s3 = boto3.client('s3')
                            bucket_prefix = source[5:]
                            bucket, key = bucket_prefix.split('/', 1)
                            s3.download_file(bucket, key, tmp.name)
                        else:
                            with open(source, 'r') as f:
                                tmp.write(f.read())
                        
                        with open(tmp.name, 'r') as f:
                            lines = f.readlines()
                            fin = None
                            for line in reversed(lines):
                                if line.strip():
                                    try:
                                        case_attrs = json.loads(line)
                                        fin = str(case_attrs.get('FIN__c', 'Unknown'))
                                        break
                                    except json.JSONDecodeError:
                                        continue
                            sample_fins.append(fin if fin else "Unknown")
                    os.unlink(tmp.name)
                except Exception as e:
                    logger.warning(f"Error extracting FIN from {source}: {str(e)}")
                    sample_fins.append("Error")
            
            # Log the sample data in a readable format
            logger.info("Sample of regression predictions (FIN, Actual, Predicted):")
            for i in range(len(sample_fins)):
                logger.info(f"  {sample_fins[i]}, {sample_labels[i]:.4f}, {sample_predictions[i]:.4f}")
            
        except AttributeError:
            # If data_loader.dataset is not available, just log predictions without FINs
            logger.warning("Cannot access validation dataset sources. Logging predictions without FINs.")
            sample_size = min(5, len(labels))
            logger.info("Sample of regression predictions (Actual, Predicted):")
            for i in range(sample_size):
                logger.info(f"  {labels[i]:.4f}, {predictions[i]:.4f}")
                
    except Exception as e:
        logger.error(f"Failed to log prediction samples: {str(e)}")
    
    return {
        'eval_loss': mse,
        'eval_mse': mse,
        'eval_mae': mae
    }

def compute_metrics_classification(eval_pred):
    """Compute metrics for classification tasks and log sample predictions."""
    logger.info("Starting compute_metrics_classification")
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    # Create and log confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    
    # Format the confusion matrix for logging
    class_names = ["0", "1"]
    if cm.shape[0] > 2:  # Handle multi-class case
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Print confusion matrix header
    logger.info("Confusion Matrix:")
    header = "True\\Pred |" + " | ".join(f"{c:^5}" for c in class_names) + " |"
    logger.info(header)
    logger.info("-" * len(header))
    
    # Print each row of the confusion matrix
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:^9} |" + " | ".join(f"{val:^5}" for val in row) + " |"
        logger.info(row_str)
    
    # Calculate class-wise metrics for binary classification
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall for positive class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # recall for negative class
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # precision for positive class
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # precision for negative class
        
        logger.info("-" * len(header))
        logger.info(f"Class 1 (Sensitivity/Recall): {sensitivity:.4f}")
        logger.info(f"Class 0 (Specificity): {specificity:.4f}")
        logger.info(f"Class 1 (Precision/PPV): {ppv:.4f}")
        logger.info(f"Class 0 (NPV): {npv:.4f}")
        logger.info(f"Class 1 count (actual): {tp + fn}")
        logger.info(f"Class 0 count (actual): {tn + fp}")
    
    # Print metrics in the exact format expected by SageMaker
    metrics = {
        'eval_accuracy': acc,
        'eval_f1': f1,
        'eval_precision': precision,
        'eval_recall': recall
    }
    print(json.dumps(metrics))
    
    # Try to extract and log a sample of predictions with FINs
    try:
        try:
            val_dataset = eval_pred.data_loader.dataset
            sources = val_dataset['source']
            logger.info(f"Extracting sample predictions from {len(sources)} validation samples")
            
            # Just extract a few FINs for the sample (limit to 5)
            sample_size = min(5, len(sources))
            sample_indices = range(sample_size)
            sample_sources = [sources[i] for i in sample_indices]
            sample_labels = [labels[i] for i in sample_indices]
            sample_predictions = [predictions[i] for i in sample_indices]
            
            # Extract FINs from the sample sources
            sample_fins = []
            for source in sample_sources:
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        if source.startswith('s3://'):
                            s3 = boto3.client('s3')
                            bucket_prefix = source[5:]
                            bucket, key = bucket_prefix.split('/', 1)
                            s3.download_file(bucket, key, tmp.name)
                        else:
                            with open(source, 'r') as f:
                                tmp.write(f.read())
                        
                        with open(tmp.name, 'r') as f:
                            lines = f.readlines()
                            fin = None
                            for line in reversed(lines):
                                if line.strip():
                                    try:
                                        case_attrs = json.loads(line)
                                        fin = str(case_attrs.get('FIN__c', 'Unknown'))
                                        break
                                    except json.JSONDecodeError:
                                        continue
                            sample_fins.append(fin if fin else "Unknown")
                    os.unlink(tmp.name)
                except Exception as e:
                    logger.warning(f"Error extracting FIN from {source}: {str(e)}")
                    sample_fins.append("Error")
            
            # Log the sample data in a readable format
            logger.info("Sample of predictions (FIN, Actual, Predicted):")
            for i in range(len(sample_fins)):
                logger.info(f"  {sample_fins[i]}, {sample_labels[i]}, {sample_predictions[i]}")
            
        except AttributeError:
            # If data_loader.dataset is not available, just log predictions without FINs
            logger.warning("Cannot access validation dataset sources. Logging predictions without FINs.")
            sample_size = min(5, len(labels))
            logger.info("Sample of predictions (Actual, Predicted):")
            for i in range(sample_size):
                logger.info(f"  {labels[i]}, {predictions[i]}")
                
    except Exception as e:
        logger.error(f"Failed to log prediction samples: {str(e)}")
    
    logger.info("Completed compute_metrics_classification")
    return metrics

def extract_model_files(model_tar_path, output_dir):
    """Extract model files from tar.gz and validate contents."""
    logger.info(f"Extracting model files from {model_tar_path}")
    
    # Create a temporary directory for downloading the model if needed
    temp_model_path = None
    try:
        # Check if model_tar_path exists, attempt to download from S3 if missing
        s3_model_path = f"s3://keck-dev-transfercenter-models/mortality/mortality.tar.gz"
        if not os.path.exists(model_tar_path):
            logger.warning(f"Model file not found at {model_tar_path}. Attempting to download from {s3_model_path}")
            try:
                # Download to a temporary location first
                temp_model_path = os.path.join('/tmp', os.path.basename(model_tar_path))
                s3 = boto3.client('s3')
                bucket, key = s3_model_path[5:].split('/', 1)
                s3.download_file(bucket, key, temp_model_path)
                logger.info(f"Successfully downloaded model from {s3_model_path} to {temp_model_path}")
                
                # Use the temp file as our model tar path
                model_tar_path = temp_model_path
            except Exception as e:
                logger.error(f"Failed to download model from {s3_model_path}: {str(e)}")
                raise FileNotFoundError(f"Could not download model from {s3_model_path}: {str(e)}")
        
        # Clean output directory (don't delete the model.tar.gz file)
        if os.path.exists(output_dir):
            import shutil
            logger.info(f"Clearing existing model directory: {output_dir}")
            # List files to delete, excluding our model.tar.gz file
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if item_path != model_tar_path:  # Don't delete our model file
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract files to root of output_dir
        logger.info(f"Extracting from {model_tar_path} to {output_dir}")
        with tarfile.open(model_tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                member.name = os.path.basename(member.name)  # Remove any path structure
                tar.extract(member, output_dir)
        
        # Validate extracted files
        expected_files = {
            'config.json', 'model.safetensors', 'merges.txt', 'special_tokens_map.json',
            'tokenizer_config.json', 'tokenizer.json', 'vocab.json'
        }
        files = set(os.listdir(output_dir))
        logger.info(f"Extracted files: {files}")
        
        if not expected_files.issubset(files):
            missing_files = expected_files - files
            logger.warning(f"Missing some expected files: {missing_files}")
            # Continue anyway as some models might not have all files
        
        return output_dir
    finally:
        # Clean up temp file if created
        if temp_model_path and os.path.exists(temp_model_path):
            os.unlink(temp_model_path)

def main():
    # Set environment variables for offline mode
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    # Set up paths for manifest files in SageMaker environment
    train_manifest = '/opt/ml/input/data/train/train_manifest.json'
    val_manifest = '/opt/ml/input/data/validation/validation_manifest.json'
    
    # Get hyperparameters
    with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
        hyperparameters = json.load(f)
    
    # Strip quotes from all hyperparameter values
    for key in hyperparameters:
        if isinstance(hyperparameters[key], str):
            hyperparameters[key] = hyperparameters[key].strip('"')
    
    # Match hyperparameters against MODEL_CONFIGS
    target_model = hyperparameters.get('TARGET_MODEL', '').strip('"')
    problem_type = hyperparameters.get('problem_type', '').strip('"')
    num_labels = int(hyperparameters.get('num_labels', 1))
    
    # Add debug logging
    logger.info(f"Received hyperparameters: {hyperparameters}")
    logger.info(f"Looking for model with TARGET_MODEL='{target_model}', problem_type='{problem_type}', and num_labels={num_labels}")
    logger.info(f"Available configurations: {MODEL_CONFIGS}")
    
    # Find matching model config
    model_config = None
    if target_model in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[target_model]
        logger.info(f"Using model configuration for: {target_model}")
    else:
        logger.warning(f"TARGET_MODEL '{target_model}' not found in MODEL_CONFIGS. Falling back to problem_type and num_labels.")
        for model_name, config in MODEL_CONFIGS.items():
            logger.info(f"Checking {model_name}: {config}")
            if config['problem_type'] == problem_type and config['num_labels'] == num_labels:
                target_model = model_name
                model_config = config
                logger.info(f"Selected model configuration for: {target_model}")
                break
    
    if not model_config:
        raise ValueError(f"No matching model found for TARGET_MODEL='{target_model}', problem_type={problem_type}, and num_labels={num_labels}")
    
    # Load datasets
    train_dataset = load_dataset(train_manifest)
    val_dataset = load_dataset(val_manifest)
    
    # Log dataset sizes
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    if model_config['problem_type'] != 'regression':
        # Count Class 0 and Class 1 in training dataset
        train_labels = train_dataset['label']
        train_class_0_count = sum(1 for label in train_labels if label == 0)
        train_class_1_count = sum(1 for label in train_labels if label == 1)
        logger.info(f"Training dataset class distribution: Class 0: {train_class_0_count}, Class 1: {train_class_1_count}")
        
        # Count Class 0 and Class 1 in validation dataset
        val_labels = val_dataset['label']
        val_class_0_count = sum(1 for label in val_labels if label == 0)
        val_class_1_count = sum(1 for label in val_labels if label == 1)
        logger.info(f"Validation dataset class distribution: Class 0: {val_class_0_count}, Class 1: {val_class_1_count}")
    
    # Debug: Log contents of model channel
    model_channel_dir = '/opt/ml/input/data/model'
    if os.path.exists(model_channel_dir):
        model_channel_files = os.listdir(model_channel_dir)
        logger.info(f"Contents of model channel {model_channel_dir}: {model_channel_files}")
    else:
        logger.warning(f"Model channel directory {model_channel_dir} does not exist")
    
    # Extract model files to ensure they're at root level
    model_tar = '/opt/ml/input/data/model/model.tar.gz'
    model_dir = extract_model_files(model_tar, '/opt/ml/input/data/model')
    
    logger.info(f"Checking model directory: {model_dir}")
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        logger.info(f"Files in model directory: {files}")
    else:
        logger.error(f"Model directory does not exist: {model_dir}")
        raise FileNotFoundError(f"No such directory: {model_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_safetensors=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer with AutoTokenizer: {str(e)}")
        logger.info("Attempting to load with LongformerTokenizer...")
        tokenizer = LongformerTokenizer.from_pretrained(model_dir, local_files_only=True, use_safetensors=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        num_labels=model_config['num_labels'],
        problem_type=model_config['problem_type'],
        use_safetensors=True
    )
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=1024)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    logger.info("Tokenization completed")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        num_train_epochs=int(hyperparameters.get('num_train_epochs', 10)),
        per_device_train_batch_size=int(hyperparameters.get('per_device_train_batch_size', 4)),
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=int(hyperparameters.get('gradient_accumulation_steps', 4)),
        fp16=hyperparameters.get('bf16', '').lower() == 'true',
        fp16_opt_level="O2",
        max_grad_norm=float(hyperparameters.get('max_grad_norm', 1.0)),
        warmup_steps=int(hyperparameters.get('warmup_steps', 50)),
        weight_decay=float(hyperparameters.get('weight_decay', 0.01)),
        logging_dir='/opt/ml/output/logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model=model_config['metric_for_best'],
        greater_is_better=model_config['greater_is_better'],
        save_total_limit=2
    )
    
    # Initialize appropriate trainer based on problem type
    if model_config['problem_type'] == 'regression':
        trainer = RegressionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_regression,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_classification,
        )
    
    # Train and save
    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")
    trainer.save_model()
    tokenizer.save_pretrained("/opt/ml/model")
    logger.info("Model and tokenizer saved")

if __name__ == "__main__":
    logger.info("Starting train.py execution")
    main()
    logger.info("train.py execution completed")
'''
    
    # Save the training script
    with open('code/train.py', 'w') as f:
        f.write(training_script)
    logger.info("Created training script at code/train.py")
    
    # Create requirements.txt
    requirements = '''
# Note: pip root warning is expected in SageMaker's environment
transformers==4.28.1
datasets==2.13.1
scikit-learn==1.3.0
torch==2.0.0
torchvision==0.15.1
numpy==1.24.3
pandas==2.0.3
safetensors==0.4.3
'''
    with open('code/requirements.txt', 'w') as f:
        f.write(requirements)
    logger.info("Created requirements.txt at code/requirements.txt")

def main():
    try:
        # Use target model defined at top of file
        if target_model not in LABEL_MAPPING:
            raise ValueError(f"Invalid target model. Must be one of: {list(LABEL_MAPPING.keys())}")
        
        # Get paths for this model
        paths = get_s3_paths(target_model)
        logger.info(f"Using paths for {target_model}: {paths}")
        
        # Validate model file in S3
        if not download_and_extract_model(paths['original_model']):
            raise Exception("Failed to download and extract model")

        # Verify model files
        model_files = os.listdir(LOCAL_MODEL_DIR)
        logger.info(f"Model files: {model_files}")

        # Prepare training data manifests
        train_count, val_count = prepare_training_data(paths, target_model)
        logger.info(f"Created train manifest at {paths['train_manifest']} with {train_count} entries")
        logger.info(f"Created validation manifest at {paths['val_manifest']} with {val_count} entries")
        
        # Create training script
        create_training_script()
        
        problem_type = PROBLEM_TYPE_MAPPING[target_model]
        num_labels = MODEL_CONFIGS[target_model]['num_labels']
        
        # Configure hyperparameter tuning job
        huggingface_estimator = HuggingFace(
            entry_point='train.py',
            source_dir='code',
            instance_type='ml.g5.4xlarge',
            instance_count=1,
            role=role,
            transformers_version='4.28.1',
            pytorch_version='2.0.0',
            py_version='py310',
            environment={
                'HF_MODEL_ID': '/opt/ml/input/data/model',
                'HF_TASK': 'text-classification',
                'TRANSFORMERS_OFFLINE': '1',
                'HF_DATASETS_OFFLINE': '1',
                'USE_CUDA_VISIBLE_DEVICES': 'true',
                'CUDA_VISIBLE_DEVICES': '0',
                'TARGET_MODEL': target_model
            },
            output_path=paths['output_model'],
            hyperparameters={
                'per_device_train_batch_size': 4,
                'gradient_accumulation_steps': 8,
                'dataloader_num_workers': 2,
                'max_grad_norm': 1.0,
                'bf16': True,
                'gradient_checkpointing': True,
                'problem_type': problem_type,
                'num_labels': num_labels,
                'TARGET_MODEL': target_model
            }
        )

        # Set up tuner with random search strategy for testing
        if problem_type == 'regression':
            metric_definitions = [
                {'Name': 'eval_loss', 'Regex': r'"eval_loss": ([\d\.]+)'},
                {'Name': 'eval_mse', 'Regex': r'"eval_mse": ([\d\.]+)'},
                {'Name': 'eval_mae', 'Regex': r'"eval_mae": ([\d\.]+)'}
            ]
            objective_metric_name = 'eval_mse'
        else:  # classification
            metric_definitions = [
                {'Name': 'eval_loss', 'Regex': r'"eval_loss": ([\d\.]+)'},
                {'Name': 'eval_accuracy', 'Regex': r'"eval_accuracy": ([\d\.]+)'},
                {'Name': 'eval_f1', 'Regex': r'"eval_f1": ([\d\.]+)'},
                {'Name': 'eval_precision', 'Regex': r'"eval_precision": ([\d\.]+)'},
                {'Name': 'eval_recall', 'Regex': r'"eval_recall": ([\d\.]+)'}
            ]
            objective_metric_name = 'eval_f1'
        
        logger.info(f"Using objective metric: {objective_metric_name}")
        logger.info(f"Metric definitions: {metric_definitions}")

        # Use Random strategy for testing (allows limiting jobs)
        strategy = 'Random' if IS_TESTING else 'Bayesian'
        max_jobs = 2 if IS_TESTING else 15

        logger.info(f"Running in {'testing' if IS_TESTING else 'production'} mode")
        logger.info(f"Using {strategy} strategy with {max_jobs} jobs")

        tuner = HyperparameterTuner(
            estimator=huggingface_estimator,
            objective_metric_name=objective_metric_name,
            hyperparameter_ranges=HYPERPARAMETER_RANGES,
            metric_definitions=metric_definitions,
            max_jobs=max_jobs,
            max_parallel_jobs=1,
            strategy=strategy
        )

        # Set up inputs
        inputs = {
            'train': sagemaker.inputs.TrainingInput(
                s3_data=paths['train_manifest'],
                content_type='application/json',
                input_mode='File'
            ),
            'validation': sagemaker.inputs.TrainingInput(
                s3_data=paths['val_manifest'],
                content_type='application/json',
                input_mode='File'
            ),
            'model': sagemaker.inputs.TrainingInput(
                s3_data=paths['original_model'],
                content_type='application/x-tar',
                input_mode='File'
            )
        }

        # Create a shortened version of model name for the job name
        model_prefix = {
            'length-of-stay': 'los',
            'mortality': 'mort',
            'readmission': 'readm',
            'discharge_disposition': 'disch',
            'care_escalation': 'care'
        }.get(target_model, target_model[:4])
        
        # Start tuning job with shortened but dynamic name
        tuning_job_name = f"{model_prefix}-tune-{time.strftime('%m%d%H%M')}"
        tuner.fit(inputs=inputs, job_name=tuning_job_name)
        logger.info(f"Started hyperparameter tuning job: {tuning_job_name}")
        
        # Get the best training job
        best_training_job = tuner.best_training_job()
        logger.info(f"Best training job: {best_training_job}")
        
        # Get hyperparameters from the best training job
        sm_client = boto3.client('sagemaker')
        best_job = sm_client.describe_training_job(TrainingJobName=best_training_job)
        best_hyperparameters = best_job['HyperParameters']
        logger.info(f"Best hyperparameters: {best_hyperparameters}")
        
        # Create a new estimator with the best hyperparameters
        best_estimator = HuggingFace(
            entry_point='train.py',
            source_dir='code',
            instance_type='ml.g5.4xlarge',
            instance_count=1,
            role=role,
            transformers_version='4.28.1',
            pytorch_version='2.0.0',
            py_version='py310',
            hyperparameters=best_hyperparameters,
            environment=huggingface_estimator.environment,
            output_path=paths['output_model']
        )
        
        logger.info("Training job completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()
