#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import boto3
import tarfile
import os
import json
import mlflow
from time import gmtime, strftime
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
import io

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32, dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_autoencoder_model(model_s3_path):
    """Load autoencoder model from S3"""
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    s3_parts = model_s3_path.replace("s3://", "").split("/", 1)
    bucket = s3_parts[0]
    key = s3_parts[1]
    
    # Download model artifacts
    local_model_path = '/tmp/model.tar.gz'
    s3_client.download_file(bucket, key, local_model_path)
    
    # Extract model
    extract_path = '/tmp/model'
    os.makedirs(extract_path, exist_ok=True)
    with tarfile.open(local_model_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    
    # Load model checkpoint - Fix for PyTorch 2.6+ weights_only issue
    checkpoint = torch.load(os.path.join(extract_path, 'model.pth'), map_location='cpu', weights_only=False)
    
    # Create and load model
    model = Autoencoder(
        checkpoint['input_dim'],
        checkpoint['encoding_dim'],
        checkpoint['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def evaluate_autoencoder(
    test_x_data_s3_path,
    test_y_data_s3_path,
    model_s3_path,
    output_s3_prefix,
    tracking_server_arn,
    experiment_name,
    pipeline_run_id=None,
):
    """
    Evaluate autoencoder model for anomaly detection
    """
    
    # Set up MLflow
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)
    
    run_name = f"evaluate-autoencoder-{strftime('%d-%H-%M-%S', gmtime())}"
    if pipeline_run_id:
        run_name = f"evaluate-{pipeline_run_id}"
    
    with mlflow.start_run(run_name=run_name, description="Autoencoder model evaluation") as run:
        
        # Load test data
        s3_client = boto3.client('s3')
        
        # Load test features
        test_x_parts = test_x_data_s3_path.replace("s3://", "").split("/", 1)
        test_x_local = '/tmp/test_features.csv'
        s3_client.download_file(test_x_parts[0], test_x_parts[1], test_x_local)
        test_features = pd.read_csv(test_x_local, header=None)
        
        # Load test targets
        test_y_parts = test_y_data_s3_path.replace("s3://", "").split("/", 1)
        test_y_local = '/tmp/test_targets.csv'
        s3_client.download_file(test_y_parts[0], test_y_parts[1], test_y_local)
        test_targets = pd.read_csv(test_y_local, header=None)[0].values
        
        print(f"Loaded test data: {test_features.shape} features, {len(test_targets)} targets")
        
        # Load model
        model, checkpoint = load_autoencoder_model(model_s3_path)
        threshold = checkpoint['threshold']
        
        print(f"Loaded model with threshold: {threshold}")
        
        # Make predictions
        test_tensor = torch.FloatTensor(test_features.values)
        
        with torch.no_grad():
            reconstructed = model(test_tensor)
            reconstruction_errors = torch.mean((test_tensor - reconstructed) ** 2, dim=1).numpy()
        
        # Calculate metrics
        precision, recall, pr_thresholds = precision_recall_curve(test_targets, reconstruction_errors)
        pr_auc = auc(recall, precision)
        
        fpr, tpr, roc_thresholds = roc_curve(test_targets, reconstruction_errors)
        roc_auc = auc(fpr, tpr)
        
        # Calculate F1 scores and find optimal threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = pr_thresholds[optimal_idx]
        max_f1_score = np.max(f1_scores)
        
        # Predictions using optimal threshold
        predictions = (reconstruction_errors > optimal_threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((test_targets == 1) & (predictions == 1))
        fp = np.sum((test_targets == 0) & (predictions == 1))
        tn = np.sum((test_targets == 0) & (predictions == 0))
        fn = np.sum((test_targets == 1) & (predictions == 0))
        
        # Calculate additional metrics
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        
        # Create evaluation results
        evaluation_result = {
            "anomaly_detection_metrics": {
                "roc_auc": {"value": float(roc_auc)},
                "pr_auc": {"value": float(pr_auc)},
                "optimal_threshold": {"value": float(optimal_threshold)},
                "max_f1_score": {"value": float(max_f1_score)},
                "precision": {"value": float(precision_score)},
                "recall": {"value": float(recall_score)},
                "accuracy": {"value": float(accuracy)},
                "true_positives": {"value": int(tp)},
                "false_positives": {"value": int(fp)},
                "true_negatives": {"value": int(tn)},
                "false_negatives": {"value": int(fn)},
                "mean_reconstruction_error": {"value": float(np.mean(reconstruction_errors))},
                "std_reconstruction_error": {"value": float(np.std(reconstruction_errors))}
            }
        }
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": optimal_threshold,
            "max_f1_score": max_f1_score,
            "precision": precision_score,
            "recall": recall_score,
            "accuracy": accuracy,
            "mean_reconstruction_error": np.mean(reconstruction_errors),
            "std_reconstruction_error": np.std(reconstruction_errors)
        })
        
        mlflow.set_tags({
            'mlflow.source.type': 'JOB',
            'model_type': 'autoencoder',
            'step': 'evaluation'
        })
        
        # Create prediction baseline for monitoring
        prediction_baseline = pd.DataFrame({
            'prediction': predictions,
            'probability': reconstruction_errors,
            'label': test_targets
        })
        
        # Save prediction baseline
        baseline_local = '/tmp/prediction_baseline.csv'
        prediction_baseline.to_csv(baseline_local, index=False)
        
        # Upload to S3
        output_bucket = output_s3_prefix.replace("s3://", "").split("/")[0]
        output_prefix = "/".join(output_s3_prefix.replace("s3://", "").split("/")[1:])
        
        baseline_key = f"{output_prefix}/prediction_baseline/prediction_baseline.csv"
        s3_client.upload_file(baseline_local, output_bucket, baseline_key)
        prediction_baseline_s3_url = f"s3://{output_bucket}/{baseline_key}"
        
        # Save evaluation results
        eval_results_local = '/tmp/evaluation.json'
        with open(eval_results_local, 'w') as f:
            json.dump(evaluation_result, f, indent=2)
        
        eval_key = f"{output_prefix}/evaluation/evaluation.json"
        s3_client.upload_file(eval_results_local, output_bucket, eval_key)
        
        # Log artifacts
        mlflow.log_artifact(baseline_local, "prediction_baseline")
        mlflow.log_artifact(eval_results_local, "evaluation")
        
        print(f"Evaluation completed. ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        
        return {
            'evaluation_result': evaluation_result,
            'prediction_baseline_data': prediction_baseline_s3_url
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-x-data-s3-path', type=str, required=True)
    parser.add_argument('--test-y-data-s3-path', type=str, required=True)
    parser.add_argument('--model-s3-path', type=str, required=True)
    parser.add_argument('--output-s3-prefix', type=str, required=True)
    parser.add_argument('--tracking-server-arn', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--pipeline-run-id', type=str, default=None)
    
    args = parser.parse_args()
    
    result = evaluate_autoencoder(
        test_x_data_s3_path=args.test_x_data_s3_path,
        test_y_data_s3_path=args.test_y_data_s3_path,
        model_s3_path=args.model_s3_path,
        output_s3_prefix=args.output_s3_prefix,
        tracking_server_arn=args.tracking_server_arn,
        experiment_name=args.experiment_name,
        pipeline_run_id=args.pipeline_run_id
    )
    
    print(f"Evaluation result: {result}")
