#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os
import mlflow
from time import gmtime, strftime
from sklearn.preprocessing import StandardScaler
import boto3

def preprocess_autoencoder(
    input_data_s3_path,
    output_s3_prefix,
    tracking_server_arn,
    experiment_name,
    pipeline_run_name=None,
):
    """
    Preprocess data for autoencoder training - unsupervised learning approach
    """
    
    # Set up MLflow
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)
    
    run_name = f"preprocess-autoencoder-{strftime('%d-%H-%M-%S', gmtime())}"
    if pipeline_run_name:
        run_name = f"preprocess-{pipeline_run_name}"
    
    with mlflow.start_run(run_name=run_name, description="Data preprocessing for autoencoder") as run:
        
        # Download and load data
        print(f"Loading data from {input_data_s3_path}")
        
        # Extract bucket and key from S3 path
        s3_parts = input_data_s3_path.replace("s3://", "").split("/", 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        # Download file locally
        s3_client = boto3.client('s3')
        local_file = '/tmp/input_data.csv'
        s3_client.download_file(bucket, key, local_file)
        
        # Load data
        df_raw = pd.read_csv(local_file, sep=";")
        print(f"Original data shape: {df_raw.shape}")
        
        # Feature engineering (same as before but we'll use all features for reconstruction)
        df_data = df_raw.copy()
        df_data["no_previous_contact"] = np.where(df_data["pdays"] == 999, 1, 0)
        df_data["not_working"] = np.where(
            np.in1d(df_data["job"], ["student", "retired", "unemployed"]), 1, 0
        )

        # Remove unnecessary data but keep more features for autoencoder
        df_model_data = df_data.drop(
            ["duration", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
            axis=1,
        )

        # Age binning
        bins = [18, 30, 40, 50, 60, 70, 90]
        labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-plus']
        df_model_data['age_range'] = pd.cut(df_model_data.age, bins, labels=labels, include_lowest=True)
        df_model_data = pd.concat([df_model_data, pd.get_dummies(df_model_data['age_range'], prefix='age', dtype=int)], axis=1)
        df_model_data.drop('age', axis=1, inplace=True)
        df_model_data.drop('age_range', axis=1, inplace=True)

        # Scale numerical features
        scaled_features = ['pdays', 'previous', 'campaign']
        scaler = StandardScaler()
        df_model_data[scaled_features] = scaler.fit_transform(df_model_data[scaled_features])

        # Convert categorical variables to dummy variables
        df_model_data = pd.get_dummies(df_model_data, dtype=int)

        # For autoencoder, we'll separate the target for evaluation but not use it in training
        target_col = "y"
        if 'y_yes' in df_model_data.columns and 'y_no' in df_model_data.columns:
            # Keep target for anomaly evaluation
            target_data = df_model_data["y_yes"].copy()
            # Remove target columns from features for unsupervised learning
            feature_data = df_model_data.drop(["y_no", "y_yes"], axis=1)
        else:
            target_data = None
            feature_data = df_model_data
        
        print(f"Feature data shape after processing: {feature_data.shape}")
        
        # For autoencoder, we typically use normal data for training and test on both normal and anomalous
        # Split data: 70% train, 15% validation, 15% test
        train_size = int(0.7 * len(feature_data))
        val_size = int(0.15 * len(feature_data))
        
        # Shuffle data
        shuffled_indices = np.random.permutation(len(feature_data))
        feature_data_shuffled = feature_data.iloc[shuffled_indices].reset_index(drop=True)
        if target_data is not None:
            target_data_shuffled = target_data.iloc[shuffled_indices].reset_index(drop=True)
        
        # Split features
        train_features = feature_data_shuffled[:train_size]
        val_features = feature_data_shuffled[train_size:train_size + val_size]
        test_features = feature_data_shuffled[train_size + val_size:]
        
        # Split targets (for evaluation)
        if target_data is not None:
            train_targets = target_data_shuffled[:train_size]
            val_targets = target_data_shuffled[train_size:train_size + val_size]
            test_targets = target_data_shuffled[train_size + val_size:]
        
        print(f"Data split > train:{train_features.shape} | validation:{val_features.shape} | test:{test_features.shape}")
        
        # Log parameters to MLflow
        mlflow.log_params({
            "train_features": train_features.shape,
            "val_features": val_features.shape,
            "test_features": test_features.shape,
            "total_features": feature_data.shape[1]
        })

        mlflow.set_tags({
            'mlflow.source.type': 'JOB',
            'model_type': 'autoencoder',
            'step': 'preprocessing'
        })
        
        # Upload datasets to S3
        s3_client = boto3.client('s3')
        
        # Extract bucket from output prefix
        output_bucket = output_s3_prefix.replace("s3://", "").split("/")[0]
        output_prefix = "/".join(output_s3_prefix.replace("s3://", "").split("/")[1:])
        
        # Save and upload train data
        train_local = '/tmp/train.csv'
        train_features.to_csv(train_local, index=False, header=False)
        train_key = f"{output_prefix}/train/train.csv"
        s3_client.upload_file(train_local, output_bucket, train_key)
        train_s3_url = f"s3://{output_bucket}/{train_key}"
        
        # Save and upload validation data
        val_local = '/tmp/validation.csv'
        val_features.to_csv(val_local, index=False, header=False)
        val_key = f"{output_prefix}/validation/validation.csv"
        s3_client.upload_file(val_local, output_bucket, val_key)
        validation_s3_url = f"s3://{output_bucket}/{val_key}"
        
        # Save and upload test features
        test_x_local = '/tmp/test_features.csv'
        test_features.to_csv(test_x_local, index=False, header=False)
        test_x_key = f"{output_prefix}/test/test_features.csv"
        s3_client.upload_file(test_x_local, output_bucket, test_x_key)
        test_x_s3_url = f"s3://{output_bucket}/{test_x_key}"
        
        # Save and upload test targets (for evaluation)
        if target_data is not None:
            test_y_local = '/tmp/test_targets.csv'
            test_targets.to_csv(test_y_local, index=False, header=False)
            test_y_key = f"{output_prefix}/test/test_targets.csv"
            s3_client.upload_file(test_y_local, output_bucket, test_y_key)
            test_y_s3_url = f"s3://{output_bucket}/{test_y_key}"
        else:
            test_y_s3_url = None
        
        # Save and upload baseline data
        baseline_local = '/tmp/baseline.csv'
        feature_data.to_csv(baseline_local, index=False, header=False)
        baseline_key = f"{output_prefix}/baseline/baseline.csv"
        s3_client.upload_file(baseline_local, output_bucket, baseline_key)
        baseline_s3_url = f"s3://{output_bucket}/{baseline_key}"
        
        # Log artifacts to MLflow
        mlflow.log_artifact(baseline_local, "baseline")
        
        print("## Processing complete.")
        
        return {
            'train_data': train_s3_url,
            'validation_data': validation_s3_url,
            'test_x_data': test_x_s3_url,
            'test_y_data': test_y_s3_url,
            'baseline_data': baseline_s3_url,
            'experiment_name': experiment_name,
            'pipeline_run_id': pipeline_run_name or run.info.run_id
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-s3-path', type=str, required=True)
    parser.add_argument('--output-s3-prefix', type=str, required=True)
    parser.add_argument('--tracking-server-arn', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--pipeline-run-name', type=str, default=None)
    
    args = parser.parse_args()
    
    result = preprocess_autoencoder(
        input_data_s3_path=args.input_data_s3_path,
        output_s3_prefix=args.output_s3_prefix,
        tracking_server_arn=args.tracking_server_arn,
        experiment_name=args.experiment_name,
        pipeline_run_name=args.pipeline_run_name
    )
    
    print(f"Preprocessing result: {result}")
