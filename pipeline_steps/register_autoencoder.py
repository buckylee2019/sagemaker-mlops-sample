#!/usr/bin/env python3

import boto3
import json
import mlflow
from time import gmtime, strftime
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.drift_check_baselines import DriftCheckBaselines

def register_autoencoder(
    training_job_name,
    model_package_group_name,
    model_approval_status,
    evaluation_result,
    output_s3_prefix,
    tracking_server_arn,
    experiment_name,
    pipeline_run_id=None,
    model_statistics_s3_path=None,
    model_constraints_s3_path=None,
    model_data_statistics_s3_path=None,
    model_data_constraints_s3_path=None,
):
    """
    Register autoencoder model in SageMaker Model Registry
    """
    
    # Set up MLflow
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)
    
    run_name = f"register-autoencoder-{strftime('%d-%H-%M-%S', gmtime())}"
    if pipeline_run_id:
        run_name = f"register-{pipeline_run_id}"
    
    with mlflow.start_run(run_name=run_name, description="Autoencoder model registration") as run:
        
        # Get SageMaker client
        sm_client = boto3.client('sagemaker')
        
        # Ensure Model Package Group exists with proper tags
        try:
            # Check if model package group exists
            sm_client.describe_model_package_group(ModelPackageGroupName=model_package_group_name)
            print(f"Model Package Group {model_package_group_name} already exists")
        except sm_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                # Model Package Group doesn't exist, create it with tags
                print(f"Creating Model Package Group: {model_package_group_name}")
                sm_client.create_model_package_group(
                    ModelPackageGroupName=model_package_group_name,
                    ModelPackageGroupDescription=f"PyTorch autoencoder models for anomaly detection",
                    Tags=[
                        {"Key": "ModelType", "Value": "Autoencoder"},
                        {"Key": "Framework", "Value": "PyTorch"},
                        {"Key": "UseCase", "Value": "AnomalyDetection"},
                        {"Key": "Project", "Value": "from-idea-to-prod"}
                    ]
                )
            else:
                raise e
        
        # Get training job details
        training_job = sm_client.describe_training_job(TrainingJobName=training_job_name)
        model_data_url = training_job['ModelArtifacts']['S3ModelArtifacts']
        
        # Create model metrics
        model_metrics = None
        if evaluation_result:
            # Save evaluation metrics to S3
            s3_client = boto3.client('s3')
            output_bucket = output_s3_prefix.replace("s3://", "").split("/")[0]
            output_prefix = "/".join(output_s3_prefix.replace("s3://", "").split("/")[1:])
            
            metrics_local = '/tmp/model_metrics.json'
            with open(metrics_local, 'w') as f:
                json.dump(evaluation_result, f, indent=2)
            
            metrics_key = f"{output_prefix}/model_metrics/model_metrics.json"
            s3_client.upload_file(metrics_local, output_bucket, metrics_key)
            metrics_s3_url = f"s3://{output_bucket}/{metrics_key}"
            
            model_metrics = ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=metrics_s3_url,
                    content_type="application/json"
                )
            )
        
        # Create drift check baselines if provided
        drift_check_baselines = None
        if any([model_statistics_s3_path, model_constraints_s3_path, 
                model_data_statistics_s3_path, model_data_constraints_s3_path]):
            drift_check_baselines = DriftCheckBaselines(
                model_statistics=MetricsSource(
                    s3_uri=model_statistics_s3_path,
                    content_type="application/json"
                ) if model_statistics_s3_path else None,
                model_constraints=MetricsSource(
                    s3_uri=model_constraints_s3_path,
                    content_type="application/json"
                ) if model_constraints_s3_path else None,
                model_data_statistics=MetricsSource(
                    s3_uri=model_data_statistics_s3_path,
                    content_type="application/json"
                ) if model_data_statistics_s3_path else None,
                model_data_constraints=MetricsSource(
                    s3_uri=model_data_constraints_s3_path,
                    content_type="application/json"
                ) if model_data_constraints_s3_path else None,
            )
        
        # Get execution role
        execution_role = training_job['RoleArn']
        
        # Get container image
        container_image = training_job['AlgorithmSpecification']['TrainingImage']
        
        # Create model package (without tags - tags go on the group, not individual versions)
        model_package_input_dict = {
            "ModelPackageGroupName": model_package_group_name,
            "ModelPackageDescription": f"PyTorch autoencoder for anomaly detection. Training job: {training_job_name}",
            "ModelApprovalStatus": model_approval_status,
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": container_image,
                        "ModelDataUrl": model_data_url,
                        "Framework": "PYTORCH",
                        "FrameworkVersion": "1.12"
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": [
                    "ml.t2.medium",
                    "ml.m5.large",
                    "ml.m5.xlarge"
                ],
                "SupportedTransformInstanceTypes": [
                    "ml.m5.large",
                    "ml.m5.xlarge"
                ]
            }
            # Note: Tags removed - they should be on the Model Package Group, not individual versions
        }
        
        # Add model metrics if available
        if model_metrics:
            model_package_input_dict["ModelMetrics"] = {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": metrics_s3_url
                    }
                }
            }
        
        # Add drift check baselines if available
        if drift_check_baselines:
            model_package_input_dict["DriftCheckBaselines"] = drift_check_baselines.to_request()
        
        # Create model package
        try:
            response = sm_client.create_model_package(**model_package_input_dict)
            model_package_arn = response['ModelPackageArn']
            
            print(f"✅ Model package created: {model_package_arn}")
            
            # Log to MLflow
            mlflow.log_params({
                "model_package_group_name": model_package_group_name,
                "model_approval_status": model_approval_status,
                "training_job_name": training_job_name
            })
            
            if evaluation_result and 'anomaly_detection_metrics' in evaluation_result:
                metrics = evaluation_result['anomaly_detection_metrics']
                mlflow.log_metrics({
                    "registered_model_roc_auc": metrics.get('roc_auc', {}).get('value', 0),
                    "registered_model_pr_auc": metrics.get('pr_auc', {}).get('value', 0),
                    "registered_model_f1_score": metrics.get('max_f1_score', {}).get('value', 0)
                })
            
            mlflow.set_tags({
                'mlflow.source.type': 'JOB',
                'model_type': 'autoencoder',
                'step': 'registration',
                'model_package_arn': model_package_arn
            })
            
            return {
                'model_package_arn': model_package_arn,
                'model_package_group_name': model_package_group_name,
                'model_approval_status': model_approval_status
            }
            
        except Exception as e:
            print(f"❌ Error creating model package: {str(e)}")
            raise e

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-job-name', type=str, required=True)
    parser.add_argument('--model-package-group-name', type=str, required=True)
    parser.add_argument('--model-approval-status', type=str, required=True)
    parser.add_argument('--evaluation-result', type=str, required=True)
    parser.add_argument('--output-s3-prefix', type=str, required=True)
    parser.add_argument('--tracking-server-arn', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--pipeline-run-id', type=str, default=None)
    
    args = parser.parse_args()
    
    # Parse evaluation result from JSON string
    evaluation_result = json.loads(args.evaluation_result)
    
    result = register_autoencoder(
        training_job_name=args.training_job_name,
        model_package_group_name=args.model_package_group_name,
        model_approval_status=args.model_approval_status,
        evaluation_result=evaluation_result,
        output_s3_prefix=args.output_s3_prefix,
        tracking_server_arn=args.tracking_server_arn,
        experiment_name=args.experiment_name,
        pipeline_run_id=args.pipeline_run_id
    )
    
    print(f"Registration result: {result}")
