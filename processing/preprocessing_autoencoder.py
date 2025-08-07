from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import argparse
import os
import mlflow
from time import gmtime, strftime

user_profile_name = os.getenv('USER', 'sagemaker')

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='bank-additional-full.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    parser.add_argument('--mlflow_tracking_arn', type=str, required=True)
    parser.add_argument('--mlflow_run_id', type=str, required=True)
    return parser.parse_args()

def process_data_for_autoencoder(df_data):
    """Process data for autoencoder training - unsupervised learning"""
    
    print(f"Original data shape: {df_data.shape}")
    
    # Feature engineering
    df_data["no_previous_contact"] = np.where(df_data["pdays"] == 999, 1, 0)
    df_data["not_working"] = np.where(
        np.in1d(df_data["job"], ["student", "retired", "unemployed"]), 1, 0
    )

    # Drop unnecessary columns
    df_model_data = df_data.drop(
        ["duration", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
        axis=1,
    )

    # Age binning
    bins = [18, 30, 40, 50, 60, 70, 90]
    labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-plus']
    df_model_data['age_range'] = pd.cut(df_model_data.age, bins, labels=labels, include_lowest=True)
    df_model_data = pd.concat([df_model_data, pd.get_dummies(df_model_data['age_range'], prefix='age', dtype=int)], axis=1)
    df_model_data.drop(['age', 'age_range'], axis=1, inplace=True)

    # Scale numerical features
    scaled_features = ['pdays', 'previous', 'campaign']
    scaler = StandardScaler()
    df_model_data[scaled_features] = scaler.fit_transform(df_model_data[scaled_features])

    # Convert categorical variables to dummies
    df_model_data = pd.get_dummies(df_model_data, dtype=int)

    # Separate target
    target_col = "y"
    if 'y_yes' in df_model_data.columns and 'y_no' in df_model_data.columns:
        target_data = df_model_data["y_yes"].copy()
        feature_data = df_model_data.drop(["y_no", "y_yes"], axis=1)
    else:
        target_data = None
        feature_data = df_model_data
    
    print(f"Feature data shape after processing: {feature_data.shape}")
    return feature_data, target_data, scaler

if __name__ == "__main__":
    args = _parse_args()
    
    # Configure MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_arn)
    mlflow.autolog()

    with mlflow.start_run(run_id=args.mlflow_run_id) as run:
        # Read CSV
        df_raw = pd.read_csv(os.path.join(args.filepath, args.filename), sep=";")
        feature_data, target_data, scaler = process_data_for_autoencoder(df_raw)
        
        # Split data
        train_size = int(0.7 * len(feature_data))
        val_size = int(0.15 * len(feature_data))
        
        shuffled_indices = np.random.permutation(len(feature_data))
        feature_data_shuffled = feature_data.iloc[shuffled_indices].reset_index(drop=True)
        if target_data is not None:
            target_data_shuffled = target_data.iloc[shuffled_indices].reset_index(drop=True)
        
        train_features = feature_data_shuffled[:train_size]
        val_features = feature_data_shuffled[train_size:train_size + val_size]
        test_features = feature_data_shuffled[train_size + val_size:]
        
        if target_data is not None:
            train_targets = target_data_shuffled[:train_size]
            val_targets = target_data_shuffled[train_size:train_size + val_size]
            test_targets = target_data_shuffled[train_size + val_size:]
        
        print(f"Data split > train:{train_features.shape} | validation:{val_features.shape} | test:{test_features.shape}")
        
        # Log params
        mlflow.log_params({
            "train_features": train_features.shape,
            "val_features": val_features.shape,
            "test_features": test_features.shape,
            "total_features": feature_data.shape[1]
        })

        mlflow.set_tags({
            'mlflow.user': user_profile_name,
            'mlflow.source.type': 'JOB',
            'model_type': 'autoencoder'
        })
        
        # Ensure directories exist
        os.makedirs(os.path.join(args.outputpath, 'train'), exist_ok=True)
        os.makedirs(os.path.join(args.outputpath, 'validation'), exist_ok=True)
        os.makedirs(os.path.join(args.outputpath, 'test'), exist_ok=True)
        os.makedirs(os.path.join(args.outputpath, 'baseline'), exist_ok=True)

        # Save datasets
        train_features.to_csv(os.path.join(args.outputpath, 'train/train.csv'), index=False, header=False)
        val_features.to_csv(os.path.join(args.outputpath, 'validation/validation.csv'), index=False, header=False)
        test_features.to_csv(os.path.join(args.outputpath, 'test/test_features.csv'), index=False, header=False)
        
        if target_data is not None:
            train_targets.to_csv(os.path.join(args.outputpath, 'train/train_targets.csv'), index=False, header=False)
            val_targets.to_csv(os.path.join(args.outputpath, 'validation/val_targets.csv'), index=False, header=False)
            test_targets.to_csv(os.path.join(args.outputpath, 'test/test_targets.csv'), index=False, header=False)
        
        feature_data.to_csv(os.path.join(args.outputpath, 'baseline/baseline.csv'), index=False, header=False)
        
        feature_names = list(feature_data.columns)
        with open(os.path.join(args.outputpath, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(feature_names))
        
        mlflow.log_artifact(local_path=os.path.join(args.outputpath, 'baseline/baseline.csv'))
        mlflow.log_artifact(local_path=os.path.join(args.outputpath, 'feature_names.txt'))
    
    print("## ✅ Processing complete. Exiting.")
