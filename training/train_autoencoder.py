import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from time import gmtime, strftime
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def encode(self, x):
        return self.encoder(x)

def load_data(data_path):
    """Load data from CSV file"""
    data = pd.read_csv(data_path, header=None)
    return torch.FloatTensor(data.values)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the autoencoder model"""
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data[0].to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data[0].to(device)
                reconstructed = model(batch_data)
                loss = criterion(reconstructed, batch_data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)
    
    return train_losses, val_losses

def calculate_reconstruction_threshold(model, val_loader, device, percentile=95):
    """Calculate reconstruction error threshold for anomaly detection"""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data[0].to(device)
            reconstructed = model(batch_data)
            mse = torch.mean((batch_data - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(mse.cpu().numpy())
    
    threshold = np.percentile(reconstruction_errors, percentile)
    return threshold, reconstruction_errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--encoding_dim', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # SageMaker specific arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Set up MLflow
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_ARN'))
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME')
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    suffix = strftime('%d-%H-%M-%S', gmtime())
    user_profile_name = os.getenv('USER', 'sagemaker')
    region = os.getenv('REGION', 'us-east-1')
    
    with mlflow.start_run(
        run_name=f"autoencoder-training-{suffix}",
        description="PyTorch autoencoder training in SageMaker container"
    ) as run:
        
        # Log parameters
        mlflow.log_params({
            'encoding_dim': args.encoding_dim,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'weight_decay': args.weight_decay,
            'device': str(device)
        })
        
        mlflow.set_tags({
            'mlflow.user': user_profile_name,
            'mlflow.source.type': 'JOB',
            'model_type': 'autoencoder',
            'framework': 'pytorch'
        })
        
        # Load data
        logger.info("Loading training data...")
        train_data = load_data(os.path.join(args.train, 'train.csv'))
        val_data = load_data(os.path.join(args.validation, 'validation.csv'))
        
        input_dim = train_data.shape[1]
        logger.info(f"Input dimension: {input_dim}")
        
        # Create data loaders
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = Autoencoder(input_dim, args.encoding_dim, args.dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        logger.info(f"Model architecture: {model}")
        
        # Train model
        logger.info("Starting training...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device
        )
        
        # Calculate reconstruction threshold
        logger.info("Calculating reconstruction threshold...")
        threshold, reconstruction_errors = calculate_reconstruction_threshold(model, val_loader, device)
        
        # Log final metrics
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        
        mlflow.log_metrics({
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'reconstruction_threshold': threshold,
            'mean_reconstruction_error': np.mean(reconstruction_errors),
            'std_reconstruction_error': np.std(reconstruction_errors)
        })
        
        logger.info(f"Training completed. Final train loss: {final_train_loss:.6f}, Final val loss: {final_val_loss:.6f}")
        logger.info(f"Reconstruction threshold (95th percentile): {threshold:.6f}")
        
        # Save model
        model_path = os.path.join(args.model_dir, 'model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'encoding_dim': args.encoding_dim,
            'dropout_rate': args.dropout_rate,
            'threshold': threshold,
            'train_loss': final_train_loss,
            'val_loss': final_val_loss
        }, model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(
            model, 
            "model",
            extra_files=[model_path]
        )
        
        logger.info(f"Model saved to {model_path}")

def model_fn(model_dir):
    """Load model for inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.join(model_dir, 'model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = Autoencoder(
        checkpoint['input_dim'], 
        checkpoint['encoding_dim'], 
        checkpoint['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def input_fn(request_body, request_content_type):
    """Parse input data for inference"""
    if request_content_type == 'text/csv':
        data = pd.read_csv(io.StringIO(request_body), header=None)
        return torch.FloatTensor(data.values)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    with torch.no_grad():
        reconstructed = model(input_data)
        reconstruction_error = torch.mean((input_data - reconstructed) ** 2, dim=1)
    
    return {
        'reconstructed': reconstructed.cpu().numpy(),
        'reconstruction_error': reconstruction_error.cpu().numpy()
    }

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps({
            'reconstruction_errors': prediction['reconstruction_error'].tolist()
        })
    else:
        return str(prediction)
