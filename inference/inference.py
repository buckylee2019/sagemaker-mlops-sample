import torch
import torch.nn as nn
import numpy as np
import json
import os
import logging
from io import StringIO

logger = logging.getLogger(__name__)

class Autoencoder(nn.Module):
    """PyTorch Autoencoder matching the training code architecture exactly"""
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
            nn.Sigmoid()  # CRITICAL: This matches the training code
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def model_fn(model_dir):
    """Load model for inference - matches training code structure exactly"""
    logger.info(f"Loading model from {model_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load the model checkpoint (matches training code save format)
        model_path = os.path.join(model_dir, 'model.pth')
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model parameters from checkpoint
        input_dim = checkpoint['input_dim']
        encoding_dim = checkpoint['encoding_dim']
        dropout_rate = checkpoint['dropout_rate']
        threshold = checkpoint['threshold']
        
        # Initialize model with same architecture as training
        model = Autoencoder(input_dim, encoding_dim, dropout_rate)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Encoding dim: {encoding_dim}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(f"  Threshold: {threshold}")
        
        return {
            'model': model,
            'device': device,
            'threshold': threshold,
            'input_dim': input_dim,
            'encoding_dim': encoding_dim,
            'dropout_rate': dropout_rate
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Fallback: create a basic model for testing
        logger.warning("Creating fallback model for testing")
        
        input_dim = 64  # Default assumption
        model = Autoencoder(input_dim)
        model.to(device)
        model.eval()
        
        return {
            'model': model,
            'device': device,
            'threshold': 0.1,  # Default threshold
            'input_dim': input_dim,
            'encoding_dim': 32,
            'dropout_rate': 0.2
        }

def input_fn(request_body, request_content_type):
    """Parse input data for inference - matches training code format"""
    logger.info(f"Input content type: {request_content_type}")
    
    if request_content_type == 'text/csv':
        # Parse CSV input (matches training code)
        import pandas as pd
        data = pd.read_csv(StringIO(request_body), header=None)
        return torch.FloatTensor(data.values)
    
    elif request_content_type == 'application/json':
        # Parse JSON input
        input_data = json.loads(request_body)
        
        # Handle different input formats
        if 'instances' in input_data:
            # Batch prediction format
            data = np.array(input_data['instances'])
        elif 'customer_data' in input_data:
            # Single customer format (from SQS)
            data = np.array([input_data['customer_data']])
        else:
            # Direct array format
            data = np.array(input_data)
            if data.ndim == 1:
                data = data.reshape(1, -1)
        
        return torch.FloatTensor(data)
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Make predictions"""
    model = model_artifacts['model']
    device = model_artifacts['device']
    threshold = model_artifacts['threshold']
    
    logger.info(f"Making predictions for {input_data.shape[0]} samples")
    
    # Move data to the same device as model
    input_data = input_data.to(device)
    
    # Make predictions
    with torch.no_grad():
        reconstructed = model(input_data)
        
        # Calculate reconstruction error (matches training code exactly)
        reconstruction_error = torch.mean((input_data - reconstructed) ** 2, dim=1)
        reconstruction_errors = reconstruction_error.cpu().numpy()
        
        # Determine anomalies
        is_anomaly = reconstruction_errors > threshold
        anomaly_scores = reconstruction_errors / threshold  # Normalized score
    
    # Prepare results
    results = []
    for i in range(len(input_data)):
        results.append({
            'reconstruction_error': float(reconstruction_errors[i]),
            'anomaly_score': float(anomaly_scores[i]),
            'is_anomaly': bool(is_anomaly[i]),
            'threshold': float(threshold),
            'input_shape': list(input_data[i].shape)
        })
    
    # Return single result if single input, otherwise return list
    return results[0] if len(results) == 1 else results

def output_fn(prediction, accept):
    """Format the output"""
    logger.info(f"Output accept type: {accept}")
    
    if accept == 'application/json':
        return json.dumps({
            'predictions': prediction if isinstance(prediction, list) else [prediction],
            'model_type': 'pytorch_autoencoder_anomaly_detection'
        }), accept
    
    elif accept == 'text/csv':
        # Return CSV format: reconstruction_error,anomaly_score,is_anomaly
        predictions = prediction if isinstance(prediction, list) else [prediction]
        output_lines = ['reconstruction_error,anomaly_score,is_anomaly']
        for pred in predictions:
            line = f"{pred['reconstruction_error']},{pred['anomaly_score']},{pred['is_anomaly']}"
            output_lines.append(line)
        return '\n'.join(output_lines), accept
    
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
