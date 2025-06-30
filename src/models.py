"""
Models Module for Archaeological Site Detection
Implements supervised CNN and unsupervised autoencoder models (PyTorch only)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerrainDataset(Dataset):
    """PyTorch dataset for terrain data"""
    
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
                 transform=None):
        """
        Initialize dataset
        
        Args:
            features: Feature arrays of shape (N, C, H, W)
            labels: Optional labels of shape (N,)
            transform: Optional transformations
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        if self.labels is not None:
            return feature, self.labels[idx]
        else:
            return feature


class ArchaeologicalCNN(nn.Module):
    """Convolutional Neural Network for archaeological site detection"""
    
    def __init__(self, input_channels: int = 5, num_classes: int = 2, 
                 dropout_rate: float = 0.5):
        """
        Initialize CNN model
        
        Args:
            input_channels: Number of input channels (elevation, slope, aspect, etc.)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(ArchaeologicalCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size after convolutions and pooling
        # Assuming input size of 100x100
        self.flatten_size = 128 * 12 * 12  # After 3 pooling operations
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class ArchaeologicalAutoencoder(nn.Module):
    """Autoencoder for unsupervised anomaly detection"""
    
    def __init__(self, input_channels: int = 5, latent_dim: int = 128):
        """
        Initialize autoencoder
        
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
        """
        super(ArchaeologicalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate flattened size
        self.flatten_size = 256 * 6 * 6  # After 4 pooling operations on 100x100 input
        
        # Latent space
        self.latent = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_size),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        latent = self.latent(encoded)
        
        # Decode
        decoded = self.decoder[0](latent)  # Linear layer
        decoded = self.decoder[1](decoded) # ReLU
        decoded = decoded.view(-1, 256, 6, 6)  # Reshape to match encoder output
        decoded = self.decoder[2:](decoded)  # Pass through ConvTranspose2d layers
        # Upsample to (100, 100) to match input
        decoded = F.interpolate(decoded, size=(100, 100), mode='bilinear', align_corners=False)
        return decoded, latent
    
    def encode(self, x):
        """Encode input to latent space"""
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        return self.latent(encoded)


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model_type: str = 'cnn', device: str = 'cpu'):
        """
        Initialize model trainer
        
        Args:
            model_type: Type of model ('cnn', 'autoencoder')
            device: Device to use for training
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        logger.info(f"Using device: {self.device}")
    
    def create_model(self, input_channels: int = 5, **kwargs):
        """Create model based on type"""
        if self.model_type == 'cnn':
            num_classes = kwargs.get('num_classes', 2)
            self.model = ArchaeologicalCNN(input_channels=input_channels, num_classes=num_classes)
        elif self.model_type == 'autoencoder':
            latent_dim = kwargs.get('latent_dim', 128)
            self.model = ArchaeologicalAutoencoder(input_channels=input_channels, latent_dim=latent_dim)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self.model.to(self.device)
        logger.info(f"Created {self.model_type} model")
    
    def setup_training(self, learning_rate: float = 0.001, **kwargs):
        """Setup optimizer and loss function"""
        if self.model_type == 'autoencoder':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Setup training with learning rate: {learning_rate}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for data in train_loader:
            if self.model_type == 'autoencoder':
                if isinstance(data, (list, tuple)):
                    features = data[0]
                else:
                    features = data
                features = features.to(self.device)
                self.optimizer.zero_grad()
                reconstructed, _ = self.model(features)
                loss = self.criterion(reconstructed, features)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            else:
                features, labels = data
                features = features.to(self.device)
                labels = labels.to(self.device)
                labels = labels.long()  # Ensure labels are Long type for CrossEntropyLoss
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data in val_loader:
                if self.model_type == 'autoencoder':
                    if isinstance(data, (list, tuple)):
                        features = data[0]
                    else:
                        features = data
                    features = features.to(self.device)
                    reconstructed, _ = self.model(features)
                    loss = self.criterion(reconstructed, features)
                    total_loss += loss.item()
                else:
                    features, labels = data
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    labels = labels.long()  # Ensure labels are Long type for CrossEntropyLoss
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.cpu().numpy())
                    targets.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        if self.model_type == 'cnn':
            metrics = self._calculate_metrics(targets, predictions)
        else:
            metrics = {'val_loss': avg_loss}
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, targets: List, predictions: List) -> Dict:
        """Calculate classification metrics"""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        if len(targets) == 0 or len(predictions) == 0:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, patience: int = 10) -> Dict:
        """Train model with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, metrics = self.validate(val_loader)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['metrics'].append(metrics)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'best_{self.model_type}_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Make predictions on test data"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in test_loader:
                if self.model_type == 'autoencoder':
                    if isinstance(data, (list, tuple)):
                        features = data[0]
                    else:
                        features = data
                    features = features.to(self.device)
                    reconstructed, _ = self.model(features)
                    # For autoencoder, return reconstruction error as anomaly score
                    error = torch.mean((features - reconstructed) ** 2, dim=[1, 2, 3])
                    predictions.extend(error.cpu().numpy())
                else:
                    # For CNN, data should be (features, labels) tuple
                    if isinstance(data, (list, tuple)) and len(data) == 2:
                        features, _ = data
                    else:
                        features = data
                    features = features.to(self.device)
                    outputs = self.model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.cpu().numpy())
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type
        }, path)
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from: {path}")


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_scores: Optional[np.ndarray] = None) -> Dict:
    """
    Evaluate model performance with comprehensive metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    
    metrics = {}
    
    # Basic classification metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    })
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # ROC-AUC if scores provided
    if y_scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
            metrics['roc_auc'] = roc_auc
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer(model_type='cnn')
    trainer.create_model(input_channels=5, num_classes=2)
    trainer.setup_training(learning_rate=0.001)
    
    print("Model created successfully!")
    print(f"Model type: {trainer.model_type}")
    print(f"Device: {trainer.device}") 