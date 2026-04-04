import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from .metrics import Metrics

class Evaluator:
    """Model evaluator."""
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: Neural network model
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.device = device
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate model on data loader.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        predictions = np.concatenate(all_preds, axis=0).squeeze()
        targets = np.concatenate(all_targets, axis=0).squeeze()
        
        return predictions, targets
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        return Metrics.compute_all_metrics(targets, predictions)
    
    def full_evaluation(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Perform full evaluation.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Dictionary of metrics
        """
        predictions, targets = self.evaluate(data_loader)
        return self.compute_metrics(predictions, targets)
