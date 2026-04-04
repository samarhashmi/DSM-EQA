import torch
import torch.nn as nn
from typing import Callable

class LossFunctions:
    """Collection of loss functions for model training."""
    
    @staticmethod
    def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error loss."""
        return nn.MSELoss()(predictions, targets)
    
    @staticmethod
    def mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Error loss."""
        return nn.L1Loss()(predictions, targets)
    
    @staticmethod
    def smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Smooth L1 loss."""
        return nn.SmoothL1Loss()(predictions, targets)
    
    @staticmethod
    def huber_loss(predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Huber loss."""
        return nn.HuberLoss(delta=delta)(predictions, targets)
    
    @staticmethod
    def get_loss_function(loss_type: str) -> Callable:
        """
        Get loss function by type.
        
        Args:
            loss_type: Type of loss function
            
        Returns:
            Loss function callable
        """
        loss_map = {
            'mse': LossFunctions.mse_loss,
            'mae': LossFunctions.mae_loss,
            'smooth_l1': LossFunctions.smooth_l1_loss,
            'huber': LossFunctions.huber_loss
        }
        
        if loss_type not in loss_map:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss_map[loss_type]
