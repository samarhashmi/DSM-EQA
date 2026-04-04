import torch
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR
from typing import Tuple

class OptimizerFactory:
    """Factory for creating optimizers and schedulers."""
    
    @staticmethod
    def create_optimizer(model_params, optimizer_type: str = 'adam',
                        learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        """
        Create optimizer.
        
        Args:
            model_params: Model parameters
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            weight_decay: Weight decay
            
        Returns:
            Optimizer instance
        """
        if optimizer_type == 'adam':
            return Adam(model_params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return SGD(model_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    @staticmethod
    def create_scheduler(optimizer, scheduler_type: str = 'cosine', epochs: int = 100):
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            scheduler_type: Type of scheduler
            epochs: Total number of epochs
            
        Returns:
            Scheduler instance
        """
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'linear':
            return LinearLR(optimizer, total_iters=epochs)
        elif scheduler_type == 'constant':
            return ConstantLR(optimizer)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
