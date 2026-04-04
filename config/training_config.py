from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Optimizer settings
    optimizer: str = "adam"
    scheduler: str = "cosine"  
    
    # Loss function settings
    loss_function: str = "mse"
    aux_loss_weight: float = 0.1
    
    # Validation and checkpointing
    validation_split: float = 0.2
    save_checkpoint_every: int = 10
    early_stopping_patience: int = 20
    
    # Device settings
    device: str = "cuda"
    num_workers: int = 4
    
    # Logging settings
    log_every: int = 10
    tensorboard_log: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__
