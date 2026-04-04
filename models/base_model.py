import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base class for all models in DSM-EQA."""
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize base model.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        return f"Model({self.input_dim} -> {self.output_dim})"
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
