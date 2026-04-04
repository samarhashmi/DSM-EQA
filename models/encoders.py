import torch
import torch.nn as nn
from typing import List, Optional

class Encoder(nn.Module):
    """Base encoder class."""
    
    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = self._get_activation(activation)
    
    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MultiLayerEncoder(Encoder):
    """Multi-layer encoder with configurable architecture."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout: float = 0.2):
        """
        Initialize multi-layer encoder.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function type
            dropout: Dropout rate
        """
        super().__init__(input_dim, output_dim, activation)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        return self.network(x)
