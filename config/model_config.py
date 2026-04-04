from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for model architecture and hyperparameters."""
    
    # Encoder settings
    input_dim: int = 10
    hidden_dims: List[int] = None
    encoder_type: str = "multi_layer"
    encoder_dropout: float = 0.2
    
    # Fusion module settings
    fusion_type: str = "concatenation" 
    fusion_hidden_dim: int = 128
    
    # Physics-informed settings
    use_physics_constraints: bool = True
    constraint_weight: float = 0.1
    
    # Output settings
    output_dim: int = 1
    activation: str = "relu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32, 16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__
