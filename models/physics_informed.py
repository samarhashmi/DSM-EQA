import torch
import torch.nn as nn
from typing import Callable, Optional

class PhysicsInformedModule(nn.Module):
    """Module for incorporating physics constraints into neural networks."""
    
    def __init__(self, physics_constraint: Optional[Callable] = None):
        """
        Initialize physics-informed module.
        
        Args:
            physics_constraint: Callable that defines physics constraint
        """
        super().__init__()
        self.physics_constraint = physics_constraint
    
    def forward(self, x: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """
        Apply physics constraints to model output.
        
        Args:
            x: Input features
            model_output: Raw model output
            
        Returns:
            Physics-constrained output
        """
        if self.physics_constraint is None:
            return model_output
        
        return self.physics_constraint(x, model_output)
    
    def compute_constraint_loss(self, x: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """
        Compute loss from physics constraints.
        
        Args:
            x: Input features
            model_output: Model output
            
        Returns:
            Constraint loss tensor
        """
        if self.physics_constraint is None:
            return torch.tensor(0.0, device=x.device)
        
        # Enable gradient computation for constraint
        x.requires_grad_(True)
        model_output = self.physics_constraint(x, model_output)
        
        # Compute violation of constraint (simplified)
        return torch.mean(torch.abs(model_output))
