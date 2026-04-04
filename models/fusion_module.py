import torch
import torch.nn as nn

class FusionModule(nn.Module):
    """Module for fusing multiple feature representations."""
    
    def __init__(self, fusion_type: str = 'concatenation', hidden_dim: int = 128):
        """
        Initialize fusion module.
        
        Args:
            fusion_type: Type of fusion ('concatenation', 'attention', 'addition')
            hidden_dim: Hidden dimension for fusion network
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
    
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple feature tensors.
        
        Args:
            features: List of feature tensors
            
        Returns:
            Fused feature tensor
        """
        if self.fusion_type == 'concatenation':
            return torch.cat(features, dim=-1)
        elif self.fusion_type == 'addition':
            return torch.stack(features, dim=0).sum(dim=0)
        elif self.fusion_type == 'attention':
            return self._attention_fusion(features)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def _attention_fusion(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Attention-based fusion."""
        features_stacked = torch.stack(features, dim=1)  # (B, N, D)
        weights = torch.softmax(features_stacked.sum(dim=-1), dim=1)  # (B, N)
        return (features_stacked * weights.unsqueeze(-1)).sum(dim=1)
