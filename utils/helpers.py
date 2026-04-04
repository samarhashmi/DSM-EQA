import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_dir(output_dir: str = 'output') -> Path:
    """
    Create output directory.
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        Path object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def normalize_data(X: np.ndarray, method: str = 'standard') -> tuple:
    """
    Normalize data.
    
    Args:
        X: Input data
        method: Normalization method ('standard' or 'minmax')
        
    Returns:
        Tuple of (normalized_X, params)
    """
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)
        return X_norm, {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        return X_norm, {'min': min_val, 'max': max_val}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
