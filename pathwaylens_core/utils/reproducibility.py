"""
Reproducibility utilities for PathwayLens.

Provides seed management and deterministic algorithm execution.
"""

import random
import numpy as np
from typing import Optional
from loguru import logger


class SeedManager:
    """Manage random seeds for reproducibility."""
    
    _global_seed: Optional[int] = None
    
    @classmethod
    def set_global_seed(cls, seed: int):
        """
        Set global random seed.
        
        Args:
            seed: Random seed value
        """
        cls._global_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Try to set other libraries
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        
        logger.info(f"Set global random seed to {seed}")
    
    @classmethod
    def get_global_seed(cls) -> Optional[int]:
        """Get current global seed."""
        return cls._global_seed
    
    @classmethod
    def reset(cls):
        """Reset seed manager."""
        cls._global_seed = None


def ensure_deterministic(func):
    """
    Decorator to ensure function execution is deterministic.
    
    Sets random seed before function execution.
    """
    def wrapper(*args, **kwargs):
        seed = SeedManager.get_global_seed()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        return func(*args, **kwargs)
    
    return wrapper



