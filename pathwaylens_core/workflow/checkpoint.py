"""
Checkpoint system for workflows.

Saves intermediate results for resume capability.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
import hashlib


class CheckpointManager:
    """Manage workflow checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(module="checkpoint_manager")
    
    def save_checkpoint(
        self,
        workflow_id: str,
        step_id: str,
        artifacts: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            workflow_id: Workflow identifier
            step_id: Step identifier
            artifacts: Artifacts to save
            metadata: Optional metadata
            
        Returns:
            Path to checkpoint file
        """
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}_{step_id}.ckpt"
        
        checkpoint_data = {
            'workflow_id': workflow_id,
            'step_id': step_id,
            'timestamp': datetime.now().isoformat(),
            'artifacts': artifacts,
            'metadata': metadata or {}
        }
        
        # Save as pickle for complex objects
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_file}")
        return checkpoint_file
    
    def load_checkpoint(
        self,
        workflow_id: str,
        step_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            workflow_id: Workflow identifier
            step_id: Step identifier
            
        Returns:
            Checkpoint data or None
        """
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}_{step_id}.ckpt"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"Loaded checkpoint: {checkpoint_file}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def has_checkpoint(self, workflow_id: str, step_id: str) -> bool:
        """Check if checkpoint exists."""
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}_{step_id}.ckpt"
        return checkpoint_file.exists()
    
    def list_checkpoints(self, workflow_id: str) -> list[Path]:
        """List all checkpoints for a workflow."""
        pattern = f"{workflow_id}_*.ckpt"
        return list(self.checkpoint_dir.glob(pattern))
    
    def delete_checkpoint(self, workflow_id: str, step_id: str) -> bool:
        """Delete a checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}_{step_id}.ckpt"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False



