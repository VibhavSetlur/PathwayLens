"""
Model caching system for pre-trained models.

Provides caching and versioning for downloaded models.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
from loguru import logger


class ModelCache:
    """Cache manager for pre-trained models."""
    
    def __init__(self, cache_dir: str):
        """
        Initialize model cache.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.logger = logger.bind(module="model_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache metadata: {e}")
    
    def _make_cache_key(self, model_id: str, task: str) -> str:
        """Create cache key from model ID and task."""
        key_string = f"{model_id}:{task}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get_model(
        self,
        model_id: str,
        task: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached model if available.
        
        Args:
            model_id: HuggingFace model identifier
            task: Task type
            
        Returns:
            Dictionary with 'model' and 'tokenizer' or None
        """
        cache_key = self._make_cache_key(model_id, task)
        
        if cache_key not in self.metadata:
            return None
        
        model_info = self.metadata[cache_key]
        model_path = self.cache_dir / cache_key
        
        if not model_path.exists():
            self.logger.warning(f"Cache entry exists but model directory missing: {cache_key}")
            del self.metadata[cache_key]
            self._save_metadata()
            return None
        
        # Check if cache is still valid (optional expiry)
        if "expires_at" in model_info:
            expires_at = datetime.fromisoformat(model_info["expires_at"])
            if datetime.now() > expires_at:
                self.logger.info(f"Cache expired for {model_id}")
                self.remove_model(model_id, task)
                return None
        
        # Load model from cache
        try:
            from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
            
            # Load tokenizer
            tokenizer_path = model_path / "tokenizer"
            if tokenizer_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load model
            model_path_str = str(model_path / "model")
            if task == "feature-extraction":
                model = AutoModel.from_pretrained(model_path_str)
            elif task == "sequence-classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_path_str)
            else:
                model = AutoModel.from_pretrained(model_path_str)
            
            self.logger.debug(f"Loaded cached model: {model_id}")
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "model_id": model_id,
                "cached_at": model_info.get("cached_at"),
                "version": model_info.get("version", "unknown")
            }
        
        except Exception as e:
            self.logger.warning(f"Failed to load cached model {model_id}: {e}")
            self.remove_model(model_id, task)
            return None
    
    def save_model(
        self,
        model_id: str,
        task: str,
        model: Any,
        tokenizer: Any,
        version: Optional[str] = None
    ):
        """
        Save model to cache.
        
        Args:
            model_id: HuggingFace model identifier
            task: Task type
            model: Model object
            tokenizer: Tokenizer object
            version: Optional version string
        """
        cache_key = self._make_cache_key(model_id, task)
        model_dir = self.cache_dir / cache_key
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            model_path = model_dir / "model"
            model.save_pretrained(str(model_path))
            
            # Save tokenizer
            tokenizer_path = model_dir / "tokenizer"
            tokenizer.save_pretrained(str(tokenizer_path))
            
            # Update metadata
            self.metadata[cache_key] = {
                "model_id": model_id,
                "task": task,
                "version": version or "unknown",
                "cached_at": datetime.now().isoformat(),
                "cache_path": str(model_dir)
            }
            
            self._save_metadata()
            
            self.logger.info(f"Cached model: {model_id} ({task})")
        
        except Exception as e:
            self.logger.error(f"Failed to cache model {model_id}: {e}")
    
    def remove_model(self, model_id: str, task: str):
        """
        Remove model from cache.
        
        Args:
            model_id: HuggingFace model identifier
            task: Task type
        """
        cache_key = self._make_cache_key(model_id, task)
        
        if cache_key in self.metadata:
            model_dir = self.cache_dir / cache_key
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            
            del self.metadata[cache_key]
            self._save_metadata()
            
            self.logger.info(f"Removed cached model: {model_id} ({task})")
    
    def list_cached_models(self) -> Dict[str, Any]:
        """List all cached models."""
        return {
            key: {
                "model_id": info.get("model_id"),
                "task": info.get("task"),
                "version": info.get("version"),
                "cached_at": info.get("cached_at")
            }
            for key, info in self.metadata.items()
        }
    
    def clear_cache(self):
        """Clear all cached models."""
        import shutil
        
        for cache_key in list(self.metadata.keys()):
            model_dir = self.cache_dir / cache_key
            if model_dir.exists():
                shutil.rmtree(model_dir)
        
        self.metadata = {}
        self._save_metadata()
        
        self.logger.info("Cleared all cached models")
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        return total_size



