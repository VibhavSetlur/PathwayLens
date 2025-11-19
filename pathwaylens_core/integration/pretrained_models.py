"""
Pre-trained model integration for PathwayLens.

Provides access to pre-trained models for pathway analysis without training.
Supports BioBERT, PubMedBERT, and other HuggingFace models.
"""

import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from loguru import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    from huggingface_hub import hf_hub_download, snapshot_download
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    logger.warning("transformers library not available. Pre-trained model features disabled.")

from .model_cache import ModelCache


class PretrainedModelManager:
    """
    Manager for pre-trained models for pathway analysis.
    
    Supports:
    - BioBERT (dmis-lab/biobert-v1.1)
    - PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
    - Pathway prediction models from HuggingFace
    """
    
    SUPPORTED_MODELS = {
        "biobert": "dmis-lab/biobert-v1.1",
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "biobert-pubmed": "dmis-lab/biobert-base-cased-v1.2",
        "scibert": "allenai/scibert_scivocab_uncased"
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize pre-trained model manager.
        
        Args:
            cache_dir: Directory to cache models
            use_cache: Whether to use cached models
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        
        self.logger = logger.bind(module="pretrained_models")
        self.use_cache = use_cache
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".pathwaylens", "models")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_cache = ModelCache(cache_dir=str(self.cache_dir))
        
        # Cache for loaded models
        self._loaded_models: Dict[str, Any] = {}
        self._loaded_tokenizers: Dict[str, Any] = {}
    
    def list_models(self) -> List[str]:
        """List available pre-trained models."""
        return list(self.SUPPORTED_MODELS.keys())
    
    def load_model(
        self,
        model_name: str,
        task: str = "feature-extraction",
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Load a pre-trained model.
        
        Args:
            model_name: Name of model (biobert, pubmedbert, etc.)
            task: Task type (feature-extraction, sequence-classification, etc.)
            force_reload: Force reload even if cached
            
        Returns:
            Dictionary with 'model' and 'tokenizer' keys
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        model_key = f"{model_name}_{task}"
        
        # Check if already loaded
        if not force_reload and model_key in self._loaded_models:
            self.logger.debug(f"Using cached model: {model_key}")
            return {
                "model": self._loaded_models[model_key],
                "tokenizer": self._loaded_tokenizers[model_key]
            }
        
        # Get model identifier
        model_id = self.SUPPORTED_MODELS[model_name]
        
        self.logger.info(f"Loading model: {model_name} ({model_id})")
        
        try:
            # Check cache
            if self.use_cache and not force_reload:
                cached_model = self.model_cache.get_model(model_id, task)
                if cached_model:
                    self.logger.debug(f"Using cached model from disk: {model_id}")
                    model = cached_model["model"]
                    tokenizer = cached_model["tokenizer"]
                else:
                    # Download and cache
                    model, tokenizer = self._download_model(model_id, task)
                    self.model_cache.save_model(model_id, task, model, tokenizer)
            else:
                # Download without caching
                model, tokenizer = self._download_model(model_id, task)
            
            # Store in memory cache
            self._loaded_models[model_key] = model
            self._loaded_tokenizers[model_key] = tokenizer
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "model_id": model_id,
                "model_name": model_name
            }
        
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _download_model(
        self,
        model_id: str,
        task: str
    ) -> tuple:
        """Download model from HuggingFace."""
        self.logger.info(f"Downloading model: {model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model based on task
        if task == "feature-extraction":
            model = AutoModel.from_pretrained(model_id)
        elif task == "sequence-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        else:
            model = AutoModel.from_pretrained(model_id)
        
        return model, tokenizer
    
    def extract_features(
        self,
        text: Union[str, List[str]],
        model_name: str = "biobert",
        max_length: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract features from text using a pre-trained model.
        
        Args:
            text: Input text or list of texts
            model_name: Name of model to use
            max_length: Maximum sequence length
            **kwargs: Additional parameters for tokenizer
            
        Returns:
            Dictionary with extracted features
        """
        # Load model
        model_data = self.load_model(model_name, task="feature-extraction")
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        # Tokenize input
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            **kwargs
        )
        
        # Extract features
        model.eval()
        with torch.no_grad():
            outputs = model(**encoded)
            # Use CLS token or mean pooling
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                features = outputs[0][:, 0, :]
        
        return {
            "features": features.numpy(),
            "model": model_name,
            "shape": features.shape
        }
    
    def analyze_pathway_text(
        self,
        pathway_description: str,
        model_name: str = "biobert",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze pathway text description using a pre-trained model.
        
        Args:
            pathway_description: Pathway description text
            model_name: Name of model to use
            **kwargs: Additional parameters
            
        Returns:
            Analysis results with extracted features
        """
        features = self.extract_features(
            pathway_description,
            model_name=model_name,
            **kwargs
        )
        
        return {
            "text": pathway_description,
            "features": features["features"],
            "model": model_name,
            "feature_dimension": features["shape"][1]
        }

