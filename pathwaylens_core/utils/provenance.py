"""
Provenance tracking for PathwayLens.

Tracks input versions, algorithm versions, database versions, and
all parameters for reproducibility.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger
import platform
import sys


@dataclass
class ProvenanceRecord:
    """Provenance record for an analysis."""
    job_id: str
    timestamp: str
    input_files: List[str]
    input_hashes: Dict[str, str]
    parameters: Dict[str, Any]
    algorithm_versions: Dict[str, str]
    database_versions: Dict[str, str]
    software_version: str
    python_version: str
    platform_info: Dict[str, str]
    command_line: str
    random_seed: Optional[int] = None


class ProvenanceTracker:
    """Track provenance for analyses."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize provenance tracker.
        
        Args:
            output_dir: Directory to save provenance records
        """
        self.logger = logger.bind(module="provenance_tracker")
        self.output_dir = output_dir or Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_record(
        self,
        job_id: str,
        input_files: List[str],
        parameters: Dict[str, Any],
        algorithm_versions: Optional[Dict[str, str]] = None,
        database_versions: Optional[Dict[str, str]] = None,
        random_seed: Optional[int] = None,
        command_line: Optional[str] = None
    ) -> ProvenanceRecord:
        """
        Create a provenance record.
        
        Args:
            job_id: Unique job identifier
            input_files: List of input file paths
            parameters: Analysis parameters
            algorithm_versions: Algorithm version information
            database_versions: Database version information
            random_seed: Random seed used
            command_line: Command line invocation
            
        Returns:
            ProvenanceRecord
        """
        # Calculate file hashes
        input_hashes = {}
        for file_path in input_files:
            try:
                file_hash = self._calculate_file_hash(file_path)
                input_hashes[file_path] = file_hash
            except Exception as e:
                self.logger.warning(f"Could not hash file {file_path}: {e}")
        
        # Get software versions
        software_version = self._get_software_version()
        python_version = sys.version
        
        # Get platform info
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        record = ProvenanceRecord(
            job_id=job_id,
            timestamp=datetime.now().isoformat(),
            input_files=input_files,
            input_hashes=input_hashes,
            parameters=parameters,
            algorithm_versions=algorithm_versions or {},
            database_versions=database_versions or {},
            software_version=software_version,
            python_version=python_version,
            platform_info=platform_info,
            command_line=command_line or '',
            random_seed=random_seed
        )
        
        return record
    
    def save_record(self, record: ProvenanceRecord) -> Path:
        """
        Save provenance record to file.
        
        Args:
            record: Provenance record to save
            
        Returns:
            Path to saved file
        """
        output_file = self.output_dir / f"provenance_{record.job_id}.json"
        
        # Convert to dict and save as JSON
        record_dict = asdict(record)
        
        with open(output_file, 'w') as f:
            json.dump(record_dict, f, indent=2)
        
        self.logger.info(f"Saved provenance record to {output_file}")
        return output_file
    
    def load_record(self, job_id: str) -> Optional[ProvenanceRecord]:
        """
        Load provenance record from file.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ProvenanceRecord or None if not found
        """
        provenance_file = self.output_dir / f"provenance_{job_id}.json"
        
        if not provenance_file.exists():
            return None
        
        try:
            with open(provenance_file, 'r') as f:
                record_dict = json.load(f)
            
            return ProvenanceRecord(**record_dict)
        except Exception as e:
            self.logger.error(f"Error loading provenance record: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _get_software_version(self) -> str:
        """Get PathwayLens version."""
        try:
            from pathwaylens_core import __version__
            return __version__
        except ImportError:
            return "unknown"


class ReproducibilityManager:
    """Manage reproducibility settings."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize reproducibility manager.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.logger = logger.bind(module="reproducibility_manager")
        self.seed = seed
        
        if seed is not None:
            self._set_random_seed(seed)
    
    def _set_random_seed(self, seed: int):
        """Set random seed for all random number generators."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Try to set other libraries' seeds
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        
        self.logger.info(f"Set random seed to {seed} for reproducibility")
    
    def get_seed(self) -> Optional[int]:
        """Get current random seed."""
        return self.seed
    
    def set_seed(self, seed: int):
        """Set new random seed."""
        self.seed = seed
        self._set_random_seed(seed)



