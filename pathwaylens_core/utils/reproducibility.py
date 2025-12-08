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


# =============================================================================
# Metadata Generation for Reproducibility
# =============================================================================

import os
import sys
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union
from dataclasses import dataclass, asdict, field

try:
    import importlib.metadata
    HAS_IMPORTLIB = True
except ImportError:
    HAS_IMPORTLIB = False


@dataclass
class ReproducibilityMetadata:
    """Comprehensive metadata for analysis reproducibility."""
    
    pathwaylens_version: str
    python_version: str
    platform_info: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    git_commit_hash: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_files: List[Dict[str, str]] = field(default_factory=list)
    database_versions: Dict[str, str] = field(default_factory=dict)
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)


def get_pathwaylens_version() -> str:
    """Get PathwayLens version from package metadata."""
    try:
        if HAS_IMPORTLIB:
            return importlib.metadata.version("pathwaylens")
    except Exception:
        pass
    return "unknown"


def get_git_info() -> Dict[str, Any]:
    """Get Git repository information if available."""
    info = {"commit_hash": None, "branch": None, "dirty": False}
    
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["commit_hash"] = result.stdout.strip()[:12]
        
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
        
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    
    return info


def calculate_file_checksum(filepath: Union[str, Path]) -> str:
    """Calculate SHA256 checksum of a file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return "file_not_found"
    
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_dependency_versions() -> Dict[str, str]:
    """Get versions of key dependencies."""
    deps = ["numpy", "pandas", "scipy", "statsmodels", "plotly", "typer", "pydantic"]
    versions = {}
    for pkg in deps:
        try:
            if HAS_IMPORTLIB:
                versions[pkg] = importlib.metadata.version(pkg)
            else:
                versions[pkg] = "unknown"
        except Exception:
            versions[pkg] = "not_installed"
    return versions


def _make_serializable(obj: Any) -> Any:
    """Convert an object to JSON-serializable format."""
    if hasattr(obj, 'value'):
        return obj.value
    elif hasattr(obj, '__dict__'):
        return {k: _make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def generate_metadata_json(
    params: Any,
    input_files: List[Union[str, Path]],
    output_dir: Union[str, Path],
    database_versions: Optional[Dict[str, str]] = None
) -> Path:
    """
    Generate and save _metadata.json to output directory.
    
    Args:
        params: Analysis parameters
        input_files: List of input file paths
        output_dir: Output directory
        database_versions: Optional database version info
        
    Returns:
        Path to generated metadata file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    git_info = get_git_info()
    
    # Calculate checksums
    file_info = []
    for fp in input_files:
        fp = Path(fp)
        file_info.append({
            "filename": fp.name,
            "path": str(fp.absolute()) if fp.exists() else str(fp),
            "sha256": calculate_file_checksum(fp),
        })
    
    # Convert params
    if hasattr(params, '__dataclass_fields__'):
        params_dict = asdict(params)
    elif hasattr(params, '__dict__'):
        params_dict = vars(params)
    elif isinstance(params, dict):
        params_dict = params
    else:
        params_dict = {"value": str(params)}
    
    metadata = ReproducibilityMetadata(
        pathwaylens_version=get_pathwaylens_version(),
        python_version=sys.version,
        platform_info=platform.platform(),
        git_commit_hash=git_info["commit_hash"],
        git_branch=git_info["branch"],
        git_dirty=git_info["dirty"],
        parameters=_make_serializable(params_dict),
        input_files=file_info,
        database_versions=database_versions or {},
        dependency_versions=get_dependency_versions(),
    )
    
    metadata_path = output_dir / "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2, default=str)
    
    logger.info(f"Reproducibility metadata saved: {metadata_path}")
    return metadata_path


def print_version_info() -> str:
    """Print detailed version information."""
    git_info = get_git_info()
    deps = get_dependency_versions()
    
    lines = [
        "=" * 50,
        "PathwayLens Version Information",
        "=" * 50,
        f"PathwayLens: {get_pathwaylens_version()}",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
    ]
    
    if git_info["commit_hash"]:
        lines.extend([
            f"Git: {git_info['commit_hash']} ({git_info['branch']})"
            + (" [dirty]" if git_info["dirty"] else ""),
        ])
    
    lines.append("\nDependencies:")
    for pkg, ver in deps.items():
        lines.append(f"  {pkg}: {ver}")
    
    lines.append("=" * 50)
    return "\n".join(lines)
