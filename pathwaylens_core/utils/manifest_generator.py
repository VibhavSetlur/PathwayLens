"""
Manifest generation utilities for reproducible pathway analysis.

This module provides functions to capture complete analysis provenance
including environment, database versions, and input data checksums.
"""

import os
import sys
import platform
import hashlib
import psutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import importlib.metadata

from ..schemas.provenance import (
    ExecutionEnvironment,
    DatabaseVersion,
    InputFileMetadata,
    AnalysisManifest
)


def capture_environment() -> ExecutionEnvironment:
    """
    Capture current execution environment.
    
    Returns:
        ExecutionEnvironment with system and software information
    """
    # Get key dependencies
    dependencies = {}
    key_packages = [
        'numpy', 'scipy', 'pandas', 'plotly', 'pydantic',
        'statsmodels', 'scikit-learn', 'networkx'
    ]
    
    for package in key_packages:
        try:
            version = importlib.metadata.version(package)
            dependencies[package] = version
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not installed"
    
    # Get hardware info
    try:
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
    except:
        cpu_count = None
        memory_gb = None
    
    # Get PathwayLens version
    try:
        pathwaylens_version = importlib.metadata.version('pathwaylens')
    except:
        pathwaylens_version = "development"
    
    return ExecutionEnvironment(
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        pathwaylens_version=pathwaylens_version,
        dependencies=dependencies,
        cpu_count=cpu_count,
        memory_gb=memory_gb
    )


def calculate_file_checksum(filepath: str) -> str:
    """
    Calculate MD5 checksum of a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        MD5 checksum as hex string
    """
    md5_hash = hashlib.md5()
    
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest()


def create_input_file_metadata(filepath: str) -> InputFileMetadata:
    """
    Create metadata for an input file.
    
    Args:
        filepath: Path to input file
        
    Returns:
        InputFileMetadata with file information
    """
    path = Path(filepath)
    
    return InputFileMetadata(
        filename=path.name,
        filepath=str(path.absolute()),
        checksum=calculate_file_checksum(filepath),
        size_bytes=path.stat().st_size,
        file_format=path.suffix.lstrip('.')
    )


def calculate_reproducibility_hash(
    environment: ExecutionEnvironment,
    database_versions: Dict[str, DatabaseVersion],
    parameters: Dict[str, Any],
    input_checksums: Dict[str, str],
    random_seed: Optional[int] = None
) -> str:
    """
    Calculate reproducibility hash from all critical components.
    
    This hash uniquely identifies the exact configuration used for analysis.
    
    Args:
        environment: Execution environment
        database_versions: Database version information
        parameters: Analysis parameters
        input_checksums: Input file checksums
        random_seed: Random seed if used
        
    Returns:
        SHA256 hash as hex string
    """
    # Create deterministic string representation
    components = []
    
    # Environment
    components.append(f"python:{environment.python_version}")
    components.append(f"pathwaylens:{environment.pathwaylens_version}")
    for pkg, ver in sorted(environment.dependencies.items()):
        components.append(f"{pkg}:{ver}")
    
    # Databases
    for db_name, db_ver in sorted(database_versions.items()):
        components.append(f"db:{db_name}:{db_ver.version}:{db_ver.checksum or 'none'}")
    
    # Parameters (sorted for determinism)
    for key, value in sorted(parameters.items()):
        components.append(f"param:{key}:{value}")
    
    # Input checksums
    for filename, checksum in sorted(input_checksums.items()):
        components.append(f"input:{filename}:{checksum}")
    
    # Random seed
    if random_seed is not None:
        components.append(f"seed:{random_seed}")
    
    # Calculate hash
    hash_input = "|".join(components).encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()


def generate_manifest(
    analysis_id: str,
    analysis_type: str,
    parameters: Dict[str, Any],
    input_files: List[str],
    database_versions: Dict[str, DatabaseVersion],
    random_seed: Optional[int] = None
) -> AnalysisManifest:
    """
    Generate complete analysis manifest.
    
    Args:
        analysis_id: Unique analysis identifier
        analysis_type: Type of analysis
        parameters: Analysis parameters
        input_files: List of input file paths
        database_versions: Database version information
        random_seed: Random seed if used
        
    Returns:
        Complete AnalysisManifest
    """
    # Capture environment
    environment = capture_environment()
    
    # Process input files
    input_metadata = {}
    input_checksums = {}
    for filepath in input_files:
        metadata = create_input_file_metadata(filepath)
        input_metadata[metadata.filename] = metadata
        input_checksums[metadata.filename] = metadata.checksum
    
    # Calculate reproducibility hash
    repro_hash = calculate_reproducibility_hash(
        environment=environment,
        database_versions=database_versions,
        parameters=parameters,
        input_checksums=input_checksums,
        random_seed=random_seed
    )
    
    # Create manifest
    manifest = AnalysisManifest(
        manifest_version="1.0",
        analysis_id=analysis_id,
        created_at=datetime.now().isoformat(),
        pathwaylens_version=environment.pathwaylens_version,
        execution_environment=environment,
        database_versions=database_versions,
        analysis_type=analysis_type,
        parameters=parameters,
        input_files=input_metadata,
        random_seed=random_seed,
        reproducibility_hash=repro_hash
    )
    
    return manifest


def save_manifest(manifest: AnalysisManifest, output_dir: Path) -> Path:
    """
    Save manifest to JSON file.
    
    Args:
        manifest: AnalysisManifest to save
        output_dir: Output directory
        
    Returns:
        Path to saved manifest file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "manifest.json"
    
    with open(manifest_path, 'w') as f:
        f.write(manifest.model_dump_json(indent=2))
    
    return manifest_path


def load_manifest(manifest_path: Path) -> AnalysisManifest:
    """
    Load manifest from JSON file.
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        Loaded AnalysisManifest
    """
    with open(manifest_path, 'r') as f:
        return AnalysisManifest.model_validate_json(f.read())


def validate_reproducibility(
    manifest1: AnalysisManifest,
    manifest2: AnalysisManifest
) -> Dict[str, Any]:
    """
    Validate if two manifests represent reproducible analyses.
    
    Args:
        manifest1: First manifest
        manifest2: Second manifest
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "reproducible": False,
        "hash_match": False,
        "differences": []
    }
    
    # Check reproducibility hash
    if manifest1.reproducibility_hash == manifest2.reproducibility_hash:
        results["hash_match"] = True
        results["reproducible"] = True
        return results
    
    # Identify differences
    if manifest1.pathwaylens_version != manifest2.pathwaylens_version:
        results["differences"].append(
            f"PathwayLens version: {manifest1.pathwaylens_version} vs {manifest2.pathwaylens_version}"
        )
    
    if manifest1.execution_environment.python_version != manifest2.execution_environment.python_version:
        results["differences"].append(
            f"Python version: {manifest1.execution_environment.python_version} vs "
            f"{manifest2.execution_environment.python_version}"
        )
    
    # Check database versions
    for db_name in set(manifest1.database_versions.keys()) | set(manifest2.database_versions.keys()):
        v1 = manifest1.database_versions.get(db_name)
        v2 = manifest2.database_versions.get(db_name)
        
        if v1 and v2:
            if v1.version != v2.version:
                results["differences"].append(
                    f"Database {db_name}: {v1.version} vs {v2.version}"
                )
        elif v1:
            results["differences"].append(f"Database {db_name}: present vs absent")
        else:
            results["differences"].append(f"Database {db_name}: absent vs present")
    
    return results
