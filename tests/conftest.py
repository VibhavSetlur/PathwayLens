"""
Pytest configuration and shared fixtures for PathwayLens testing.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

from pathwaylens_core.data.database_manager import DatabaseManager
from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.comparison.engine import ComparisonEngine
from pathwaylens_core.visualization.engine import VisualizationEngine
from pathwaylens_core.normalization.normalizer import NormalizationEngine


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def database_manager() -> DatabaseManager:
    """Create a database manager for testing."""
    return DatabaseManager()


@pytest.fixture
def analysis_engine(database_manager: DatabaseManager) -> AnalysisEngine:
    """Create an analysis engine for testing."""
    return AnalysisEngine(database_manager)


@pytest.fixture
def comparison_engine() -> ComparisonEngine:
    """Create a comparison engine for testing."""
    return ComparisonEngine()


@pytest.fixture
def visualization_engine() -> VisualizationEngine:
    """Create a visualization engine for testing."""
    return VisualizationEngine()


@pytest.fixture
def normalization_engine() -> NormalizationEngine:
    """Create a normalization engine for testing."""
    return NormalizationEngine()


@pytest.fixture
def sample_gene_list() -> list[str]:
    """Sample gene list for testing."""
    return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]


@pytest.fixture
def sample_pathway_data() -> Dict[str, Dict[str, Any]]:
    """Sample pathway data for testing."""
    return {
        "hsa00010": {
            "name": "Glycolysis / Gluconeogenesis",
            "genes": ["GENE1", "GENE2", "GENE3"]
        },
        "hsa00020": {
            "name": "Citrate cycle (TCA cycle)",
            "genes": ["GENE4", "GENE5", "GENE6"]
        }
    }


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration for testing."""
    return {
        "api": {
            "base_url": "http://localhost:8000",
            "timeout": 30
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "pathwaylens"
        },
        "cache": {
            "enabled": True,
            "ttl": 3600
        }
    }
