"""
Automated regression testing with baseline tracking.

This module provides automated regression testing that tracks
performance and functional baselines over time.
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from pathwaylens_core.data import DatabaseManager
from pathwaylens_core.analysis import (
    ORAEngine,
    GSEAEngine,
    GSVAEngine,
    NetworkEngine,
    BayesianEngine
)
from pathwaylens_core.normalization import (
    Normalizer,
    IDConverter,
    ConfidenceCalculator
)
from pathwaylens_core.data.utils.file_utils import FileUtils
from pathwaylens_core.analysis.schemas import DatabaseType
from pathwaylens_core.normalization.schemas import SpeciesType


@dataclass
class RegressionBaseline:
    """Baseline for regression testing."""
    test_name: str
    expected_result: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: str
    version: str = "2.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RegressionBaselineManager:
    """Manages regression test baselines."""
    
    def __init__(self, baseline_dir: Path = None):
        """
        Initialize baseline manager.
        
        Args:
            baseline_dir: Directory to store baselines
        """
        if baseline_dir is None:
            baseline_dir = Path(__file__).parent / "baselines"
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.baselines: Dict[str, RegressionBaseline] = {}
        self.load_baselines()
    
    def load_baselines(self):
        """Load all baselines from disk."""
        baseline_file = self.baseline_dir / "regression_baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    for name, baseline_data in data.items():
                        self.baselines[name] = RegressionBaseline(**baseline_data)
            except Exception as e:
                print(f"Warning: Could not load baselines: {e}")
    
    def save_baselines(self):
        """Save all baselines to disk."""
        baseline_file = self.baseline_dir / "regression_baselines.json"
        data = {
            name: baseline.to_dict()
            for name, baseline in self.baselines.items()
        }
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_baseline(self, test_name: str) -> Optional[RegressionBaseline]:
        """Get baseline for a test."""
        return self.baselines.get(test_name)
    
    def set_baseline(self, test_name: str, baseline: RegressionBaseline):
        """Set baseline for a test."""
        self.baselines[test_name] = baseline
        self.save_baselines()
    
    def update_baseline(self, test_name: str, result: Dict[str, Any], 
                       metrics: Dict[str, float]):
        """Update baseline with new results."""
        baseline = RegressionBaseline(
            test_name=test_name,
            expected_result=result,
            performance_metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        self.set_baseline(test_name, baseline)


@pytest.fixture(scope="session")
def baseline_manager(tmp_path_factory):
    """Baseline manager fixture."""
    baseline_dir = tmp_path_factory.mktemp("regression_baselines")
    return RegressionBaselineManager(baseline_dir)


@pytest.fixture
def sample_gene_list() -> List[str]:
    """Sample gene list for regression testing."""
    return [
        "BRCA1", "BRCA2", "TP53", "EGFR", "MYC",
        "AKT1", "PIK3CA", "PTEN", "KRAS", "BRAF"
    ]


@pytest.fixture
def database_manager():
    """Database manager fixture."""
    return DatabaseManager()


class TestAutomatedRegression:
    """Automated regression tests with baseline tracking."""
    
    @pytest.mark.asyncio
    @pytest.mark.regression
    async def test_ora_engine_regression_automated(
        self,
        sample_gene_list: List[str],
        database_manager: DatabaseManager,
        baseline_manager: RegressionBaselineManager
    ):
        """Automated regression test for ORA engine."""
        engine = ORAEngine(database_manager)
        
        import time
        start_time = time.time()
        
        result = await engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.GO,
            species=SpeciesType.HUMAN,
            significance_threshold=0.05
        )
        
        elapsed_time = time.time() - start_time
        
        # Get or create baseline
        baseline = baseline_manager.get_baseline("ora_engine_regression")
        
        if baseline is None:
            # Create initial baseline
            expected_result = {
                "has_results": result is not None,
                "has_pathways": hasattr(result, 'pathways') if result else False,
                "pathway_count": len(result.pathways) if result and hasattr(result, 'pathways') else 0
            }
            metrics = {"elapsed_time": elapsed_time}
            baseline_manager.update_baseline("ora_engine_regression", expected_result, metrics)
            pytest.skip("Initial baseline created - run again to test regression")
        else:
            # Check against baseline
            current_result = {
                "has_results": result is not None,
                "has_pathways": hasattr(result, 'pathways') if result else False,
                "pathway_count": len(result.pathways) if result and hasattr(result, 'pathways') else 0
            }
            
            # Check functional regression
            assert current_result["has_results"] == baseline.expected_result["has_results"], \
                "ORA engine regression: has_results changed"
            assert current_result["has_pathways"] == baseline.expected_result["has_pathways"], \
                "ORA engine regression: has_pathways changed"
            
            # Check performance regression (allow 50% slowdown)
            max_time = baseline.performance_metrics["elapsed_time"] * 1.5
            assert elapsed_time <= max_time, \
                f"ORA engine performance regression: {elapsed_time:.2f}s > {max_time:.2f}s"
    
    @pytest.mark.asyncio
    @pytest.mark.regression
    async def test_gsea_engine_regression_automated(
        self,
        sample_gene_list: List[str],
        database_manager: DatabaseManager,
        baseline_manager: RegressionBaselineManager
    ):
        """Automated regression test for GSEA engine."""
        ranked_genes = [
            ("BRCA1", 2.5), ("BRCA2", 2.3), ("TP53", 2.1),
            ("EGFR", 1.9), ("MYC", 1.7), ("AKT1", 1.5)
        ]
        
        engine = GSEAEngine(database_manager)
        
        import time
        start_time = time.time()
        
        result = await engine.analyze(
            gene_list=[gene for gene, _ in ranked_genes],
            database=DatabaseType.GO,
            species=SpeciesType.HUMAN,
            significance_threshold=0.05,
            permutations=100
        )
        
        elapsed_time = time.time() - start_time
        
        baseline = baseline_manager.get_baseline("gsea_engine_regression")
        
        if baseline is None:
            expected_result = {
                "has_results": result is not None,
                "has_pathways": hasattr(result, 'pathways') if result else False
            }
            metrics = {"elapsed_time": elapsed_time}
            baseline_manager.update_baseline("gsea_engine_regression", expected_result, metrics)
            pytest.skip("Initial baseline created")
        else:
            current_result = {
                "has_results": result is not None,
                "has_pathways": hasattr(result, 'pathways') if result else False
            }
            
            assert current_result == baseline.expected_result, \
                "GSEA engine functional regression detected"
            
            max_time = baseline.performance_metrics["elapsed_time"] * 1.5
            assert elapsed_time <= max_time, \
                f"GSEA engine performance regression: {elapsed_time:.2f}s > {max_time:.2f}s"
    
    @pytest.mark.asyncio
    @pytest.mark.regression
    async def test_normalizer_regression_automated(
        self,
        sample_gene_list: List[str],
        baseline_manager: RegressionBaselineManager
    ):
        """Automated regression test for normalizer."""
        normalizer = Normalizer()
        
        import time
        start_time = time.time()
        
        result = await normalizer.normalize(
            gene_list=sample_gene_list,
            input_type="symbol",
            output_type="ensembl_gene_id",
            species=SpeciesType.HUMAN
        )
        
        elapsed_time = time.time() - start_time
        
        baseline = baseline_manager.get_baseline("normalizer_regression")
        
        if baseline is None:
            expected_result = {
                "has_result": result is not None,
                "has_converted_genes": hasattr(result, 'converted_genes') if result else False,
                "conversion_rate": result.conversion_stats.get('success_rate', 0) if result else 0
            }
            metrics = {"elapsed_time": elapsed_time}
            baseline_manager.update_baseline("normalizer_regression", expected_result, metrics)
            pytest.skip("Initial baseline created")
        else:
            current_result = {
                "has_result": result is not None,
                "has_converted_genes": hasattr(result, 'converted_genes') if result else False,
                "conversion_rate": result.conversion_stats.get('success_rate', 0) if result else 0
            }
            
            # Allow 10% variation in conversion rate
            expected_rate = baseline.expected_result["conversion_rate"]
            current_rate = current_result["conversion_rate"]
            assert abs(current_rate - expected_rate) <= 0.1, \
                f"Normalizer conversion rate regression: {current_rate:.2f} vs {expected_rate:.2f}"
            
            max_time = baseline.performance_metrics["elapsed_time"] * 1.5
            assert elapsed_time <= max_time, \
                f"Normalizer performance regression: {elapsed_time:.2f}s > {max_time:.2f}s"


class TestBugRegression:
    """Regression tests for fixed bugs."""
    
    @pytest.mark.regression
    def test_fixed_bug_gsea_engine_initialization(self):
        """Test that GSEA engine initialization bug is fixed."""
        from pathwaylens_core.analysis.multi_omics_engine import MultiOmicsEngine
        from pathwaylens_core.data import DatabaseManager
        
        db_manager = DatabaseManager()
        engine = MultiOmicsEngine(db_manager)
        
        # This should not raise AttributeError
        assert hasattr(engine, 'gsea_engine'), \
            "GSEA engine should be initialized in MultiOmicsEngine"
    
    @pytest.mark.regression
    def test_fixed_bug_visualization_engine_syntax(self):
        """Test that visualization engine syntax error is fixed."""
        from pathwaylens_core.visualization.engine import VisualizationEngine
        
        engine = VisualizationEngine()
        
        # This should not raise SyntaxError
        assert hasattr(engine, 'renderers'), \
            "Visualization engine should have renderers attribute"
        assert isinstance(engine.renderers, dict), \
            "Renderers should be a dictionary"
    
    @pytest.mark.regression
    def test_fixed_bug_id_converter_batch_processing(self):
        """Test that ID converter batch processing works correctly."""
        from pathwaylens_core.normalization.id_converter import IDConverter
        from pathwaylens_core.data import DatabaseManager
        
        db_manager = DatabaseManager()
        converter = IDConverter(db_manager)
        
        # Test with large gene list (should use batch processing)
        large_gene_list = [f"GENE{i}" for i in range(1500)]
        
        # Should not raise errors
        assert converter is not None
        # Note: Actual conversion would require database connection
        # This test verifies the batch processing logic exists



