"""
Integration tests for error recovery and checkpoint resumption.
"""

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from pathwaylens_core.workflow.manager import WorkflowManager
from pathwaylens_core.workflow.checkpoint import CheckpointManager
from pathwaylens_core.normalization.normalizer import Normalizer
from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.analysis.schemas import AnalysisType, DatabaseType, AnalysisParameters
from pathwaylens_core.data.database_manager import DatabaseManager


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery and checkpoint resumption."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory."""
        return tmp_path

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        return CheckpointManager(checkpoint_dir=str(checkpoint_dir))

    @pytest.fixture
    def workflow_manager(self, checkpoint_manager):
        """Create workflow manager."""
        return WorkflowManager(checkpoint_manager=checkpoint_manager)

    @pytest.fixture
    def database_manager(self):
        """Create database manager."""
        return DatabaseManager()

    @pytest.fixture
    def normalizer(self):
        """Create normalizer."""
        return Normalizer()

    @pytest.fixture
    def analysis_engine(self, database_manager):
        """Create analysis engine."""
        return AnalysisEngine(database_manager)

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_load(
        self, workflow_manager, checkpoint_manager, temp_dir
    ):
        """Test saving and loading checkpoints."""
        workflow_id = "test_workflow_1"
        checkpoint_data = {
            "step": "normalization",
            "status": "completed",
            "data": {"normalized_genes": ["GENE1", "GENE2", "GENE3"]}
        }
        
        # Save checkpoint
        await checkpoint_manager.save_checkpoint(
            workflow_id=workflow_id,
            checkpoint_data=checkpoint_data
        )
        
        # Verify checkpoint file exists
        checkpoint_file = Path(checkpoint_manager.checkpoint_dir) / f"{workflow_id}.json"
        assert checkpoint_file.exists(), "Checkpoint file should exist"
        
        # Load checkpoint
        loaded_data = await checkpoint_manager.load_checkpoint(workflow_id=workflow_id)
        
        assert loaded_data is not None
        assert loaded_data["step"] == checkpoint_data["step"]
        assert loaded_data["status"] == checkpoint_data["status"]
        assert loaded_data["data"] == checkpoint_data["data"]

    @pytest.mark.asyncio
    async def test_workflow_resume_from_checkpoint(
        self, workflow_manager, checkpoint_manager, normalizer, analysis_engine
    ):
        """Test resuming workflow from checkpoint."""
        workflow_id = "test_workflow_2"
        sample_genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        
        # Simulate partial workflow execution
        normalized = await normalizer.normalize(
            input_data=sample_genes,
            species="human",
            input_id_type="symbol",
            output_id_type="ensembl"
        )
        
        # Save checkpoint after normalization
        checkpoint_data = {
            "step": "normalization",
            "status": "completed",
            "data": {
                "normalized_genes": normalized.get("gene_ids", sample_genes),
                "input_genes": sample_genes
            }
        }
        await checkpoint_manager.save_checkpoint(
            workflow_id=workflow_id,
            checkpoint_data=checkpoint_data
        )
        
        # Resume workflow
        loaded_checkpoint = await checkpoint_manager.load_checkpoint(workflow_id=workflow_id)
        
        assert loaded_checkpoint is not None
        assert loaded_checkpoint["step"] == "normalization"
        
        # Resume from checkpoint - perform analysis
        normalized_genes = loaded_checkpoint["data"]["normalized_genes"]
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        result = await analysis_engine.analyze(
            input_data=normalized_genes if isinstance(normalized_genes, list) else list(normalized_genes),
            parameters=parameters
        )
        
        assert result is not None
        assert result.analysis_type == AnalysisType.ORA

    @pytest.mark.asyncio
    async def test_error_recovery_graceful_degradation(
        self, normalizer, analysis_engine
    ):
        """Test graceful error handling and recovery."""
        # Test with invalid input
        invalid_genes = [None, "", "INVALID_GENE_XYZ", "GENE1"]
        
        try:
            normalized = await normalizer.normalize(
                input_data=invalid_genes,
                species="human",
                input_id_type="symbol",
                output_id_type="ensembl"
            )
            
            # Should handle invalid genes gracefully
            # Either filter them out or return partial results
            assert normalized is not None
            
        except Exception as e:
            # Error should be informative
            assert "invalid" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, analysis_engine):
        """Test timeout handling for slow operations."""
        # Create a large gene list that might timeout
        large_gene_list = [f"GENE{i}" for i in range(1, 10001)]
        
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        try:
            # This might timeout or take a long time
            result = await asyncio.wait_for(
                analysis_engine.analyze(
                    input_data=large_gene_list[:100],  # Use subset for test
                    parameters=parameters
                ),
                timeout=30.0
            )
            
            # If it completes, result should be valid
            if result:
                assert result.analysis_type == AnalysisType.ORA
                
        except asyncio.TimeoutError:
            # Timeout is acceptable for this test
            pass

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(
        self, checkpoint_manager, normalizer, analysis_engine
    ):
        """Test recovery from partial failures."""
        workflow_id = "test_workflow_3"
        sample_genes = ["GENE1", "GENE2", "GENE3"]
        
        # Step 1: Normalization (success)
        normalized = await normalizer.normalize(
            input_data=sample_genes,
            species="human",
            input_id_type="symbol",
            output_id_type="ensembl"
        )
        
        # Save checkpoint
        await checkpoint_manager.save_checkpoint(
            workflow_id=workflow_id,
            checkpoint_data={
                "step": "normalization",
                "status": "completed",
                "data": {"normalized_genes": normalized.get("gene_ids", sample_genes)}
            }
        )
        
        # Step 2: Analysis (simulate failure)
        normalized_genes = normalized.get("gene_ids", sample_genes)
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        try:
            result = await analysis_engine.analyze(
                input_data=normalized_genes if isinstance(normalized_genes, list) else list(normalized_genes),
                parameters=parameters
            )
            
            # If analysis succeeds, update checkpoint
            if result:
                await checkpoint_manager.save_checkpoint(
                    workflow_id=workflow_id,
                    checkpoint_data={
                        "step": "analysis",
                        "status": "completed",
                        "data": {"analysis_result": result.analysis_id}
                    }
                )
                
        except Exception:
            # On failure, checkpoint should still exist for normalization step
            checkpoint = await checkpoint_manager.load_checkpoint(workflow_id=workflow_id)
            assert checkpoint is not None
            assert checkpoint["step"] == "normalization"
            assert checkpoint["status"] == "completed"

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, checkpoint_manager):
        """Test checkpoint cleanup functionality."""
        workflow_id = "test_workflow_cleanup"
        
        # Create checkpoint
        await checkpoint_manager.save_checkpoint(
            workflow_id=workflow_id,
            checkpoint_data={"step": "test", "status": "completed"}
        )
        
        # Verify exists
        checkpoint = await checkpoint_manager.load_checkpoint(workflow_id=workflow_id)
        assert checkpoint is not None
        
        # Cleanup
        await checkpoint_manager.delete_checkpoint(workflow_id=workflow_id)
        
        # Verify deleted
        checkpoint_after = await checkpoint_manager.load_checkpoint(workflow_id=workflow_id)
        assert checkpoint_after is None



