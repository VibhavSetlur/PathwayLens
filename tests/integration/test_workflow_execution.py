"""
Integration tests for workflow execution.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.workflow import WorkflowManager, WorkflowValidationError
from pathwaylens_core.analysis.schemas import AnalysisType, DatabaseType


class TestWorkflowExecution:
    """Test cases for workflow execution."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def workflow_manager(self):
        """Create a workflow manager instance."""
        return WorkflowManager()

    @pytest.fixture
    def sample_workflow_spec(self, temp_dir):
        """Create a sample workflow specification."""
        gene_file = temp_dir / "test_genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\nGENE4\nGENE5\n")
        
        return {
            "steps": [
                {
                    "step_id": "normalize_genes",
                    "type": "normalization",
                    "input": str(gene_file),
                    "species": "human",
                    "input_type": "gene_list",
                    "target_type": "symbol",
                },
                {
                    "step_id": "analyze_pathways",
                    "type": "analysis",
                    "input": "normalize_genes",
                    "method": "ORA",
                    "databases": ["kegg"],
                    "species": "human",
                    "fdr": 0.05,
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_workflow_validation(self, workflow_manager, sample_workflow_spec, temp_dir):
        """Test workflow validation."""
        # Create a workflow file
        workflow_file = temp_dir / "test_workflow.yaml"
        import yaml
        workflow_file.write_text(yaml.dump(sample_workflow_spec))
        
        # Load and validate workflow
        spec = workflow_manager.load(workflow_file)
        steps = workflow_manager.validate(spec)
        
        assert len(steps) == 2
        assert steps[0].step_id == "normalize_genes"
        assert steps[0].type == "normalization"
        assert steps[1].step_id == "analyze_pathways"
        assert steps[1].type == "analysis"

    @pytest.mark.asyncio
    async def test_workflow_validation_duplicate_ids(self, workflow_manager, temp_dir):
        """Test workflow validation with duplicate step IDs."""
        invalid_spec = {
            "steps": [
                {"step_id": "step1", "type": "normalization", "input": "test.txt"},
                {"step_id": "step1", "type": "analysis", "input": "step1"},
            ]
        }
        
        workflow_file = temp_dir / "invalid_workflow.yaml"
        import yaml
        workflow_file.write_text(yaml.dump(invalid_spec))
        
        spec = workflow_manager.load(workflow_file)
        with pytest.raises(WorkflowValidationError, match="Duplicate step_id"):
            workflow_manager.validate(spec)

    @pytest.mark.asyncio
    async def test_workflow_validation_invalid_type(self, workflow_manager, temp_dir):
        """Test workflow validation with invalid step type."""
        invalid_spec = {
            "steps": [
                {"step_id": "step1", "type": "invalid_type", "input": "test.txt"},
            ]
        }
        
        workflow_file = temp_dir / "invalid_workflow.yaml"
        import yaml
        workflow_file.write_text(yaml.dump(invalid_spec))
        
        spec = workflow_manager.load(workflow_file)
        with pytest.raises(WorkflowValidationError, match="unsupported type"):
            workflow_manager.validate(spec)

    @pytest.mark.asyncio
    async def test_workflow_execution_normalization_and_analysis(self, workflow_manager, sample_workflow_spec, temp_dir):
        """Test workflow execution with normalization and analysis steps."""
        # Mock the normalization and analysis engines
        with patch.object(workflow_manager.normalizer, 'normalize_identifiers') as mock_norm, \
             patch.object(workflow_manager.analysis_engine, 'analyze') as mock_analysis:
            
            # Mock normalization result
            mock_norm_result = Mock()
            mock_norm_result.normalized_data = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
            mock_norm_result.mapping_stats = {"total": 5, "converted": 5, "failed": 0}
            mock_norm.return_value = mock_norm_result
            
            # Mock analysis result
            mock_analysis_result = Mock()
            mock_analysis_result.analysis_id = "test_analysis_1"
            mock_analysis_result.analysis_name = "test_analysis"
            mock_analysis.return_value = mock_analysis_result
            
            # Validate and run workflow
            steps = workflow_manager.validate(sample_workflow_spec)
            result = await workflow_manager.run(steps)
            
            # Verify results
            assert result["workflow_id"] is not None
            assert len(result["steps"]) == 2
            assert result["steps"][0]["step_id"] == "normalize_genes"
            assert result["steps"][0]["status"] == "completed"
            assert result["steps"][1]["step_id"] == "analyze_pathways"
            assert result["steps"][1]["status"] == "completed"
            
            # Verify engines were called
            mock_norm.assert_called_once()
            mock_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_execution_with_comparison(self, workflow_manager, temp_dir):
        """Test workflow execution with comparison step."""
        gene_file = temp_dir / "test_genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        workflow_spec = {
            "steps": [
                {
                    "step_id": "normalize1",
                    "type": "normalization",
                    "input": str(gene_file),
                    "species": "human",
                },
                {
                    "step_id": "analyze1",
                    "type": "analysis",
                    "input": "normalize1",
                    "method": "ORA",
                    "databases": ["kegg"],
                },
                {
                    "step_id": "analyze2",
                    "type": "analysis",
                    "input": "normalize1",
                    "method": "GSEA",
                    "databases": ["kegg"],
                },
                {
                    "step_id": "compare",
                    "type": "comparison",
                    "inputs": ["analyze1", "analyze2"],
                    "comparison_type": "pathway_concordance",
                }
            ]
        }
        
        # Mock all engines
        with patch.object(workflow_manager.normalizer, 'normalize_identifiers') as mock_norm, \
             patch.object(workflow_manager.analysis_engine, 'analyze') as mock_analysis, \
             patch.object(workflow_manager.comparison_engine, 'compare') as mock_compare:
            
            # Mock normalization result
            mock_norm.return_value = Mock(normalized_data=["GENE1", "GENE2", "GENE3"])
            
            # Mock analysis results
            mock_analysis.return_value = Mock(analysis_id="test_analysis")
            
            # Mock comparison result
            mock_compare.return_value = Mock(comparison_id="test_comparison")
            
            # Run workflow
            steps = workflow_manager.validate(workflow_spec)
            result = await workflow_manager.run(steps)
            
            # Verify results
            assert len(result["steps"]) == 4
            assert result["steps"][3]["step_id"] == "compare"
            assert result["steps"][3]["status"] == "completed"
            
            # Verify comparison was called with correct inputs
            mock_compare.assert_called_once()
            call_args = mock_compare.call_args
            assert len(call_args[1]["analysis_results"]) == 2

    @pytest.mark.asyncio
    async def test_workflow_execution_error_handling(self, workflow_manager, temp_dir):
        """Test workflow execution error handling."""
        gene_file = temp_dir / "test_genes.txt"
        gene_file.write_text("GENE1\nGENE2\n")
        
        workflow_spec = {
            "steps": [
                {
                    "step_id": "normalize",
                    "type": "normalization",
                    "input": str(gene_file),
                    "species": "human",
                },
                {
                    "step_id": "analyze",
                    "type": "analysis",
                    "input": "normalize",
                    "method": "ORA",
                }
            ]
        }
        
        # Mock normalization to raise an error
        with patch.object(workflow_manager.normalizer, 'normalize_identifiers') as mock_norm:
            mock_norm.side_effect = Exception("Normalization failed")
            
            steps = workflow_manager.validate(workflow_spec)
            
            # Workflow should propagate the error
            with pytest.raises(Exception, match="Normalization failed"):
                await workflow_manager.run(steps)

    @pytest.mark.asyncio
    async def test_workflow_execution_missing_input(self, workflow_manager):
        """Test workflow execution with missing input reference."""
        workflow_spec = {
            "steps": [
                {
                    "step_id": "analyze",
                    "type": "analysis",
                    "input": "nonexistent_step",
                    "method": "ORA",
                }
            ]
        }
        
        steps = workflow_manager.validate(workflow_spec)
        
        # The workflow should handle missing input gracefully or raise an error
        # This depends on implementation - testing that it doesn't crash silently
        with patch.object(workflow_manager.analysis_engine, 'analyze') as mock_analysis:
            mock_analysis.side_effect = KeyError("nonexistent_step")
            
            with pytest.raises(Exception):
                await workflow_manager.run(steps)

    @pytest.mark.asyncio
    async def test_workflow_batch_job_submission(self, workflow_manager, sample_workflow_spec):
        """Test batch job submission."""
        job_id = await workflow_manager.submit_batch_job(
            workflow_spec=sample_workflow_spec,
            priority=5,
            max_retries=3
        )
        
        assert job_id is not None
        
        # Check job status
        job = await workflow_manager.get_job_status(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.priority == 5

    @pytest.mark.asyncio
    async def test_workflow_batch_job_cancellation(self, workflow_manager, sample_workflow_spec):
        """Test batch job cancellation."""
        job_id = await workflow_manager.submit_batch_job(
            workflow_spec=sample_workflow_spec,
            priority=5
        )
        
        # Cancel the job
        cancelled = await workflow_manager.cancel_job(job_id)
        assert cancelled is True
        
        # Check job status
        job = await workflow_manager.get_job_status(job_id)
        assert job is not None



