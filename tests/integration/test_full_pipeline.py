"""
Integration tests for full normalization → analysis pipeline.

Tests complete workflows from input data to analysis results.
"""

import pytest
import asyncio
from pathwaylens_core.normalization.normalizer import NormalizationEngine
from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.analysis.schemas import AnalysisType, DatabaseType, AnalysisParameters
from pathwaylens_core.data import DatabaseManager


@pytest.mark.integration
class TestFullPipeline:
    """Test suite for full analysis pipeline."""
    
    @pytest.fixture
    def database_manager(self):
        """Create database manager."""
        return DatabaseManager()
    
    @pytest.fixture
    def normalizer(self):
        """Create normalization engine."""
        return NormalizationEngine()
    
    @pytest.fixture
    def analysis_engine(self, database_manager):
        """Create analysis engine."""
        return AnalysisEngine(database_manager)
    
    @pytest.mark.asyncio
    async def test_normalize_then_analyze_ora(
        self,
        normalizer,
        analysis_engine,
        sample_gene_list
    ):
        """Test full pipeline: normalize → ORA analysis."""
        # Step 1: Normalize gene IDs
        normalized = await normalizer.normalize(
            input_data=sample_gene_list,
            species="human",
            input_id_type="symbol",
            output_id_type="ensembl"
        )
        
        assert normalized is not None
        assert "normalized_data" in normalized or "gene_ids" in normalized
        
        # Extract normalized gene IDs
        if "normalized_data" in normalized:
            gene_ids = normalized["normalized_data"]
        else:
            gene_ids = normalized["gene_ids"]
        
        # Step 2: Perform ORA analysis
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05,
            min_pathway_size=5,
            max_pathway_size=500
        )
        
        result = await analysis_engine.analyze(
            input_data=gene_ids if isinstance(gene_ids, list) else list(gene_ids),
            parameters=parameters
        )
        
        assert result is not None
        assert result.analysis_type == AnalysisType.ORA
        assert len(result.database_results) > 0
    
    @pytest.mark.asyncio
    async def test_normalize_then_analyze_gsea(
        self,
        normalizer,
        analysis_engine,
        sample_gene_list
    ):
        """Test full pipeline: normalize → GSEA analysis."""
        # Normalize
        normalized = await normalizer.normalize(
            input_data=sample_gene_list,
            species="human",
            input_id_type="symbol",
            output_id_type="ensembl"
        )
        
        # Extract gene IDs
        if "normalized_data" in normalized:
            gene_ids = normalized["normalized_data"]
        else:
            gene_ids = normalized.get("gene_ids", sample_gene_list)
        
        # GSEA analysis
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.GSEA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05,
            min_pathway_size=15,
            max_pathway_size=500
        )
        
        result = await analysis_engine.analyze(
            input_data=gene_ids if isinstance(gene_ids, list) else list(gene_ids),
            parameters=parameters
        )
        
        assert result is not None
        assert result.analysis_type == AnalysisType.GSEA
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_processing(
        self,
        normalizer,
        analysis_engine
    ):
        """Test batch processing through pipeline."""
        # Multiple gene lists
        batch_gene_lists = [
            ["GENE1", "GENE2", "GENE3"],
            ["GENE4", "GENE5", "GENE6"],
            ["GENE7", "GENE8", "GENE9"]
        ]
        
        results = []
        
        for gene_list in batch_gene_lists:
            # Normalize
            normalized = await normalizer.normalize(
                input_data=gene_list,
                species="human",
                input_id_type="symbol",
                output_id_type="ensembl"
            )
            
            # Extract gene IDs
            if "normalized_data" in normalized:
                gene_ids = normalized["normalized_data"]
            else:
                gene_ids = normalized.get("gene_ids", gene_list)
            
            # Analyze
            parameters = AnalysisParameters(
                analysis_type=AnalysisType.ORA,
                databases=[DatabaseType.KEGG],
                species="human"
            )
            
            result = await analysis_engine.analyze(
                input_data=gene_ids if isinstance(gene_ids, list) else list(gene_ids),
                parameters=parameters
            )
            
            results.append(result)
        
        assert len(results) == len(batch_gene_lists)
        assert all(r is not None for r in results)



