"""
Main analysis engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
from loguru import logger

from .schemas import (
    AnalysisResult, AnalysisParameters, AnalysisType, DatabaseType,
    PathwayResult, DatabaseResult, ConsensusResult
)
from .ora_engine import ORAEngine
from .gsea_engine import GSEAEngine
from .gsva_engine import GSVAEngine
from .topology_engine import TopologyEngine
from .multi_omics_engine import MultiOmicsEngine
from .consensus_engine import ConsensusEngine
from ..data import DatabaseManager


class AnalysisEngine:
    """Main analysis engine for pathway enrichment analysis."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the analysis engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="analysis_engine")
        self.database_manager = database_manager or DatabaseManager()
        
        # Initialize analysis engines
        self.ora_engine = ORAEngine(self.database_manager)
        self.gsea_engine = GSEAEngine(self.database_manager)
        self.consensus_engine = ConsensusEngine()
        self.gsva_engine = GSVAEngine(self.database_manager)
        self.topology_engine = TopologyEngine(self.database_manager)
        self.multi_omics_engine = MultiOmicsEngine(self.database_manager)
    
    async def analyze(
        self,
        input_data: Union[str, pd.DataFrame, List[str]],
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Perform pathway analysis.
        
        Args:
            input_data: Input data (file path, DataFrame, or gene list)
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with analysis information
        """
        job_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"Starting analysis job {job_id}")
        
        input_info = {}  # Initialize input_info early
        try:
            # Step 1: Prepare input data
            gene_list, input_info = await self._prepare_input_data(input_data)
            
            # Step 2: Validate parameters
            self._validate_parameters(parameters, gene_list)
            
            # Step 3: Perform analysis by type
            if parameters.analysis_type == AnalysisType.ORA:
                database_results = await self._perform_ora_analysis(gene_list, parameters)
            elif parameters.analysis_type == AnalysisType.GSEA:
                database_results = await self._perform_gsea_analysis(gene_list, parameters)
            elif parameters.analysis_type == AnalysisType.GSVA:
                database_results = await self._perform_gsva_analysis(gene_list, parameters)
            elif parameters.analysis_type == AnalysisType.TOPOLOGY:
                database_results = await self._perform_topology_analysis(gene_list, parameters)
            elif parameters.analysis_type == AnalysisType.MULTI_OMICS:
                database_results = await self._perform_multi_omics_analysis(gene_list, parameters)
            else:
                raise ValueError(f"Unsupported analysis type: {parameters.analysis_type}")
            
            # Step 4: Perform consensus analysis if multiple databases
            consensus_results = None
            if len(parameters.databases) > 1:
                consensus_results = await self._perform_consensus_analysis(
                    database_results, parameters
                )
            
            # Step 5: Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(
                database_results, consensus_results
            )
            
            # Step 6: Generate output files
            output_files = {}
            if output_dir:
                output_files = await self._generate_output_files(
                    job_id, database_results, consensus_results, 
                    parameters, output_dir
                )
            
            # Step 7: Create analysis result
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = AnalysisResult(
                job_id=job_id,
                analysis_type=parameters.analysis_type,
                parameters=parameters,
                input_file=input_info.get('file_path', ''),
                input_gene_count=len(gene_list),
                input_species=parameters.species,
                database_results=database_results,
                consensus_results=consensus_results,
                total_pathways=summary_stats['total_pathways'],
                significant_pathways=summary_stats['significant_pathways'],
                significant_databases=summary_stats['significant_databases'],
                overall_quality=summary_stats['overall_quality'],
                reproducibility=summary_stats['reproducibility'],
                created_at=start_time.isoformat(),
                completed_at=end_time.isoformat(),
                processing_time=processing_time,
                output_files=output_files
            )
            
            self.logger.info(f"Analysis job {job_id} completed successfully")
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Analysis job {job_id} failed: {error_msg}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Provide more helpful error messages
            user_friendly_error = self._format_error_message(error_msg, input_info)
            
            return AnalysisResult(
                job_id=job_id,
                analysis_type=parameters.analysis_type,
                parameters=parameters,
                input_file=input_info.get('file_path', ''),
                input_gene_count=0,
                input_species=parameters.species,
                database_results={},
                total_pathways=0,
                significant_pathways=0,
                significant_databases=0,
                overall_quality=0.0,
                reproducibility=0.0,
                created_at=start_time.isoformat(),
                completed_at=end_time.isoformat(),
                processing_time=processing_time,
                errors=[user_friendly_error]
            )
    
    def _format_error_message(self, error_msg: str, input_info: Dict[str, Any]) -> str:
        """
        Format error message to be more user-friendly with suggestions.
        
        Args:
            error_msg: Original error message
            input_info: Input data information
            
        Returns:
            Formatted error message with suggestions
        """
        base_msg = f"Analysis failed: {error_msg}"
        suggestions = []
        
        # Check for common error patterns and provide suggestions
        error_lower = error_msg.lower()
        
        if "file" in error_lower or "not found" in error_lower:
            suggestions.append("Please verify that the input file path is correct and the file exists.")
            if input_info.get('file_path'):
                suggestions.append(f"Attempted to read: {input_info['file_path']}")
        
        if "database" in error_lower or "connection" in error_lower:
            suggestions.append("Database connection issue detected.")
            suggestions.append("Please check your internet connection and database availability.")
            suggestions.append("You can try again later or use cached results if available.")
        
        if "empty" in error_lower or "no genes" in error_lower:
            suggestions.append("No genes were found in the input data.")
            suggestions.append("Please verify that your input file contains gene identifiers.")
            suggestions.append("Common gene identifier columns: 'gene', 'gene_id', 'gene_symbol', 'symbol'")
        
        if "species" in error_lower or "invalid" in error_lower:
            suggestions.append("Species validation failed.")
            suggestions.append("Please verify that the specified species is supported.")
            suggestions.append("Supported species: human, mouse, rat, drosophila, zebrafish, c_elegans, s_cerevisiae")
        
        if "timeout" in error_lower:
            suggestions.append("Request timed out. This may be due to:")
            suggestions.append("  - Large input dataset (consider processing in batches)")
            suggestions.append("  - Slow network connection")
            suggestions.append("  - Database server overload")
            suggestions.append("Please try again with a smaller dataset or check your network connection.")
        
        if suggestions:
            return f"{base_msg}\n\nSuggestions:\n" + "\n".join(f"  • {s}" for s in suggestions)
        
        return base_msg
    
    async def _prepare_input_data(
        self, 
        input_data: Union[str, pd.DataFrame, List[str]]
    ) -> tuple[List[str], Dict[str, Any]]:
        """Prepare input data for analysis."""
        input_info = {}
        
        if isinstance(input_data, str):
            # File path
            input_info['file_path'] = input_data
            df = pd.read_csv(input_data)
            gene_list = self._extract_gene_list(df)
        elif isinstance(input_data, pd.DataFrame):
            # DataFrame
            gene_list = self._extract_gene_list(input_data)
        elif isinstance(input_data, list):
            # Gene list
            gene_list = input_data
        else:
            raise ValueError("Unsupported input data type")
        
        if not gene_list:
            raise ValueError("No genes found in input data")
        
        self.logger.info(f"Prepared {len(gene_list)} genes for analysis")
        return gene_list, input_info
    
    def _extract_gene_list(self, df: pd.DataFrame) -> List[str]:
        """Extract gene list from DataFrame."""
        gene_list = []
        
        # Look for gene identifier columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if this looks like gene identifiers
                sample_values = df[col].dropna().head(100)
                if self._looks_like_gene_ids(sample_values):
                    gene_list = df[col].dropna().unique().tolist()
                    break
        
        return gene_list
    
    def _looks_like_gene_ids(self, values: pd.Series) -> bool:
        """Check if values look like gene identifiers."""
        # Check for common gene ID patterns
        patterns = [
            r'^[A-Za-z][A-Za-z0-9]*$',  # Gene symbols
            r'^ENS[A-Z]*G\d{11}$',      # Ensembl IDs
            r'^\d+$'                    # Entrez IDs
        ]
        
        for pattern in patterns:
            matches = values.str.match(pattern).sum()
            if matches / len(values) >= 0.8:
                return True
        
        return False
    
    def _validate_parameters(self, parameters: AnalysisParameters, gene_list: List[str]):
        """Validate analysis parameters."""
        if parameters is None:
            raise ValueError("Analysis parameters cannot be None")
        
        if not parameters.databases:
            raise ValueError("At least one database must be specified")
        
        if len(gene_list) < parameters.min_pathway_size:
            raise ValueError(f"Input gene list too small (minimum: {parameters.min_pathway_size})")
        
        if parameters.min_pathway_size > parameters.max_pathway_size:
            raise ValueError("Minimum pathway size cannot be greater than maximum pathway size")
    
    async def _perform_ora_analysis(
        self, 
        gene_list: List[str], 
        parameters: AnalysisParameters
    ) -> Dict[str, DatabaseResult]:
        """Perform ORA analysis."""
        database_results = {}
        
        for database in parameters.databases:
            self.logger.info(f"Performing ORA analysis with {database.value}")
            
            try:
                result = await self.ora_engine.analyze(
                    gene_list=gene_list,
                    database=database,
                    species=parameters.species,
                    significance_threshold=parameters.significance_threshold,
                    correction_method=parameters.correction_method,
                    min_pathway_size=parameters.min_pathway_size,
                    max_pathway_size=parameters.max_pathway_size
                )
                
                database_results[database.value] = result
                
            except Exception as e:
                self.logger.error(f"ORA analysis failed for {database.value}: {e}")
                # Create empty result
                database_results[database.value] = DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=parameters.species,
                    coverage=0.0
                )
        
        return database_results

    async def _perform_gsva_analysis(
        self, 
        gene_list: List[str], 
        parameters: AnalysisParameters
    ) -> Dict[str, DatabaseResult]:
        """Perform GSVA analysis."""
        database_results = {}
        for database in parameters.databases:
            self.logger.info(f"Performing GSVA analysis with {database.value}")
            try:
                result = await self.gsva_engine.analyze(
                    gene_list=gene_list,
                    database=database,
                    species=parameters.species,
                    min_size=parameters.gsea_min_size,
                    max_size=parameters.gsea_max_size
                )
                database_results[database.value] = result
            except Exception as e:
                self.logger.error(f"GSVA analysis failed for {database.value}: {e}")
                database_results[database.value] = DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=parameters.species,
                    coverage=0.0
                )
        return database_results

    async def _perform_topology_analysis(
        self, 
        gene_list: List[str], 
        parameters: AnalysisParameters
    ) -> Dict[str, DatabaseResult]:
        """Perform topology-based (SPIA-like) analysis."""
        database_results = {}
        for database in parameters.databases:
            self.logger.info(f"Performing topology analysis with {database.value}")
            try:
                result = await self.topology_engine.analyze(
                    gene_list=gene_list,
                    database=database,
                    species=parameters.species,
                    significance_threshold=parameters.significance_threshold,
                )
                database_results[database.value] = result
            except Exception as e:
                self.logger.error(f"Topology analysis failed for {database.value}: {e}")
                database_results[database.value] = DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=parameters.species,
                    coverage=0.0
                )
        return database_results

    async def _perform_multi_omics_analysis(
        self, 
        gene_list: List[str], 
        parameters: AnalysisParameters
    ) -> Dict[str, DatabaseResult]:
        """Perform multi-omics pathway analysis (placeholder wiring)."""
        database_results = {}
        for database in parameters.databases:
            self.logger.info(f"Performing multi-omics analysis with {database.value}")
            try:
                result = await self.multi_omics_engine.analyze(
                    gene_list=gene_list,
                    database=database,
                    species=parameters.species,
                )
                database_results[database.value] = result
            except Exception as e:
                self.logger.error(f"Multi-omics analysis failed for {database.value}: {e}")
                database_results[database.value] = DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=parameters.species,
                    coverage=0.0
                )
        return database_results
    
    async def _perform_gsea_analysis(
        self, 
        gene_list: List[str], 
        parameters: AnalysisParameters
    ) -> Dict[str, DatabaseResult]:
        """Perform GSEA analysis."""
        database_results = {}
        
        for database in parameters.databases:
            self.logger.info(f"Performing GSEA analysis with {database.value}")
            
            try:
                result = await self.gsea_engine.analyze(
                    gene_list=gene_list,
                    database=database,
                    species=parameters.species,
                    significance_threshold=parameters.significance_threshold,
                    correction_method=parameters.correction_method,
                    permutations=parameters.gsea_permutations,
                    min_size=parameters.gsea_min_size,
                    max_size=parameters.gsea_max_size
                )
                
                database_results[database.value] = result
                
            except Exception as e:
                self.logger.error(f"GSEA analysis failed for {database.value}: {e}")
                # Create empty result
                database_results[database.value] = DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=parameters.species,
                    coverage=0.0
                )
        
        return database_results
    
    async def _perform_consensus_analysis(
        self,
        database_results: Dict[str, DatabaseResult],
        parameters: AnalysisParameters
    ) -> ConsensusResult:
        """Perform consensus analysis."""
        self.logger.info("Performing consensus analysis")
        
        try:
            result = await self.consensus_engine.analyze(
                database_results=database_results,
                method=parameters.consensus_method,
                min_databases=parameters.min_databases,
                significance_threshold=parameters.significance_threshold
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Consensus analysis failed: {e}")
            return ConsensusResult(
                consensus_method=parameters.consensus_method,
                total_pathways=0,
                significant_pathways=0,
                pathways=[],
                database_agreement={},
                consensus_score=0.0,
                reproducibility=0.0,
                stability=0.0
            )
    
    def _calculate_summary_statistics(
        self,
        database_results: Dict[str, DatabaseResult],
        consensus_results: Optional[ConsensusResult]
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        total_pathways = sum(result.total_pathways for result in database_results.values())
        significant_pathways = sum(result.significant_pathways for result in database_results.values())
        significant_databases = sum(1 for result in database_results.values() if result.significant_pathways > 0)
        
        # Calculate overall quality
        if database_results:
            coverage_scores = [result.coverage for result in database_results.values()]
            overall_quality = np.mean(coverage_scores)
        else:
            overall_quality = 0.0
        
        # Calculate reproducibility
        if consensus_results:
            reproducibility = consensus_results.reproducibility
        else:
            reproducibility = 0.0
        
        return {
            'total_pathways': total_pathways,
            'significant_pathways': significant_pathways,
            'significant_databases': significant_databases,
            'overall_quality': overall_quality,
            'reproducibility': reproducibility
        }
    
    async def _generate_output_files(
        self,
        job_id: str,
        database_results: Dict[str, DatabaseResult],
        consensus_results: Optional[ConsensusResult],
        parameters: AnalysisParameters,
        output_dir: str
    ) -> Dict[str, str]:
        """Generate output files."""
        output_files = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate JSON output
        json_file = output_path / f"{job_id}_results.json"
        # TODO: Implement JSON export
        output_files['json'] = str(json_file)
        
        # Generate CSV output
        csv_file = output_path / f"{job_id}_results.csv"
        # TODO: Implement CSV export
        output_files['csv'] = str(csv_file)
        
        return output_files
    
    def analyze_sync(
        self,
        input_data: Union[str, pd.DataFrame, List[str]],
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Synchronous version of analyze.
        
        Args:
            input_data: Input data (file path, DataFrame, or gene list)
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with analysis information
        """
        return asyncio.run(self.analyze(input_data, parameters, output_dir))
