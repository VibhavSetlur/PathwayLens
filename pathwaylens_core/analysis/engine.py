"""
Main analysis engine for PathwayLens.
"""

import asyncio
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import platform
import sys
import importlib.metadata
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
from ..visualization.comparison import ComparisonVisualizer
from ..visualization.network_renderer import NetworkRenderer
from ..reporting.html_report import HTMLReportGenerator
import json


class AnalysisEngine:
    """Main analysis engine for pathway enrichment analysis."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None, use_gprofiler: bool = True):
        """
        Initialize analysis engine.
        
        Args:
            database_manager: Database manager instance
            use_gprofiler: Whether to use g:Profiler for ORA
        """
        self.logger = logger.bind(module="analysis_engine")
        self.database_manager = database_manager or DatabaseManager()
        
        # Initialize analysis engines
        self.ora_engine = ORAEngine(self.database_manager, use_gprofiler=use_gprofiler)
        self.gsea_engine = GSEAEngine(self.database_manager)
        self.consensus_engine = ConsensusEngine()
        self.gsva_engine = GSVAEngine(self.database_manager)
        self.topology_engine = TopologyEngine(self.database_manager)
        self.topology_engine = TopologyEngine(self.database_manager)
        self.multi_omics_engine = MultiOmicsEngine(self.database_manager)
        
        # Initialize reporting and visualization
        self.html_generator = HTMLReportGenerator()
        self.network_renderer = NetworkRenderer()
    
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
            gene_list, input_info, gene_stats = await self._prepare_input_data(input_data, parameters.tool)
            
            # Filter by LFC if stats available
            if parameters.lfc_threshold > 0 and gene_stats:
                filtered_genes = []
                for gene in gene_list:
                    stats = gene_stats.get(gene)
                    if stats and 'logFC' in stats:
                        if abs(stats['logFC']) >= parameters.lfc_threshold:
                            filtered_genes.append(gene)
                    else:
                        # Keep genes without stats
                        filtered_genes.append(gene)
                
                if len(filtered_genes) < len(gene_list):
                    self.logger.info(f"Filtered {len(gene_list) - len(filtered_genes)} genes by LFC threshold {parameters.lfc_threshold}")
                    gene_list = filtered_genes
            
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
                    parameters, output_dir, gene_stats
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
                output_files=output_files,
                metadata={
                    "system": {
                        "os": platform.system(),
                        "os_release": platform.release(),
                        "python_version": sys.version,
                        "command": " ".join(sys.argv),
                        "pathwaylens_version": "1.0.0"  # Should be dynamic
                    },
                    "environment": dict(os.environ) if False else {} # Don't log full env for privacy
                }
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
        input_data: Union[str, pd.DataFrame, List[str]],
        tool: str = "auto"
    ) -> tuple[List[str], Dict[str, Any], Dict[str, Dict[str, float]]]:
        """Prepare input data for analysis."""
        input_info = {}
        gene_stats = {} # gene -> {logFC, pval, etc}
        
        if isinstance(input_data, str):
            # File path
            input_path = Path(input_data)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            input_info['file_path'] = str(input_path)
            
            # Try to read the file - detect separator
            try:
                # First, try to detect if it's tab-separated
                with open(input_path, 'r') as f:
                    first_line = f.readline()
                    if '\t' in first_line and ',' not in first_line:
                        # Tab-separated
                        df = pd.read_csv(input_path, sep='\t')
                    else:
                        # Comma-separated or auto-detect
                        df = pd.read_csv(input_path)
            except Exception as e:
                self.logger.error(f"Failed to read input file: {e}")
                raise ValueError(f"Could not read input file: {input_path}")
            
            gene_list = self._parse_tool_output(df, tool)
            gene_stats = self._extract_gene_stats(df)
        elif isinstance(input_data, pd.DataFrame):
            # DataFrame
            gene_list = self._parse_tool_output(input_data, tool)
            gene_stats = self._extract_gene_stats(input_data)
        elif isinstance(input_data, list):
            # Gene list
            gene_list = input_data
        else:
            raise ValueError("Unsupported input data type")
        
        if not gene_list:
            raise ValueError("No genes found in input data")
        
        self.logger.info(f"Prepared {len(gene_list)} genes for analysis")
        return gene_list, input_info, gene_stats

    def _parse_tool_output(self, df: pd.DataFrame, tool: str) -> List[str]:
        """Parse output from specific tools."""
        tool = tool.lower()
        
        if tool == "auto":
            tool = self._detect_tool_format(df)
            self.logger.info(f"Auto-detected tool format: {tool}")
            
        if tool == "deseq2":
            # Look for gene names in index or specific columns
            # DESeq2 usually has gene IDs as index
            if df.index.name in ['gene', 'gene_id', 'id'] or (df.index.dtype == 'object' and self._looks_like_gene_ids(df.index.to_series())):
                return df.index.tolist()
            # Or look for columns
            return self._extract_gene_list(df)
            
        elif tool == "limma":
            # Limma often has 'ID' or 'Symbol'
            if 'ID' in df.columns:
                return df['ID'].tolist()
            if 'Symbol' in df.columns:
                return df['Symbol'].tolist()
            return self._extract_gene_list(df)
            
        elif tool.lower() == 'maxquant':
            # MaxQuant protein groups - the df is already loaded
            # MaxQuant uses 'Gene names' column
            if 'Gene names' in df.columns:
                genes = df['Gene names'].dropna().astype(str).tolist()
                # Split multiple gene names (semicolon-separated)
                all_genes = []
                for gene_str in genes:
                    if gene_str and gene_str != 'nan':
                        all_genes.extend([g.strip() for g in gene_str.split(';') if g.strip()])
                return list(set(all_genes))  # Remove duplicates
            elif 'gene' in df.columns:
                return df['gene'].dropna().astype(str).tolist()
            else:
                self.logger.warning(f"Could not find gene column in MaxQuant file. Columns: {df.columns.tolist()}")
                return self._extract_gene_list(df)
            
        elif tool == "edger":
            # EdgeR often has gene IDs as index
            if df.index.dtype == 'object':
                return df.index.tolist()
            return self._extract_gene_list(df)
            
        elif tool == "maxquant":
            # MaxQuant 'Gene names' column
            if 'Gene names' in df.columns:
                # MaxQuant can have multiple genes separated by semicolon
                genes = []
                for g in df['Gene names'].dropna():
                    genes.extend(g.split(';'))
                return list(set(genes))
            return self._extract_gene_list(df)
            
        else:
            # Fallback to generic extraction
            return self._extract_gene_list(df)

    def _extract_gene_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Extract statistics (logFC, p-value) for genes."""
        stats = {}
        
        # Identify columns
        logfc_col = next((c for c in df.columns if c.lower() in ['log2foldchange', 'logfc', 'log2fc']), None)
        pval_col = next((c for c in df.columns if c.lower() in ['pvalue', 'p.value', 'pval', 'p_value']), None)
        padj_col = next((c for c in df.columns if c.lower() in ['padj', 'adj.p.val', 'fdr', 'qvalue']), None)
        
        # Identify gene column (or index)
        gene_col = None
        for col in df.columns:
            if df[col].dtype == 'object' and self._looks_like_gene_ids(df[col].dropna().head(100)):
                gene_col = col
                break
        
        # Iterate and populate
        try:
            for idx, row in df.iterrows():
                gene = row[gene_col] if gene_col else (idx if isinstance(idx, str) else str(idx))
                
                gene_stat = {}
                if logfc_col: gene_stat['logFC'] = float(row[logfc_col])
                if pval_col: gene_stat['p_value'] = float(row[pval_col])
                if padj_col: gene_stat['adj_p_value'] = float(row[padj_col])
                
                if gene_stat:
                    stats[str(gene)] = gene_stat
        except Exception as e:
            self.logger.warning(f"Failed to extract gene stats: {e}")
            
        return stats

    def _detect_tool_format(self, df: pd.DataFrame) -> str:
        """Detect tool format based on columns."""
        cols = set(df.columns)
        
        if {'baseMean', 'log2FoldChange', 'padj'}.issubset(cols):
            return "deseq2"
        if {'logFC', 'AveExpr', 'P.Value', 'adj.P.Val'}.issubset(cols):
            return "limma"
        if {'logFC', 'logCPM', 'PValue', 'FDR'}.issubset(cols):
            return "edger"
        if 'Gene names' in cols and 'Protein IDs' in cols:
            return "maxquant"
            
        return "generic"
    
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
        
        # If no column found, check index
        if not gene_list and df.index.dtype == 'object':
             sample_values = df.index.to_series().dropna().head(100)
             if self._looks_like_gene_ids(sample_values):
                 gene_list = df.index.unique().tolist()
        
        return gene_list
    
    def _looks_like_gene_ids(self, values: pd.Series) -> bool:
        """Check if values look like gene identifiers."""
        if len(values) == 0:
            return False
            
        # Check for common gene ID patterns
        patterns = [
            r'^[A-Za-z][A-Za-z0-9\-\.]*$',  # Gene symbols (allow dots/dashes)
            r'^ENS[A-Z]*G\d{11}(\.\d+)?$',  # Ensembl IDs (allow version)
            r'^\d+$'                        # Entrez IDs
        ]
        
        for pattern in patterns:
            matches = values.astype(str).str.match(pattern).sum()
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
        
        # Create tasks for concurrent execution
        tasks = []
        for database in parameters.databases:
            self.logger.info(f"Scheduling ORA analysis with {database.value}")
            task = self.ora_engine.analyze(
                gene_list=gene_list,
                database=database,
                species=parameters.species,
                significance_threshold=parameters.significance_threshold,
                correction_method=parameters.correction_method,
                min_pathway_size=parameters.min_pathway_size,
                max_pathway_size=parameters.max_pathway_size,
                background_genes=parameters.custom_background,
                background_size=parameters.background_size
            )
            tasks.append(task)
            
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for database, result in zip(parameters.databases, results):
            if isinstance(result, Exception):
                self.logger.error(f"Analysis failed for {database.value}: {result}")
                # Create empty result with error metadata
                database_results[database.value] = DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=parameters.species,
                    coverage=0.0,
                    metadata={"error": str(result)}
                )
                continue
                
            database_results[database.value] = result
        
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
        output_dir: str,
        gene_stats: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, str]:
        """Generate output files with proper directory structure."""
        output_files = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        # my_analysis/
        # ├── run.json
        # ├── kegg/
        # │   ├── enrichment_results.tsv
        # │   ├── gene_pathway_mapping.tsv
        # │   └── visualizations/
        # ├── reactome/
        # └── summary/
        #     └── combined_results.tsv
        
        # 1. Generate per-database results
        for db_name, result in database_results.items():
            db_dir = output_path / db_name
            db_dir.mkdir(exist_ok=True)
            
            # Create visualizations subdirectory
            viz_dir = db_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Enrichment results TSV
            if result.pathways:
                enrichment_data = []
                for pathway in result.pathways:
                    enrichment_data.append({
                        'pathway_id': pathway.pathway_id,
                        'pathway_name': pathway.pathway_name,
                        'p_value': pathway.p_value,
                        'adjusted_p_value': pathway.adjusted_p_value,
                        'overlap_count': pathway.overlap_count,
                        'pathway_count': pathway.pathway_count,
                        'input_count': pathway.input_count,
                        'fold_enrichment': pathway.fold_enrichment,
                        'overlapping_genes': ','.join(pathway.overlapping_genes) if pathway.overlapping_genes else ''
                    })
                
                df_enrich = pd.DataFrame(enrichment_data)
                enrich_file = db_dir / "enrichment_results.tsv"
                df_enrich.to_csv(enrich_file, sep='\t', index=False)
                output_files[f'{db_name}_enrichment'] = str(enrich_file)
                
                # Gene-pathway mapping TSV
                gene_pathway_data = []
                for pathway in result.pathways:
                    for gene in pathway.overlapping_genes:
                        gene_pathway_data.append({
                            'gene': gene,
                            'pathway_id': pathway.pathway_id,
                            'pathway_name': pathway.pathway_name,
                            'p_value': pathway.adjusted_p_value
                        })
                        
                        # Add stats if available
                        if gene_stats and gene in gene_stats:
                            gene_pathway_data[-1].update(gene_stats[gene])
                
                if gene_pathway_data:
                    df_mapping = pd.DataFrame(gene_pathway_data)
                    mapping_file = db_dir / "gene_pathway_mapping.tsv"
                    df_mapping.to_csv(mapping_file, sep='\t', index=False)
                    output_files[f'{db_name}_mapping'] = str(mapping_file)
        
        # 2. Generate summary directory
        summary_dir = output_path / "summary"
        summary_dir.mkdir(exist_ok=True)
        
        # Combined results across all databases
        combined_data = []
        for db_name, result in database_results.items():
            for pathway in result.pathways:
                combined_data.append({
                    'database': db_name,
                    'pathway_id': pathway.pathway_id,
                    'pathway_name': pathway.pathway_name,
                    'p_value': pathway.p_value,
                    'adjusted_p_value': pathway.adjusted_p_value,
                    'overlap_count': pathway.overlap_count,
                    'pathway_count': pathway.pathway_count,
                    'fold_enrichment': pathway.fold_enrichment
                })
        
        if combined_data:
            df_combined = pd.DataFrame(combined_data)
            combined_file = summary_dir / "combined_results.tsv"
            df_combined.to_csv(combined_file, sep='\t', index=False)
            output_files['combined_results'] = str(combined_file)
        
        # 3. Generate visualizations
        if parameters.include_plots:
            visualizer = ComparisonVisualizer(str(summary_dir))
            
            # Database comparison
            db_comp_file = visualizer.plot_database_comparison(database_results)
            if db_comp_file:
                output_files['database_comparison_plot'] = db_comp_file
                
            # Pathway overlap (if multiple databases)
            if len(database_results) > 1:
                overlap_file = visualizer.plot_pathway_overlap(database_results)
                if overlap_file:
                    output_files['pathway_overlap_plot'] = overlap_file
                    
                consistency_file = visualizer.plot_enrichment_consistency(database_results)
                if consistency_file:
                    output_files['enrichment_consistency_plot'] = consistency_file
        
        # 4. Generate run.json at root
        run_data = {
            "run_name": parameters.model_dump().get('run_name', job_id),
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters.model_dump(),
            "results": {
                db: result.model_dump() for db, result in database_results.items()
            },
            "database_comparison": {
                "total_pathways": sum(r.total_pathways for r in database_results.values()),
                "significant_pathways": sum(r.significant_pathways for r in database_results.values()),
                "databases_analyzed": list(database_results.keys()),
                "agreement_score": consensus_results.consensus_score if consensus_results else None
            }
        }
        
        json_file = output_path / "run.json"
        with open(json_file, 'w') as f:
            json.dump(run_data, f, indent=2, default=str)
        output_files['json'] = str(json_file)
        
        # Save as analysis_metadata.json for reproducibility
        metadata_file = output_path / "analysis_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(run_data, f, indent=2, default=str)
        output_files['metadata'] = str(metadata_file)
        
        # 6. Generate HTML Report
        html_file = output_path / "report.html"
        
        # Generate plots for report
        report_plots = {}
        if parameters.include_networks and database_results:
            # Generate Enrichment Map
            for db_name, result in database_results.items():
                if result.pathways:
                    network_data = self.network_renderer.create_enrichment_map(
                        result.pathways,
                        title=f"Enrichment Map - {db_name}"
                    )
                    if network_data:
                        fig = self.network_renderer.create_interactive_network_plot(network_data)
                        report_plots[f"Enrichment Map ({db_name})"] = fig.to_html(full_html=False, include_plotlyjs='cdn')
                        
                        # Save standalone map
                        map_file = output_path / db_name / "visualizations" / "enrichment_map.html"
                        fig.write_html(str(map_file))
                        output_files[f'{db_name}_enrichment_map'] = str(map_file)

        self.html_generator.generate_report(
            analysis_result=run_data,
            output_file=str(html_file),
            plots=report_plots
        )
        output_files['html_report'] = str(html_file)
        
        # 7. Legacy results.csv for backwards compatibility
        if combined_data:
            csv_file = output_path / "results.csv"
            df_combined.to_csv(csv_file, index=False)
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
