"""
Over-Representation Analysis (ORA) engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from datetime import datetime
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from .schemas import (
    AnalysisType, DatabaseType, CorrectionMethod,
    AnalysisParameters, PathwayResult, DatabaseResult
)
from ..data.database_manager import DatabaseManager
from .statistical_utils import calculate_enrichment_statistics, calculate_statistical_power


class ORAEngine:
    """Over-Representation Analysis engine."""
    
    def __init__(self, database_manager: DatabaseManager, use_gprofiler: bool = True):
        """
        Initialize the ORA engine.
        
        Args:
            database_manager: Database manager instance
            use_gprofiler: Whether to use g:Profiler API when available
        """
        self.logger = logger.bind(module="ora_engine")
        self.database_manager = database_manager
        self.use_gprofiler = use_gprofiler
    
    async def analyze(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str,
        significance_threshold: float = 0.05,
        correction_method: CorrectionMethod = CorrectionMethod.FDR_BH,
        min_pathway_size: int = 5,
        max_pathway_size: int = 500,
        background_genes: Optional[List[str]] = None,
        background_size: Optional[int] = None
    ) -> DatabaseResult:
        """
        Perform Over-Representation Analysis.
        
        Args:
            gene_list: List of input genes
            database: Database to use for analysis
            species: Species for analysis
            significance_threshold: Significance threshold
            correction_method: Multiple testing correction method
            min_pathway_size: Minimum pathway size
            min_pathway_size: Minimum pathway size
            max_pathway_size: Maximum pathway size
            background_genes: Optional list of background genes
            background_size: Optional total number of background genes
            
        Returns:
            DatabaseResult with ORA analysis results
        """
        db_value = database.value if hasattr(database, 'value') else str(database)
        self.logger.info(f"Starting ORA analysis with {db_value} for {species}")
        self.logger.warning("Methodological Note: ORA relies on a binary significance cutoff, which may lose information compared to GSEA or topology-based methods.")
        
        try:
            # Try using g:Profiler API for enrichment (works for KEGG, Reactome, GO)
            if self.use_gprofiler and db_value in ['kegg', 'reactome', 'go']:
                return await self._analyze_with_gprofiler(
                    gene_list, database, species, significance_threshold,
                    min_pathway_size, max_pathway_size
                )
            
            # Fallback to database manager for other databases
            # Validate input parameters
            if not self._validate_input_parameters(gene_list, database, species):
                raise ValueError("Invalid input parameters")
            
            if not self._validate_threshold_parameters(significance_threshold, min_pathway_size, max_pathway_size):
                raise ValueError("Invalid threshold parameters")

            # Get pathway data from database
            pathway_data_dict = await self.database_manager.get_pathways(
                databases=[db_value],
                species=species
            )
            
            # Extract pathways for the specific database
            pathways = pathway_data_dict.get(db_value, [])
            
            if not pathways:
                self.logger.warning(f"No pathways found for {db_value} in {species}")
                return self._create_empty_result(database, species)
            
            # Determine background size
            if background_size is not None:
                final_background_size = background_size
            elif background_genes is not None:
                final_background_size = len(background_genes)
            else:
                # Fallback to total unique genes in the database for this species
                # This is a better approximation than a hardcoded number
                all_db_genes = set()
                for p in pathways:
                    if hasattr(p, 'gene_ids'):
                        all_db_genes.update(p.gene_ids)
                    elif isinstance(p, dict):
                        all_db_genes.update(p.get('genes', []))
                
                final_background_size = len(all_db_genes)
                if final_background_size == 0:
                    # Last resort fallback if database is empty or malformed
                    self.logger.warning("Could not determine background size from database, using default 20000")
                    final_background_size = 20000
            
            # Perform ORA analysis
            pathway_results = []
            total_genes = len(gene_list)
            
            for pathway in pathways:
                # Extract pathway genes
                if hasattr(pathway, 'gene_ids'):
                    pathway_genes = pathway.gene_ids
                elif hasattr(pathway, 'genes'):
                    pathway_genes = pathway.genes
                elif isinstance(pathway, dict):
                    pathway_genes = pathway.get('gene_ids', pathway.get('genes', []))
                else:
                    pathway_genes = []
                
                pathway_name = getattr(pathway, 'name', None) or (pathway.get('name') if isinstance(pathway, dict) else 'Unknown')
                pathway_id = getattr(pathway, 'pathway_id', None) or (pathway.get('pathway_id') if isinstance(pathway, dict) else pathway.get('id', 'unknown'))
                
                # Calculate overlap
                overlapping_genes = list(set(gene_list) & set(pathway_genes))
                overlap_count = len(overlapping_genes)
                pathway_count = len(pathway_genes)
                
                if overlap_count == 0:
                    continue
                
                # Filter by pathway size
                if not (min_pathway_size <= pathway_count <= max_pathway_size):
                    continue
                
                # Calculate p-value using hypergeometric test
                p_value = self._calculate_hypergeometric_pvalue(
                    overlap_count, total_genes, pathway_count, 
                    final_background_size
                )
                
                # Calculate enrichment score
                enrichment_score = self._calculate_enrichment_score(
                    overlap_count, total_genes, pathway_count, final_background_size
                )
                
                # Calculate research-grade enrichment statistics
                enrichment_stats = calculate_enrichment_statistics(
                    overlap_count=overlap_count,
                    pathway_count=pathway_count,
                    input_count=total_genes,
                    background_size=final_background_size
                )
                
                # Calculate statistical power
                statistical_power = calculate_statistical_power(
                    alpha=significance_threshold,
                    overlap_observed=overlap_count,
                    pathway_count=pathway_count,
                    input_count=total_genes,
                    background_size=final_background_size
                )
                
                # Extract metadata
                if hasattr(pathway, 'url'):
                    p_url = getattr(pathway, 'url', None)
                    p_desc = getattr(pathway, 'description', None)
                    p_cat = getattr(pathway, 'category', None)
                elif isinstance(pathway, dict):
                    p_url = pathway.get('url')
                    p_desc = pathway.get('description')
                    p_cat = pathway.get('category')
                else:
                    p_url = None
                    p_desc = None
                    p_cat = None

                pathway_result = PathwayResult(
                    pathway_id=pathway_id,
                    pathway_name=pathway_name,
                    database=database,
                    p_value=p_value,
                    adjusted_p_value=p_value,  # Will be corrected later
                    enrichment_score=enrichment_score,
                    # Research-grade statistics
                    odds_ratio=enrichment_stats.odds_ratio,
                    odds_ratio_ci_lower=enrichment_stats.odds_ratio_ci_lower,
                    odds_ratio_ci_upper=enrichment_stats.odds_ratio_ci_upper,
                    fold_enrichment=enrichment_stats.fold_enrichment,
                    effect_size=enrichment_stats.effect_size,
                    genes_expected=enrichment_stats.genes_expected,
                    statistical_power=statistical_power,
                    # Gene counts
                    overlap_count=overlap_count,
                    pathway_count=pathway_count,
                    input_count=total_genes,
                    overlapping_genes=overlapping_genes,
                    pathway_genes=pathway_genes,
                    pathway_url=p_url,
                    pathway_description=p_desc,
                    pathway_category=p_cat,
                    analysis_method="ORA"
                )
                
                pathway_results.append(pathway_result)
            
            # Apply multiple testing correction
            if pathway_results:
                p_values = [result.p_value for result in pathway_results]
                corrected_p_values = self._apply_correction(p_values, correction_method)
                
                for i, result in enumerate(pathway_results):
                    result.adjusted_p_value = corrected_p_values[i]
            
            # Filter significant pathways
            significant_pathways = [
                result for result in pathway_results 
                if result.adjusted_p_value <= significance_threshold
            ]
            
            # Calculate coverage
            covered_genes = set()
            for result in pathway_results:
                covered_genes.update(result.overlapping_genes)
            coverage = len(covered_genes) / total_genes if total_genes > 0 else 0.0
            
            # Sort by significance
            pathway_results.sort(key=lambda x: x.adjusted_p_value)
            significant_pathways.sort(key=lambda x: x.adjusted_p_value)
            
            self.logger.info(
                f"ORA analysis completed: {len(significant_pathways)}/{len(pathway_results)} "
                f"significant pathways found"
            )
            
            return DatabaseResult(
                database=database,
                pathways=pathway_results,
                total_pathways=len(pathway_results),
                significant_pathways=len([p for p in pathway_results if p.adjusted_p_value < significance_threshold]),
                species=species,
                timestamp=datetime.now().isoformat(),
                coverage=coverage
            )
            
        except Exception as e:
            self.logger.error(f"ORA analysis failed: {e}")
            raise
    
    async def _analyze_with_gprofiler(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str,
        significance_threshold: float,
        min_pathway_size: int,
        max_pathway_size: int
    ) -> DatabaseResult:
        """Perform ORA using g:Profiler API."""
        from ..integration.gprofiler import GProfilerClient
        
        self.logger.info(f"Using g:Profiler API for {database.value} enrichment")
        
        # Map species to g:Profiler codes
        species_map = {
            'homo_sapiens': 'hsapiens',
            'mus_musculus': 'mmusculus',
            'rattus_norvegicus': 'rnorvegicus',
            'human': 'hsapiens',
            'mouse': 'mmusculus',
            'rat': 'rnorvegicus'
        }
        
        gprofiler_species = species_map.get(species.lower(), 'hsapiens')
        
        # Map database to g:Profiler sources
        source_map = {
            'kegg': ['KEGG'],
            'reactome': ['REAC'],
            'go': ['GO:BP', 'GO:MF', 'GO:CC']
        }
        
        sources = source_map.get(database.value, ['KEGG', 'REAC'])
        
        try:
            client = GProfilerClient()
            results = await client.enrich(
                gene_list=gene_list,
                species=gprofiler_species,
                sources=sources,
                significance_threshold=significance_threshold
            )
            
            # Convert g:Profiler results to PathwayResult objects
            pathway_results = []
            for item in results.get('result', []):
                # Filter by pathway size
                term_size = item.get('term_size', 0)
                if term_size < min_pathway_size or term_size > max_pathway_size:
                    continue
                
                intersection_size = item.get('intersection_size', 0)
                query_size = item.get('query_size', len(gene_list))
                
                pathway_results.append(PathwayResult(
                    pathway_id=item.get('pathway_id', ''),
                    pathway_name=item.get('name', ''),
                    database=database,  # Required field
                    p_value=item.get('p_value', 1.0),
                    adjusted_p_value=item.get('p_value', 1.0),  # g:Profiler already applies correction
                    fold_enrichment=item.get('precision', 0.0) * 100 if item.get('precision') else 1.0,
                    # Required fields
                    overlap_count=intersection_size,
                    pathway_count=term_size,
                    input_count=query_size,
                    overlapping_genes=[],  # g:Profiler doesn't return gene lists in basic response
                    analysis_method='ORA',
                    # Optional fields
                    pathway_description=item.get('description', ''),
                    pathway_url=None
                ))
            
            # Calculate coverage as proportion of input genes found in at least one pathway
            genes_with_pathways = set()
            for pathway in pathway_results:
                # g:Profiler's basic response doesn't include overlapping genes,
                # so this coverage calculation will be 0 unless a more detailed
                # g:Profiler call is made or the PathwayResult is populated differently.
                # For now, it will reflect the current state of `overlapping_genes` (empty).
                genes_with_pathways.update(pathway.overlapping_genes)
            coverage = len(genes_with_pathways) / max(len(gene_list), 1) if gene_list else 0.0
            
            self.logger.info(f"g:Profiler returned {len(pathway_results)} significant pathways")
            
            return DatabaseResult(
                database=database,
                pathways=pathway_results,
                total_pathways=len(pathway_results),
                significant_pathways=len([p for p in pathway_results if p.adjusted_p_value < significance_threshold]),
                species=species,
                timestamp=datetime.now().isoformat(),
                coverage=min(coverage, 1.0),  # Ensure it's between 0 and 1
                metadata={'api_source': 'gprofiler'}
            )
            
        except Exception as e:
            self.logger.error(f"g:Profiler API error: {e}")
            # Return empty result instead of failing
            return self._create_empty_result(database, species)
    
    def _validate_input_parameters(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str
    ) -> bool:
        """Validate input parameters."""
        if not gene_list:
            return False
        if not database:
            return False
        if not species:
            return False
        return True
    
    def _create_empty_result(
        self,
        database: DatabaseType,
        species: str
    ) -> DatabaseResult:
        """Create an empty database result for when no pathways are found."""
        return DatabaseResult(
            database=database,
            total_pathways=0,
            significant_pathways=0,
            pathways=[],
            species=species,
            coverage=0.0
        )

    def _validate_threshold_parameters(
        self,
        significance_threshold: float,
        min_pathway_size: int,
        max_pathway_size: int
    ) -> bool:
        """Validate threshold parameters."""
        if not (0 <= significance_threshold <= 1):
            return False
        if min_pathway_size > max_pathway_size:
            return False
        if min_pathway_size < 0 or max_pathway_size < 0:
            return False
        return True

    def _filter_pathways_by_size(
        self,
        pathways: List[Any],
        min_size: int,
        max_size: int
    ) -> List[Any]:
        """Filter pathways by size."""
        filtered_pathways = []
        for pathway in pathways:
            # Handle PathwayInfo objects
            if hasattr(pathway, 'genes'):
                size = len(pathway.genes)
            # Handle dicts (legacy)
            elif isinstance(pathway, dict):
                size = len(pathway.get('genes', []))
                if 'size' in pathway:
                    size = pathway['size']
            else:
                continue
            
            if min_size <= size <= max_size:
                filtered_pathways.append(pathway)
        return filtered_pathways

    def _calculate_hypergeometric_pvalue(
        self, 
        overlap_count: int, 
        total_genes: int, 
        pathway_count: int, 
        background_size: int
    ) -> float:
        """Calculate hypergeometric p-value."""
        # Hypergeometric test: P(X >= overlap_count)
        # X ~ Hypergeometric(N=background_size, K=pathway_count, n=total_genes)
        
        # Use survival function (1 - CDF) for P(X >= overlap_count)
        p_value = stats.hypergeom.sf(
            overlap_count - 1,  # -1 because sf gives P(X > k), we want P(X >= k)
            background_size,
            pathway_count,
            total_genes
        )
        
        return min(max(p_value, 1e-300), 1.0)  # Clamp to avoid numerical issues
    
    def _calculate_enrichment_score(
        self, 
        overlap_count: int, 
        total_genes: int, 
        pathway_count: int,
        background_size: int
    ) -> float:
        """Calculate enrichment score (fold enrichment)."""
        if total_genes == 0 or pathway_count == 0:
            return 0.0
        
        expected_overlap = (total_genes * pathway_count) / background_size
        if expected_overlap == 0:
            return float('inf') if overlap_count > 0 else 0.0
        
        return overlap_count / expected_overlap

    def _calculate_odds_ratio(
        self,
        n_successes: int,
        n_draws: int,
        n_total: int,
        n_successes_total: int
    ) -> float:
        """
        Calculate Odds Ratio.
        
        a = n_successes (overlap)
        b = n_draws - n_successes (genes in list not in pathway)
        c = n_successes_total - n_successes (genes in pathway not in list)
        d = n_total - n_draws - n_successes_total + n_successes (genes not in list and not in pathway)
        """
        a = n_successes
        b = n_draws - n_successes
        c = n_successes_total - n_successes
        d = n_total - n_draws - n_successes_total + n_successes
        
        if b * c == 0:
            return float('inf')
        return (a * d) / (b * c)

    def _calculate_confidence_interval(
        self,
        n_successes: int,
        n_draws: int,
        n_total: int,
        n_successes_total: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for Odds Ratio."""
        a = n_successes
        b = n_draws - n_successes
        c = n_successes_total - n_successes
        d = n_total - n_draws - n_successes_total + n_successes
        
        if a == 0 or b == 0 or c == 0 or d == 0:
            return (0.0, float('inf'))
            
        log_or = np.log((a * d) / (b * c))
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        lower = np.exp(log_or - z * se)
        upper = np.exp(log_or + z * se)
        
        return (lower, upper)

    def _apply_correction(
        self, 
        p_values: List[float], 
        correction_method: CorrectionMethod
    ) -> List[float]:
        """Apply multiple testing correction."""
        if not p_values:
            return []
        
        # Convert to numpy array
        p_array = np.array(p_values)
        
        # Apply correction based on method
        if correction_method == CorrectionMethod.BONFERRONI:
            corrected = multipletests(p_array, method='bonferroni')[1]
        elif correction_method == CorrectionMethod.FDR_BH:
            corrected = multipletests(p_array, method='fdr_bh')[1]
        elif correction_method == CorrectionMethod.FDR_BY:
            corrected = multipletests(p_array, method='fdr_by')[1]
        elif correction_method == CorrectionMethod.FDR_TSBH:
            corrected = multipletests(p_array, method='fdr_tsbh')[1]
        elif correction_method == CorrectionMethod.FDR_TSBKY:
            corrected = multipletests(p_array, method='fdr_tsbky')[1]
        elif correction_method == CorrectionMethod.HOLM:
            corrected = multipletests(p_array, method='holm')[1]
        elif correction_method == CorrectionMethod.HOCHBERG:
            corrected = multipletests(p_array, method='hochberg')[1]
        elif correction_method == CorrectionMethod.HOMMEL:
            corrected = multipletests(p_array, method='hommel')[1]
        elif correction_method == CorrectionMethod.SIDAK:
            corrected = multipletests(p_array, method='sidak')[1]
        elif correction_method == CorrectionMethod.SIDAK_SS:
            corrected = multipletests(p_array, method='sidak_ss')[1]
        elif correction_method == CorrectionMethod.SIDAK_SD:
            corrected = multipletests(p_array, method='sidak_sd')[1]
        else:
            # Default to FDR_BH
            corrected = multipletests(p_array, method='fdr_bh')[1]
        
        return corrected.tolist()
    
    def _calculate_pathway_statistics(
        self, 
        pathway_id: str,
        pathway_info: Dict[str, Any],
        gene_list: List[str],
        background_genes: set
    ) -> Dict[str, Any]:
        """Calculate detailed statistics for a single pathway."""
        pathway_genes = set(pathway_info.get('genes', []))
        input_genes = set(gene_list)
        
        overlap = input_genes.intersection(pathway_genes)
        
        stats = {
            'pathway_id': pathway_id,
            'pathway_name': pathway_info.get('name', 'Unknown'),
            'pathway_size': len(pathway_genes),
            'gene_overlap': list(overlap),
            'gene_overlap_count': len(overlap),
            'p_value': 1.0
        }
        
        if len(overlap) > 0:
            p_value = self._calculate_hypergeometric_pvalue(
                len(overlap),
                len(input_genes),
                len(pathway_genes),
                len(background_genes)
            )
            stats['p_value'] = p_value
            
        return stats

    def calculate_pathway_statistics(
        self, 
        pathway_results: List[PathwayResult]
    ) -> Dict[str, Any]:
        """Calculate additional pathway statistics."""
        if not pathway_results:
            return {}
        
        p_values = [result.p_value for result in pathway_results]
        adjusted_p_values = [result.adjusted_p_value for result in pathway_results]
        enrichment_scores = [result.enrichment_score for result in pathway_results if result.enrichment_score is not None]
        
        stats_dict = {
            'num_pathways': len(pathway_results),
            'min_p_value': min(p_values) if p_values else 1.0,
            'max_p_value': max(p_values) if p_values else 0.0,
            'mean_p_value': np.mean(p_values) if p_values else 1.0,
            'median_p_value': np.median(p_values) if p_values else 1.0,
            'min_adjusted_p_value': min(adjusted_p_values) if adjusted_p_values else 1.0,
            'max_adjusted_p_value': max(adjusted_p_values) if adjusted_p_values else 0.0,
            'mean_adjusted_p_value': np.mean(adjusted_p_values) if adjusted_p_values else 1.0,
            'median_adjusted_p_value': np.median(adjusted_p_values) if adjusted_p_values else 1.0,
        }
        
        if enrichment_scores:
            stats_dict.update({
                'min_enrichment_score': min(enrichment_scores),
                'max_enrichment_score': max(enrichment_scores),
                'mean_enrichment_score': np.mean(enrichment_scores),
                'median_enrichment_score': np.median(enrichment_scores),
            })
        
        return stats_dict
