"""
Competitive Over-Representation Analysis (ORA) Engine.

Implements competitive ORA methods that test whether genes in a pathway
are MORE differentially expressed than genes OUTSIDE the pathway,
addressing pathway interdependence issues in standard ORA.

Methods:
- Fisher's Exact Test (Competitive): 2x2 contingency test
- Gene Length Weighted: GOseq-style bias correction
- Logistic Regression: Controls for confounders

References:
- Goeman & Bühlmann (2007). "Analyzing gene expression data..."
- Young et al. (2010). GOseq gene length bias correction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from scipy.special import expit
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime

from .schemas import (
    DatabaseType, CorrectionMethod,
    AnalysisParameters, PathwayResult, DatabaseResult
)
from ..data.database_manager import DatabaseManager
from .statistical_utils import calculate_enrichment_statistics


@dataclass
class GeneLengthData:
    """Gene length and bias information."""
    gene_id: str
    length: int
    gc_content: Optional[float] = None
    annotation_count: int = 0


@dataclass 
class CompetitiveORAResult:
    """Result from competitive ORA analysis."""
    pathway_id: str
    pathway_name: str
    p_value_competitive: float
    p_value_standard: float
    odds_ratio: float
    gene_length_bias_score: float
    annotation_bias_score: float
    overlapping_genes: List[str]
    pathway_size: int
    input_size: int


class CompetitiveORAEngine:
    """
    Competitive ORA Engine for pathway interdependence analysis.
    
    Unlike standard ORA which tests if a pathway is enriched in the
    gene list, competitive ORA tests if genes in the pathway are
    MORE differentially expressed than genes OUTSIDE the pathway.
    """
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the Competitive ORA engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="competitive_ora")
        self.database_manager = database_manager
        self._gene_lengths: Dict[str, int] = {}
        self._annotation_counts: Dict[str, int] = {}
    
    async def analyze(
        self,
        gene_list: List[str],
        gene_statistics: Optional[Dict[str, Dict[str, float]]] = None,
        database: DatabaseType = DatabaseType.KEGG,
        species: str = "human",
        significance_threshold: float = 0.05,
        correction_method: CorrectionMethod = CorrectionMethod.FDR_BH,
        min_pathway_size: int = 5,
        max_pathway_size: int = 500,
        background_genes: Optional[List[str]] = None,
        method: str = "fisher",  # "fisher", "logistic", "weighted"
        gene_lengths: Optional[Dict[str, int]] = None
    ) -> DatabaseResult:
        """
        Perform Competitive ORA analysis.
        
        Args:
            gene_list: List of input genes (differentially expressed)
            gene_statistics: Optional dict of gene -> {logFC, pvalue} for ranking
            database: Database to use
            species: Species for analysis
            significance_threshold: Threshold for significance
            correction_method: Multiple testing correction method
            min_pathway_size: Minimum pathway size to include
            max_pathway_size: Maximum pathway size to include
            background_genes: Background gene set
            method: Competitive method ("fisher", "logistic", "weighted")
            gene_lengths: Optional gene length data for bias correction
            
        Returns:
            DatabaseResult with competitive ORA results
        """
        self.logger.info(
            f"Starting competitive ORA with method='{method}' "
            f"for {len(gene_list)} genes on {database.value}"
        )
        
        # Store gene lengths if provided
        if gene_lengths:
            self._gene_lengths = gene_lengths
        
        # Get pathways from database
        pathway_data = await self.database_manager.get_pathways(
            databases=[database.value],
            species=species
        )
        pathways = pathway_data.get(database.value, [])
        
        if not pathways:
            self.logger.warning(f"No pathways found for {database.value}")
            return self._create_empty_result(database, species)
        
        # Determine background
        if background_genes:
            background = set(background_genes)
        else:
            # Use all genes in database as background
            background = set()
            for p in pathways:
                genes = p.gene_ids if hasattr(p, 'gene_ids') else p.get('genes', [])
                background.update(genes)
        
        gene_set = set(gene_list)
        
        # Calculate competitive p-values for each pathway
        results = []
        for pathway in pathways:
            p_id = pathway.pathway_id if hasattr(pathway, 'pathway_id') else pathway.get('id')
            p_name = pathway.name if hasattr(pathway, 'name') else pathway.get('name')
            p_genes = set(pathway.gene_ids if hasattr(pathway, 'gene_ids') else pathway.get('genes', []))
            
            # Filter by size
            if not (min_pathway_size <= len(p_genes) <= max_pathway_size):
                continue
            
            # Calculate overlap
            overlap = gene_set.intersection(p_genes)
            if len(overlap) == 0:
                continue
            
            # Perform competitive test based on method
            if method == "fisher":
                p_value = self._competitive_fisher_test(
                    gene_set, p_genes, background
                )
            elif method == "logistic":
                p_value = self._logistic_regression_test(
                    gene_set, p_genes, background, gene_statistics
                )
            elif method == "weighted":
                p_value = self._gene_length_weighted_test(
                    gene_set, p_genes, background
                )
            else:
                raise ValueError(f"Unknown competitive method: {method}")
            
            # Calculate standard ORA p-value for comparison
            std_pvalue = self._standard_hypergeometric_pvalue(
                len(overlap), len(gene_set), len(p_genes), len(background)
            )
            
            # Calculate bias scores
            annotation_bias = self._calculate_annotation_bias(p_genes)
            length_bias = self._calculate_length_bias(p_genes) if self._gene_lengths else 0.0
            
            # Calculate odds ratio
            odds = self._calculate_odds_ratio(
                len(overlap), len(gene_set), len(background), len(p_genes)
            )
            
            results.append(CompetitiveORAResult(
                pathway_id=p_id,
                pathway_name=p_name,
                p_value_competitive=p_value,
                p_value_standard=std_pvalue,
                odds_ratio=odds,
                gene_length_bias_score=length_bias,
                annotation_bias_score=annotation_bias,
                overlapping_genes=list(overlap),
                pathway_size=len(p_genes),
                input_size=len(gene_set)
            ))
        
        if not results:
            return self._create_empty_result(database, species)
        
        # Apply multiple testing correction
        p_values = [r.p_value_competitive for r in results]
        adjusted_pvalues = self._apply_correction(p_values, correction_method)
        
        # Convert to PathwayResult objects
        pathway_results = []
        for i, r in enumerate(results):
            pathway_results.append(PathwayResult(
                pathway_id=r.pathway_id,
                pathway_name=r.pathway_name,
                p_value=r.p_value_competitive,
                adjusted_p_value=adjusted_pvalues[i],
                enrichment_score=r.odds_ratio,
                overlap_count=len(r.overlapping_genes),
                pathway_count=r.pathway_size,
                input_count=r.input_size,
                overlapping_genes=r.overlapping_genes,
                pathway_genes=r.overlapping_genes,  # Simplified
                odds_ratio=r.odds_ratio,
                confidence_interval=(0.0, 0.0),  # Placeholder
                analysis_method=f"competitive_ora_{method}",
                source=database.value,
                # Store additional competitive info in metadata-like field
                effect_size=r.odds_ratio,
                confidence_score=1.0 - r.annotation_bias_score
            ))
        
        # Sort by adjusted p-value
        pathway_results.sort(key=lambda x: x.adjusted_p_value)
        
        # Count significant
        significant_count = sum(
            1 for p in pathway_results 
            if p.adjusted_p_value <= significance_threshold
        )
        
        return DatabaseResult(
            database=database,
            total_pathways=len(pathways),
            significant_pathways=significant_count,
            pathways=pathway_results,
            species=species,
            coverage=len(gene_set) / len(background) if background else 0.0
        )
    
    def _competitive_fisher_test(
        self,
        de_genes: set,
        pathway_genes: set,
        background: set
    ) -> float:
        """
        Competitive Fisher's exact test.
        
        Tests: Are DE genes more likely to be in the pathway than non-DE genes?
        
        Contingency table:
                      | In Pathway | Not in Pathway |
        DE genes      |     a      |       b        |
        Non-DE genes  |     c      |       d        |
        """
        # Ensure we're working within background
        de_in_bg = de_genes.intersection(background)
        pw_in_bg = pathway_genes.intersection(background)
        non_de = background - de_in_bg
        
        a = len(de_in_bg.intersection(pw_in_bg))  # DE and in pathway
        b = len(de_in_bg - pw_in_bg)              # DE and not in pathway
        c = len(non_de.intersection(pw_in_bg))    # Non-DE and in pathway
        d = len(non_de - pw_in_bg)                # Non-DE and not in pathway
        
        # Fisher's exact test (one-sided, greater)
        _, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        
        return p_value
    
    def _logistic_regression_test(
        self,
        de_genes: set,
        pathway_genes: set,
        background: set,
        gene_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> float:
        """
        Logistic regression test with covariates.
        
        Models P(DE | in_pathway, gene_length, annotation_count)
        """
        if not gene_stats:
            # Fall back to Fisher's test if no gene statistics
            return self._competitive_fisher_test(de_genes, pathway_genes, background)
        
        # Build design matrix
        genes = list(background)
        n = len(genes)
        
        # Response: is gene DE?
        y = np.array([1 if g in de_genes else 0 for g in genes])
        
        # Predictor: is gene in pathway?
        in_pathway = np.array([1 if g in pathway_genes else 0 for g in genes])
        
        # Covariate: gene length (log-transformed)
        if self._gene_lengths:
            lengths = np.array([
                np.log(self._gene_lengths.get(g, 1000) + 1) for g in genes
            ])
        else:
            lengths = np.zeros(n)
        
        # Simple logistic regression using scipy
        # X = [intercept, in_pathway, log_length]
        X = np.column_stack([np.ones(n), in_pathway, lengths])
        
        try:
            # Use iteratively reweighted least squares (simplified)
            # For production, use statsmodels or sklearn
            from scipy.optimize import minimize
            
            def neg_log_likelihood(beta):
                z = X @ beta
                p = expit(z)
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            
            result = minimize(neg_log_likelihood, x0=np.zeros(3), method='BFGS')
            beta = result.x
            
            # Wald test for pathway coefficient
            # Approximate standard error from Hessian
            z = X @ beta
            p = expit(z)
            W = np.diag(p * (1 - p))
            
            try:
                cov = np.linalg.inv(X.T @ W @ X)
                se = np.sqrt(np.diag(cov))
                z_stat = beta[1] / se[1]  # Pathway coefficient
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                # One-sided for enrichment
                if beta[1] > 0:
                    p_value = p_value / 2
                else:
                    p_value = 1 - p_value / 2
            except np.linalg.LinAlgError:
                p_value = 1.0
                
        except Exception as e:
            self.logger.warning(f"Logistic regression failed: {e}, falling back to Fisher")
            p_value = self._competitive_fisher_test(de_genes, pathway_genes, background)
        
        return p_value
    
    def _gene_length_weighted_test(
        self,
        de_genes: set,
        pathway_genes: set,
        background: set
    ) -> float:
        """
        GOseq-style gene length weighted test.
        
        Adjusts for the bias that longer genes are more likely to be DE.
        """
        if not self._gene_lengths:
            self.logger.warning("No gene lengths provided, using unweighted test")
            return self._competitive_fisher_test(de_genes, pathway_genes, background)
        
        # Calculate probability of DE based on gene length using monotonic spline
        genes = list(background)
        de_status = np.array([1 if g in de_genes else 0 for g in genes])
        lengths = np.array([self._gene_lengths.get(g, 1000) for g in genes])
        
        # Fit probability weighting function (simplified: use binned means)
        # Sort by length and bin
        n_bins = min(20, len(genes) // 50)
        if n_bins < 2:
            return self._competitive_fisher_test(de_genes, pathway_genes, background)
        
        sorted_idx = np.argsort(lengths)
        bin_size = len(genes) // n_bins
        
        pwf = {}  # Probability weighting function
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(genes)
            bin_genes = [genes[sorted_idx[j]] for j in range(start, end)]
            bin_de = sum(1 for g in bin_genes if g in de_genes)
            prob = bin_de / len(bin_genes)
            for g in bin_genes:
                pwf[g] = max(prob, 0.001)  # Avoid zero weights
        
        # Calculate weighted test statistic
        # Expected overlap under null, accounting for length bias
        in_pathway = [g for g in genes if g in pathway_genes]
        expected = sum(pwf.get(g, 0.01) for g in in_pathway)
        observed = len(de_genes.intersection(pathway_genes))
        
        # Use Poisson approximation for p-value
        if expected > 0:
            p_value = 1 - stats.poisson.cdf(observed - 1, expected)
        else:
            p_value = 1.0
        
        return p_value
    
    def _standard_hypergeometric_pvalue(
        self,
        overlap: int,
        gene_list_size: int,
        pathway_size: int,
        background_size: int
    ) -> float:
        """Calculate standard hypergeometric p-value for comparison."""
        return stats.hypergeom.sf(
            overlap - 1,
            background_size,
            pathway_size,
            gene_list_size
        )
    
    def _calculate_odds_ratio(
        self,
        overlap: int,
        gene_list_size: int,
        background_size: int,
        pathway_size: int
    ) -> float:
        """Calculate odds ratio."""
        a = overlap
        b = gene_list_size - overlap
        c = pathway_size - overlap
        d = background_size - gene_list_size - pathway_size + overlap
        
        # Add 0.5 for Haldane correction
        if a == 0 or b == 0 or c == 0 or d == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        
        return (a * d) / (b * c) if (b * c) > 0 else float('inf')
    
    def _calculate_annotation_bias(self, pathway_genes: set) -> float:
        """
        Calculate annotation bias score for pathway genes.
        
        Higher score = more heavily annotated genes (potential bias).
        Returns score between 0 and 1.
        """
        if not self._annotation_counts or not pathway_genes:
            return 0.0
        
        counts = [
            self._annotation_counts.get(g, 1) 
            for g in pathway_genes 
            if g in self._annotation_counts
        ]
        
        if not counts:
            return 0.0
        
        # Calculate Gini coefficient as bias measure
        counts = np.sort(counts)
        n = len(counts)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * counts) - (n + 1) * np.sum(counts)) / (n * np.sum(counts))
        
        return gini
    
    def _calculate_length_bias(self, pathway_genes: set) -> float:
        """
        Calculate gene length bias score.
        
        Returns z-score of mean pathway gene length vs background.
        """
        if not self._gene_lengths or not pathway_genes:
            return 0.0
        
        pathway_lengths = [
            self._gene_lengths.get(g, 1000) 
            for g in pathway_genes 
            if g in self._gene_lengths
        ]
        
        all_lengths = list(self._gene_lengths.values())
        
        if not pathway_lengths or not all_lengths:
            return 0.0
        
        mean_pathway = np.mean(pathway_lengths)
        mean_all = np.mean(all_lengths)
        std_all = np.std(all_lengths)
        
        if std_all == 0:
            return 0.0
        
        return (mean_pathway - mean_all) / std_all
    
    def _apply_correction(
        self,
        p_values: List[float],
        method: CorrectionMethod
    ) -> List[float]:
        """Apply multiple testing correction."""
        from statsmodels.stats.multitest import multipletests
        
        method_map = {
            CorrectionMethod.BONFERRONI: 'bonferroni',
            CorrectionMethod.FDR_BH: 'fdr_bh',
            CorrectionMethod.FDR_BY: 'fdr_by',
            CorrectionMethod.HOLM: 'holm',
            CorrectionMethod.HOCHBERG: 'simes-hochberg',
        }
        
        statsmodels_method = method_map.get(method, 'fdr_bh')
        
        try:
            _, adjusted, _, _ = multipletests(p_values, method=statsmodels_method)
            return adjusted.tolist()
        except Exception:
            # Fallback: Benjamini-Hochberg manual
            n = len(p_values)
            ranked = sorted(enumerate(p_values), key=lambda x: x[1])
            adjusted = [0.0] * n
            cummin = 1.0
            for i, (orig_idx, pval) in enumerate(reversed(ranked)):
                rank = n - i
                adj = min(pval * n / rank, cummin)
                adjusted[orig_idx] = adj
                cummin = min(cummin, adj)
            return adjusted
    
    def _create_empty_result(
        self,
        database: DatabaseType,
        species: str
    ) -> DatabaseResult:
        """Create empty result when no pathways found."""
        return DatabaseResult(
            database=database,
            total_pathways=0,
            significant_pathways=0,
            pathways=[],
            species=species,
            coverage=0.0
        )
    
    def set_gene_lengths(self, gene_lengths: Dict[str, int]) -> None:
        """Set gene length data for bias correction."""
        self._gene_lengths = gene_lengths
        self.logger.info(f"Loaded gene lengths for {len(gene_lengths)} genes")
    
    def set_annotation_counts(self, annotation_counts: Dict[str, int]) -> None:
        """Set annotation count data for bias scoring."""
        self._annotation_counts = annotation_counts
        self.logger.info(f"Loaded annotation counts for {len(annotation_counts)} genes")
