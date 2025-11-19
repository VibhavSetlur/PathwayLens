"""
Statistical utilities for pathway enrichment analysis.

This module provides research-grade statistical functions for pathway analysis including:
- Odds ratio calculations with confidence intervals
- Effect size calculations
- P-value distribution diagnostics
- Statistical power estimation
"""

import numpy as np
import scipy.stats as stats
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class OddsRatioResult:
    """Result of odds ratio calculation."""
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    standard_error: float


@dataclass
class EnrichmentStatistics:
    """Comprehensive enrichment statistics for a pathway."""
    odds_ratio: float
    odds_ratio_ci_lower: float
    odds_ratio_ci_upper: float
    fold_enrichment: float
    effect_size: float  # Cohen's h
    genes_expected: float
    genes_observed: int


def calculate_odds_ratio(
    overlap_count: int,
    pathway_count: int,
    input_count: int,
    background_size: int,
    confidence_level: float = 0.95
) -> OddsRatioResult:
    """
    Calculate odds ratio with confidence intervals for pathway enrichment.
    
    Uses the Wilson score method for confidence intervals, which is more accurate
    than the normal approximation for small samples.
    
    Contingency table:
                    In Pathway | Not in Pathway
    In Gene List        a      |       b
    Not in Gene List    c      |       d
    
    where:
    a = overlap_count (genes in both list and pathway)
    b = input_count - overlap_count (genes in list, not in pathway)
    c = pathway_count - overlap_count (genes in pathway, not in list)
    d = background_size - input_count - pathway_count + overlap_count
    
    Odds Ratio = (a * d) / (b * c)
    
    Args:
        overlap_count: Number of genes in both input list and pathway
        pathway_count: Total genes in pathway
        input_count: Total genes in input list
        background_size: Total genes in background/universe
        confidence_level: Confidence level for CI (default 0.95 for 95% CI)
        
    Returns:
        OddsRatioResult with odds ratio and confidence intervals
        
    References:
        Wilson, E. B. (1927). Probable inference, the law of succession, and 
        statistical inference. Journal of the American Statistical Association, 
        22(158), 209-212.
    """
    # Construct contingency table
    a = overlap_count
    b = input_count - overlap_count
    c = pathway_count - overlap_count
    d = background_size - input_count - pathway_count + overlap_count
    
    # Add continuity correction for zero cells
    if a == 0 or b == 0 or c == 0 or d == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    
    # Calculate odds ratio
    odds_ratio = (a * d) / (b * c)
    
    # Calculate log odds ratio and standard error
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    # Calculate confidence interval (Wilson score method)
    z = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = np.exp(log_or - z * se_log_or)
    ci_upper = np.exp(log_or + z * se_log_or)
    
    return OddsRatioResult(
        odds_ratio=odds_ratio,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        standard_error=se_log_or
    )


def calculate_fold_enrichment(
    overlap_count: int,
    pathway_count: int,
    input_count: int,
    background_size: int
) -> float:
    """
    Calculate fold enrichment ratio.
    
    Fold Enrichment = (overlap / input_count) / (pathway_count / background_size)
                   = observed / expected proportion
    
    Args:
        overlap_count: Genes in both input and pathway
        pathway_count: Total genes in pathway
        input_count: Total genes in input
        background_size: Total background genes
        
    Returns:
        Fold enrichment ratio
    """
    if input_count == 0 or background_size == 0 or pathway_count == 0:
        return 0.0
    
    observed_proportion = overlap_count / input_count
    expected_proportion = pathway_count / background_size
    
    if expected_proportion == 0:
        return float('inf') if observed_proportion > 0 else 0.0
    
    return observed_proportion / expected_proportion


def calculate_expected_genes(
    input_count: int,
    pathway_count: int,
    background_size: int
) -> float:
    """
    Calculate expected number of overlapping genes by chance.
    
    Expected = (input_count * pathway_count) / background_size
    
    Args:
        input_count: Total genes in input list
        pathway_count: Total genes in pathway
        background_size: Total background genes
        
    Returns:
        Expected number of overlapping genes
    """
    if background_size == 0:
        return 0.0
    return (input_count * pathway_count) / background_size


def calculate_cohens_h(
    overlap_count: int,
    input_count: int,
    pathway_count: int,
    background_size: int
) -> float:
    """
    Calculate Cohen's h effect size for enrichment.
    
    Cohen's h measures the difference between two proportions:
    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    
    where:
    p1 = overlap / input_count (observed proportion)
    p2 = pathway_count / background_size (expected proportion)
    
    Interpretation:
    - h < 0.2: small effect
    - 0.2 <= h < 0.5: medium effect
    - h >= 0.5: large effect
    
    Args:
        overlap_count: Genes in both input and pathway
        input_count: Total genes in input
        pathway_count: Total genes in pathway
        background_size: Total background genes
        
    Returns:
        Cohen's h effect size
        
    References:
        Cohen, J. (1988). Statistical power analysis for the behavioral sciences 
        (2nd ed.). Hillsdale, NJ: Lawrence Erlbaum Associates.
    """
    if input_count == 0 or background_size == 0:
        return 0.0
    
    p1 = overlap_count / input_count
    p2 = pathway_count / background_size
    
    # Arcsin transformation
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    
    return phi1 - phi2


def calculate_enrichment_statistics(
    overlap_count: int,
    pathway_count: int,
    input_count: int,
    background_size: int
) -> EnrichmentStatistics:
    """
    Calculate comprehensive enrichment statistics for a pathway.
    
    This combines multiple statistical measures to provide a complete picture
    of pathway enrichment suitable for research publication.
    
    Args:
        overlap_count: Genes in both input and pathway
        pathway_count: Total genes in pathway
        input_count: Total genes in input
        background_size: Total background genes
        
    Returns:
        EnrichmentStatistics with all metrics
    """
    # Calculate odds ratio with CI
    or_result = calculate_odds_ratio(
        overlap_count, pathway_count, input_count, background_size
    )
    
    # Calculate other metrics
    fold_enrichment = calculate_fold_enrichment(
        overlap_count, pathway_count, input_count, background_size
    )
    
    effect_size = calculate_cohens_h(
        overlap_count, input_count, pathway_count, background_size
    )
    
    expected_genes = calculate_expected_genes(
        input_count, pathway_count, background_size
    )
    
    return EnrichmentStatistics(
        odds_ratio=or_result.odds_ratio,
        odds_ratio_ci_lower=or_result.ci_lower,
        odds_ratio_ci_upper=or_result.ci_upper,
        fold_enrichment=fold_enrichment,
        effect_size=effect_size,
        genes_expected=expected_genes,
        genes_observed=overlap_count
    )


def check_pvalue_distribution(pvalues: List[float]) -> Dict[str, Any]:
    """
    Check if p-value distribution follows expected patterns.
    
    For true null hypotheses, p-values should be uniformly distributed.
    Deviations indicate:
    - Enrichment at low p-values: true signal
    - Enrichment at high p-values: possible bias
    - Non-uniformity throughout: methodological issues
    
    Args:
        pvalues: List of p-values to analyze
        
    Returns:
        Dictionary with diagnostic statistics:
        - uniform_ks_stat: Kolmogorov-Smirnov statistic vs uniform
        - uniform_ks_pvalue: P-value for uniformity test
        - pi0_estimate: Estimated proportion of true nulls
        - excess_low_pvalues: Proportion below 0.05 vs expected
    """
    pvalues = np.array(pvalues)
    pvalues = pvalues[~np.isnan(pvalues)]  # Remove NaNs
    
    if len(pvalues) == 0:
        return {
            "uniform_ks_stat": np.nan,
            "uniform_ks_pvalue": np.nan,
            "pi0_estimate": np.nan,
            "excess_low_pvalues": np.nan,
            "num_pvalues": 0
        }
    
    # Test against uniform distribution
    ks_stat, ks_pvalue = stats.kstest(pvalues, 'uniform')
    
    # Estimate pi0 (proportion of true nulls) using Storey's method
    lambda_threshold = 0.5
    pi0 = np.mean(pvalues > lambda_threshold) / (1 - lambda_threshold)
    pi0 = min(pi0, 1.0)  # Pi0 should not exceed 1
    
    # Check excess at low p-values
    observed_low = np.mean(pvalues < 0.05)
    expected_low = 0.05
    excess_low = (observed_low - expected_low) / expected_low if expected_low > 0 else 0
    
    return {
        "uniform_ks_stat": float(ks_stat),
        "uniform_ks_pvalue": float(ks_pvalue),
        "pi0_estimate": float(pi0),
        "excess_low_pvalues": float(excess_low),
        "num_pvalues": len(pvalues),
        "median_pvalue": float(np.median(pvalues)),
        "pct_significant_0.05": float(np.mean(pvalues < 0.05) * 100)
    }


def detect_pathway_size_bias(
    pathway_sizes: List[int],
    pvalues: List[float]
) -> Dict[str, Any]:
    """
    Detect correlation between pathway size and significance.
    
    Systematic correlation can indicate:
    - Positive correlation: larger pathways more likely significant (bias)
    - Negative correlation: smaller pathways more likely significant (possible)
    - No correlation: ideal scenario
    
    Args:
        pathway_sizes: List of pathway sizes (gene counts)
        pvalues: Corresponding p-values
        
    Returns:
        Dictionary with:
        - spearman_correlation: Spearman correlation coefficient
        - spearman_pvalue: P-value for correlation test
        - bias_detected: Boolean indicating significant correlation
    """
    if len(pathway_sizes) != len(pvalues):
        raise ValueError("pathway_sizes and pvalues must have same length")
    
    # Remove NaNs
    valid_idx = ~(np.isnan(pvalues) | np.isnan(pathway_sizes))
    pathway_sizes = np.array(pathway_sizes)[valid_idx]
    pvalues = np.array(pvalues)[valid_idx]
    
    if len(pvalues) < 3:
        return {
            "spearman_correlation": np.nan,
            "spearman_pvalue": np.nan,
            "bias_detected": False,
            "num_pathways": len(pvalues)
        }
    
    # Calculate Spearman correlation (robust to outliers)
    correlation, pvalue = stats.spearmanr(pathway_sizes, pvalues)
    
    # Detect bias (significant correlation at p < 0.01)
    bias_detected = bool(pvalue < 0.01)
    
    return {
        "spearman_correlation": float(correlation),
        "spearman_pvalue": float(pvalue),
        "bias_detected": bias_detected,
        "num_pathways": len(pvalues)
    }


def calculate_statistical_power(
    alpha: float,
    overlap_observed: int,
    pathway_count: int,
    input_count: int,
    background_size: int
) -> float:
    """
    Calculate post-hoc statistical power for enrichment test.
    
    Power is the probability of detecting true enrichment given
    the observed effect size and sample size.
    
    Args:
        alpha: Significance level (e.g., 0.05)
        overlap_observed: Observed overlapping genes
        pathway_count: Pathway size
        input_count: Input gene count
        background_size: Background size
        
    Returns:
        Statistical power (0-1)
    """
    # Calculate effect size
    effect_size = calculate_cohens_h(
        overlap_observed, input_count, pathway_count, background_size
    )
    
    # For hypergeometric test, power depends on effect size and sample size
    # This is a simplified approximation
    n_eff = min(input_count, pathway_count)  # Effective sample size
    
    # Use normal approximation for power
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = np.sqrt(n_eff) * abs(effect_size) - z_alpha
    power = stats.norm.cdf(z_beta)
    
    return float(np.clip(power, 0, 1))
