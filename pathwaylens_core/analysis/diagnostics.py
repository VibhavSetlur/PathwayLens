"""
Statistical diagnostics for pathway enrichment analysis.

This module provides quality control and diagnostic functions for
pathway analysis results, including:
- P-value distribution analysis
- Pathway size bias detection  
- Coverage metrics calculation
- Diagnostic report generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .schemas import DatabaseResult, PathwayResult
from .statistical_utils import check_pvalue_distribution, detect_pathway_size_bias


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report for pathway analysis."""
    
    # P-value distribution diagnostics
    pvalue_distribution: Dict[str, Any]
    
    # Pathway size bias diagnostics
    size_bias: Dict[str, Any]
    
    # Coverage metrics
    coverage_metrics: Dict[str, Any]
    
    # Quality flags
    quality_flags: Dict[str, bool]
    
    # Overall quality score (0-1)
    overall_quality_score: float


class StatisticalDiagnostics:
    """Statistical diagnostics for pathway enrichment results."""
    
    def __init__(self):
        """Initialize diagnostics analyzer."""
        pass
    
    def analyze_pvalue_distribution(
        self,
        results: DatabaseResult
    ) -> Dict[str, Any]:
        """
        Analyze p-value distribution for quality control.
        
        Checks if p-values follow expected patterns and detects
        potential methodological issues.
        
        Args:
            results: DatabaseResult containing pathway results
            
        Returns:
            Dictionary with diagnostic statistics
        """
        pvalues = [p.p_value for p in results.pathways]
        return check_pvalue_distribution(pvalues)
    
    def detect_size_bias(
        self,
        results: DatabaseResult
    ) -> Dict[str, Any]:
        """
        Detect correlation between pathway size and significance.
        
        Args:
            results: DatabaseResult containing pathway results
            
        Returns:
            Dictionary with bias detection results
        """
        pathway_sizes = [p.pathway_count for p in results.pathways]
        pvalues = [p.p_value for p in results.pathways]
        
        return detect_pathway_size_bias(pathway_sizes, pvalues)
    
    def calculate_coverage_metrics(
        self,
        results: DatabaseResult,
        gene_universe: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate gene coverage metrics.
        
        Args:
            results: DatabaseResult containing pathway results
            gene_universe: Optional background gene universe
            
        Returns:
            Dictionary with coverage metrics
        """
        # Get all unique genes covered by pathways
        all_pathway_genes = set()
        for pathway in results.pathways:
            all_pathway_genes.update(pathway.pathway_genes)
        
        # Get genes in input
        input_genes = set()
        if results.pathways:
            input_count = results.pathways[0].input_count
            # Collect overlapping genes from all pathways
            for pathway in results.pathways:
                input_genes.update(pathway.overlapping_genes)
        else:
            input_count = 0
        
        # Calculate metrics
        pathway_coverage = len(all_pathway_genes)
        input_coverage = len(input_genes)
        
        metrics = {
            "total_pathway_genes": pathway_coverage,
            "total_input_genes_covered": input_coverage,
            "input_coverage_rate": input_coverage / input_count if input_count > 0 else 0.0,
            "num_pathways": len(results.pathways),
            "avg_pathway_size": np.mean([p.pathway_count for p in results.pathways]) if results.pathways else 0,
            "median_pathway_size": np.median([p.pathway_count for p in results.pathways]) if results.pathways else 0
        }
        
        if gene_universe:
            metrics["universe_coverage_rate"] = len(all_pathway_genes) / len(gene_universe)
        
        return metrics
    
    def generate_diagnostic_report(
        self,
        results: DatabaseResult,
        gene_universe: Optional[List[str]] = None
    ) -> DiagnosticReport:
        """
        Generate comprehensive diagnostic report.
        
        Args:
            results: DatabaseResult containing pathway results
            gene_universe: Optional background gene universe
            
        Returns:
            DiagnosticReport with all diagnostic information
        """
        # Run all diagnostics
        pvalue_dist = self.analyze_pvalue_distribution(results)
        size_bias = self.detect_size_bias(results)
        coverage = self.calculate_coverage_metrics(results, gene_universe)
        
        # Determine quality flags
        quality_flags = {
            "pvalue_distribution_ok": pvalue_dist["uniform_ks_pvalue"] > 0.01,
            "no_size_bias": not size_bias["bias_detected"],
            "sufficient_coverage": coverage["input_coverage_rate"] > 0.5,
            "sufficient_pathways": len(results.pathways) >= 10
        }
        
        # Calculate overall quality score
        quality_score = sum(quality_flags.values()) / len(quality_flags)
        
        return DiagnosticReport(
            pvalue_distribution=pvalue_dist,
            size_bias=size_bias,
            coverage_metrics=coverage,
            quality_flags=quality_flags,
            overall_quality_score=quality_score
        )
    
    def create_diagnostic_summary(
        self,
        report: DiagnosticReport
    ) -> str:
        """
        Create human-readable diagnostic summary.
        
        Args:
            report: DiagnosticReport to summarize
            
        Returns:
            Formatted diagnostic summary string
        """
        summary = []
        summary.append("=== Pathway Analysis Diagnostic Report ===\n")
        
        summary.append(f"Overall Quality Score: {report.overall_quality_score:.2f}/1.00\n")
        
        summary.append("\n P-Value Distribution:")
        pv = report.pvalue_distribution
        summary.append(f"  - Number of p-values: {pv['num_pvalues']}")
        summary.append(f"  - Median p-value: {pv['median_pvalue']:.4f}")
        summary.append(f"  - % Significant (p<0.05): {pv['pct_significant_0.05']:.1f}%")
        summary.append(f"  - Uniformity test p-value: {pv['uniform_ks_pvalue']:.4f}")
        summary.append(f"  - Estimated π₀ (true nulls): {pv['pi0_estimate']:.2f}")
        
        summary.append("\n✓ Pathway Size Bias:")
        sb = report.size_bias
        summary.append(f"  - Spearman correlation: {sb['spearman_correlation']:.3f}")
        summary.append(f"  - Correlation p-value: {sb['spearman_pvalue']:.4f}")
        summary.append(f"  - Bias detected: {'YES' if sb['bias_detected'] else 'NO'}")
        
        summary.append("\n✓ Coverage Metrics:")
        cm = report.coverage_metrics
        summary.append(f"  - Input coverage rate: {cm['input_coverage_rate']:.1%}")
        summary.append(f"  - Average pathway size: {cm['avg_pathway_size']:.1f} genes")
        summary.append(f"  - Total pathways: {cm['num_pathways']}")
        
        summary.append("\n✓ Quality Flags:")
        for flag, status in report.quality_flags.items():
            symbol = "✓" if status else "✗"
            summary.append(f"  {symbol} {flag}: {'PASS' if status else 'FAIL'}")
        
        return "\n".join(summary)
