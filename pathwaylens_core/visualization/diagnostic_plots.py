"""
Diagnostic visualization plots for pathway analysis quality control.

Provides publication-ready diagnostic plots including:
- P-value distribution histograms
- Q-Q plots for statistical validation
- Pathway size distribution analysis
- Coverage heatmaps
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
from scipy import stats

from ..analysis.schemas import DatabaseResult
from .palettes import ColorPalette, get_plotly_theme


def plot_pvalue_histogram(
    pvalues: List[float],
    title: str = "P-value Distribution",
    show_uniform: bool = True
) -> go.Figure:
    """
    Create p-value distribution histogram.
    
    Under the null hypothesis, p-values should be uniformly distributed.
    Deviations indicate true signal or methodological issues.
    
    Args:
        pvalues: List of p-values
        title: Plot title
        show_uniform: Show expected uniform distribution
        
    Returns:
        Plotly figure
    """
    pvalues = np.array(pvalues)
    pvalues = pvalues[~np.isnan(pvalues)]
    
    fig = go.Figure()
    
    # Histogram of observed p-values
    fig.add_trace(go.Histogram(
        x=pvalues,
        nbinsx=20,
        name="Observed",
        marker_color=ColorPalette.get_colorblind_safe_palette()[1],
        opacity=0.7
    ))
    
    # Expected uniform distribution
    if show_uniform:
        n = len(pvalues)
        expected_count = n / 20  # 20 bins
        fig.add_hline(
            y=expected_count,
            line_dash="dash",
            line_color=ColorPalette.get_colorblind_safe_palette()[0],
            annotation_text="Expected (uniform)",
            annotation_position="top right"
        )
    
    # Apply publication theme
    theme = get_plotly_theme("publication")
    fig.update_layout(
        title=title,
        xaxis_title="P-value",
        yaxis_title="Frequency",
        **theme["layout"]
    )
    
    return fig


def plot_qq_plot(
    pvalues: List[float],
    title: str = "Q-Q Plot"
) -> go.Figure:
    """
    Create quantile-quantile plot for p-values.
    
    Compares observed p-value distribution to expected uniform distribution.
    
    Args:
        pvalues: List of p-values
        title: Plot title
        
    Returns:
        Plotly figure
    """
    pvalues = np.array(pvalues)
    pvalues = pvalues[~np.isnan(pvalues)]
    pvalues = np.sort(pvalues)
    
    n = len(pvalues)
    expected = np.linspace(0, 1, n)
    
    fig = go.Figure()
    
    # Q-Q scatter
    fig.add_trace(go.Scatter(
        x=expected,
        y=pvalues,
        mode='markers',
        name="Observed",
        marker=dict(
            color=ColorPalette.get_colorblind_safe_palette()[2],
            size=4
        )
    ))
    
    # Identity line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name="Expected (uniform)",
        line=dict(
            color=ColorPalette.get_colorblind_safe_palette()[0],
            dash='dash'
        )
    ))
    
    theme = get_plotly_theme("publication")
    fig.update_layout(
        title=title,
        xaxis_title="Expected P-value",
        yaxis_title="Observed P-value",
        **theme["layout"]
    )
    
    return fig


def plot_pathway_size_distribution(
    results: DatabaseResult,
    title: str = "Pathway Size Distribution"
) -> go.Figure:
    """
    Plot distribution of pathway sizes.
    
    Args:
        results: DatabaseResult containing pathway information
        title: Plot title
        
    Returns:
        Plotly figure
    """
    pathway_sizes = [p.pathway_count for p in results.pathways]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=pathway_sizes,
        nbinsx=30,
        marker_color=ColorPalette.get_colorblind_safe_palette()[3],
        opacity=0.7
    ))
    
    # Add median line
    median_size = np.median(pathway_sizes)
    fig.add_vline(
        x=median_size,
        line_dash="dash",
        line_color=ColorPalette.get_colorblind_safe_palette()[0],
        annotation_text=f"Median: {median_size:.0f}",
        annotation_position="top right"
    )
    
    theme = get_plotly_theme("publication")
    fig.update_layout(
        title=title,
        xaxis_title="Pathway Size (genes)",
        yaxis_title="Frequency",
        **theme["layout"]
    )
    
    return fig


def plot_size_vs_significance(
    results: DatabaseResult,
    title: str = "Pathway Size vs Significance"
) -> go.Figure:
    """
    Plot pathway size against significance to detect bias.
    
    Args:
        results: DatabaseResult containing pathway information
        title: Plot title
        
    Returns:
        Plotly figure
    """
    pathway_sizes = [p.pathway_count for p in results.pathways]
    pvalues = [p.p_value for p in results.pathways]
    neg_log_pvalues = [-np.log10(p) if p > 0 else 10 for p in pvalues]
    
    # Color by significance
    colors = ['Significant' if p.adjusted_p_value < 0.05 else 'Not Significant' 
              for p in results.pathways]
    
    fig = go.Figure()
    
    # Scatter plot
    for sig_status in ['Significant', 'Not Significant']:
        mask = [c == sig_status for c in colors]
        sizes_filtered = [s for s, m in zip(pathway_sizes, mask) if m]
        pvals_filtered = [p for p, m in zip(neg_log_pvalues, mask) if m]
        
        color = ColorPalette.get_colorblind_safe_palette()[1] if sig_status == 'Significant' else ColorPalette.get_colorblind_safe_palette()[6]
        
        fig.add_trace(go.Scatter(
            x=sizes_filtered,
            y=pvals_filtered,
            mode='markers',
            name=sig_status,
            marker=dict(color=color, size=6, opacity=0.6)
        ))
    
    # Add significance threshold line
    fig.add_hline(
        y=-np.log10(0.05),
        line_dash="dash",
        line_color="gray",
        annotation_text="p=0.05",
        annotation_position="right"
    )
    
    theme = get_plotly_theme("publication")
    fig.update_layout(
        title=title,
        xaxis_title="Pathway Size (genes)",
        yaxis_title="-log10(P-value)",
        **theme["layout"]
    )
    
    return fig


def create_diagnostic_panel(
    results: DatabaseResult,
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create comprehensive diagnostic panel with multiple plots.
    
    Args:
        results: DatabaseResult containing pathway information
        output_path: Optional path to save figure
        
    Returns:
        Plotly figure with subplots
    """
    pvalues = [p.p_value for p in results.pathways]
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "P-value Distribution",
            "Q-Q Plot",
            "Pathway Size Distribution",
            "Size vs Significance"
        ),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # P-value histogram
    fig.add_trace(
        go.Histogram(
            x=pvalues,
            nbinsx=20,
            marker_color=ColorPalette.get_colorblind_safe_palette()[1],
            name="P-values",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Q-Q plot
    pvalues_sorted = np.sort(pvalues)
    expected = np.linspace(0, 1, len(pvalues))
    fig.add_trace(
        go.Scatter(
            x=expected,
            y=pvalues_sorted,
            mode='markers',
            marker=dict(color=ColorPalette.get_colorblind_safe_palette()[2], size=4),
            name="Observed",
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name="Expected",
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Pathway size distribution
    pathway_sizes = [p.pathway_count for p in results.pathways]
    fig.add_trace(
        go.Histogram(
            x=pathway_sizes,
            nbinsx=30,
            marker_color=ColorPalette.get_colorblind_safe_palette()[3],
            name="Sizes",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Size vs significance
    neg_log_pvalues = [-np.log10(p) if p > 0 else 10 for p in pvalues]
    fig.add_trace(
        go.Scatter(
            x=pathway_sizes,
            y=neg_log_pvalues,
            mode='markers',
            marker=dict(
                color=ColorPalette.get_colorblind_safe_palette()[4],
                size=4,
                opacity=0.6
            ),
            name="Pathways",
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="P-value", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Expected P-value", row=1, col=2)
    fig.update_yaxes(title_text="Observed P-value", row=1, col=2)
    
    fig.update_xaxes(title_text="Pathway Size", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    fig.update_xaxes(title_text="Pathway Size", row=2, col=2)
    fig.update_yaxes(title_text="-log10(P-value)", row=2, col=2)
    
    # Update layout
    theme = get_plotly_theme("publication")
    fig.update_layout(
        title_text="Pathway Analysis Diagnostic Panel",
        height=800,
        width=1000,
        **theme["layout"]
    )
    
    if output_path:
        fig.write_image(output_path)
    
    return fig


def plot_pi0_estimation(
    pvalues: List[float],
    title: str = "π₀ Estimation (Storey's Method)"
) -> go.Figure:
    """
    Create π₀ (proportion of true nulls) estimation plot.
    
    Uses Storey's method to estimate the proportion of truly null
    hypotheses in the dataset.
    
    Args:
        pvalues: List of p-values
        title: Plot title
        
    Returns:
        Plotly figure
    """
    pvalues = np.array(pvalues)
    pvalues = pvalues[~np.isnan(pvalues)]
    n = len(pvalues)
    
    # Calculate π₀ at different λ thresholds
    lambdas = np.arange(0.05, 0.95, 0.05)
    pi0_estimates = []
    
    for lam in lambdas:
        n_above = np.sum(pvalues > lam)
        pi0 = n_above / (n * (1 - lam))
        pi0_estimates.append(min(pi0, 1.0))
    
    # Final estimate (average of high λ values)
    pi0_final = np.mean(pi0_estimates[-5:]) if len(pi0_estimates) >= 5 else np.mean(pi0_estimates)
    
    fig = go.Figure()
    
    # π₀ estimates
    fig.add_trace(go.Scatter(
        x=lambdas,
        y=pi0_estimates,
        mode='lines+markers',
        name='π₀(λ)',
        line=dict(color=ColorPalette.get_colorblind_safe_palette()[2]),
        marker=dict(size=6)
    ))
    
    # Final estimate line
    fig.add_hline(
        y=pi0_final,
        line_dash="dash",
        line_color=ColorPalette.get_colorblind_safe_palette()[0],
        annotation_text=f"Final π₀ = {pi0_final:.3f}",
        annotation_position="right"
    )
    
    theme = get_plotly_theme("publication")
    fig.update_layout(
        title=title,
        xaxis_title="λ threshold",
        yaxis_title="π₀ estimate",
        yaxis_range=[0, 1.1],
        **theme["layout"]
    )
    
    return fig


def generate_diagnostic_report(
    results: DatabaseResult,
    output_dir: str,
    formats: List[str] = None
) -> Dict[str, str]:
    """
    Generate comprehensive diagnostic report with all plots.
    
    Saves all diagnostic plots to the specified output directory.
    
    Args:
        results: DatabaseResult to analyze
        output_dir: Directory to save diagnostic outputs
        formats: Output formats (default: ["html"])
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    from pathlib import Path
    
    formats = formats or ["html"]
    output_path = Path(output_dir) / "diagnostics"
    output_path.mkdir(parents=True, exist_ok=True)
    
    pvalues = [p.p_value for p in results.pathways]
    output_files = {}
    
    # Generate all plots
    plots = {
        "pvalue_histogram": plot_pvalue_histogram(pvalues),
        "qq_plot": plot_qq_plot(pvalues),
        "pathway_size_distribution": plot_pathway_size_distribution(results),
        "size_vs_significance": plot_size_vs_significance(results),
        "pi0_estimation": plot_pi0_estimation(pvalues),
        "diagnostic_panel": create_diagnostic_panel(results)
    }
    
    for name, fig in plots.items():
        for fmt in formats:
            if fmt == "html":
                filepath = output_path / f"{name}.html"
                fig.write_html(str(filepath))
            else:
                filepath = output_path / f"{name}.{fmt}"
                fig.write_image(str(filepath))
            
            output_files[f"{name}_{fmt}"] = str(filepath)
    
    # Generate text summary
    summary = _generate_text_summary(results, pvalues)
    summary_path = output_path / "diagnostic_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    output_files["summary"] = str(summary_path)
    
    return output_files


def _generate_text_summary(results: DatabaseResult, pvalues: List[float]) -> str:
    """Generate text summary of diagnostic results."""
    pvalues = np.array(pvalues)
    pathway_sizes = [p.pathway_count for p in results.pathways]
    
    # Calculate statistics
    n = len(pvalues)
    n_sig = sum(1 for p in results.pathways if p.adjusted_p_value < 0.05)
    median_pval = np.median(pvalues)
    
    # KS test for uniformity
    ks_stat, ks_pval = stats.kstest(pvalues, 'uniform')
    
    # Spearman correlation for size bias
    spearman_r, spearman_p = stats.spearmanr(pathway_sizes, pvalues)
    
    # π₀ estimate
    lambdas = np.arange(0.5, 0.95, 0.05)
    pi0_estimates = [min(np.sum(pvalues > lam) / (n * (1 - lam)), 1.0) for lam in lambdas]
    pi0 = np.mean(pi0_estimates)
    
    lines = [
        "=" * 60,
        "PATHWAY ANALYSIS DIAGNOSTIC REPORT",
        "=" * 60,
        "",
        f"Database: {results.database.value}",
        f"Total Pathways: {n}",
        f"Significant Pathways (FDR < 0.05): {n_sig}",
        "",
        "P-VALUE DISTRIBUTION",
        "-" * 40,
        f"  Median p-value: {median_pval:.4f}",
        f"  % Significant (p<0.05): {100 * sum(pvalues < 0.05) / n:.1f}%",
        f"  KS test for uniformity: stat={ks_stat:.3f}, p={ks_pval:.4f}",
        f"  Estimated π₀ (true nulls): {pi0:.3f}",
        "",
        "PATHWAY SIZE BIAS",
        "-" * 40,
        f"  Spearman correlation (size vs p): {spearman_r:.3f}",
        f"  Correlation p-value: {spearman_p:.4f}",
        f"  Bias detected: {'YES' if spearman_p < 0.05 and spearman_r < -0.2 else 'NO'}",
        "",
        "INTERPRETATION",
        "-" * 40,
    ]
    
    if ks_pval < 0.01:
        lines.append("  ! P-value distribution deviates significantly from uniform")
    else:
        lines.append("  ✓ P-value distribution consistent with null expectations")
    
    if spearman_p < 0.05 and spearman_r < -0.2:
        lines.append("  ! Size bias detected - larger pathways tend to be more significant")
    else:
        lines.append("  ✓ No significant pathway size bias")
    
    if pi0 < 0.5:
        lines.append("  ✓ Strong true signal - many enriched pathways")
    elif pi0 > 0.9:
        lines.append("  ! Weak signal - few true enrichments detected")
    
    lines.extend(["", "=" * 60])
    
    return "\n".join(lines)

