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
