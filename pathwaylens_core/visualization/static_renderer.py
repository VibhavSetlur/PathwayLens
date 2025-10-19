"""
Static renderer for matplotlib-based visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import os


class StaticRenderer:
    """Renderer for creating static matplotlib visualizations."""
    
    def __init__(self, style: str = "whitegrid", dpi: int = 300):
        """
        Initialize the static renderer.
        
        Args:
            style: Matplotlib style
            dpi: DPI for saved figures
        """
        self.logger = logger.bind(module="static_renderer")
        self.style = style
        self.dpi = dpi
        
        # Set style
        sns.set_style(style)
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
    
    def create_dot_plot(
        self,
        data: pd.DataFrame,
        title: str = "Pathway Enrichment",
        x_col: str = "enrichment_score",
        y_col: str = "pathway_name",
        size_col: str = "overlap_count",
        color_col: str = "adjusted_p_value",
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a static dot plot.
        
        Args:
            data: DataFrame with pathway data
            title: Chart title
            x_col: Column for x-axis
            y_col: Column for y-axis
            size_col: Column for point size
            color_col: Column for point color
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.3)))
            
            # Create scatter plot
            scatter = ax.scatter(
                data[x_col],
                data[y_col],
                s=data[size_col] * 10,  # Scale size
                c=data[color_col],
                cmap='viridis',
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_col)
            
            # Set labels and title
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Dot plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create dot plot: {e}")
            return plt.figure()
    
    def create_volcano_plot(
        self,
        data: pd.DataFrame,
        title: str = "Volcano Plot",
        x_col: str = "log2_fold_change",
        y_col: str = "neg_log10_p_value",
        p_value_threshold: float = 0.05,
        fold_change_threshold: float = 1.0,
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a static volcano plot.
        
        Args:
            data: DataFrame with gene expression data
            title: Chart title
            x_col: Column for x-axis (log2 fold change)
            y_col: Column for y-axis (-log10 p-value)
            p_value_threshold: P-value threshold for significance
            fold_change_threshold: Fold change threshold for significance
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Create scatter plot
            ax.scatter(
                data[x_col],
                data[y_col],
                alpha=0.6,
                s=20,
                c='blue'
            )
            
            # Add significance thresholds
            ax.axhline(
                y=-np.log10(p_value_threshold),
                color='red',
                linestyle='--',
                alpha=0.8,
                label=f'P-value = {p_value_threshold}'
            )
            
            ax.axvline(
                x=fold_change_threshold,
                color='red',
                linestyle='--',
                alpha=0.8,
                label=f'Fold Change = {fold_change_threshold}'
            )
            
            ax.axvline(
                x=-fold_change_threshold,
                color='red',
                linestyle='--',
                alpha=0.8
            )
            
            # Color significant points
            significant_mask = (
                (abs(data[x_col]) >= fold_change_threshold) &
                (data[y_col] >= -np.log10(p_value_threshold))
            )
            
            if significant_mask.any():
                ax.scatter(
                    data.loc[significant_mask, x_col],
                    data.loc[significant_mask, y_col],
                    alpha=0.8,
                    s=30,
                    c='red',
                    label='Significant'
                )
            
            # Set labels and title
            ax.set_xlabel("Log2 Fold Change")
            ax.set_ylabel("-Log10 P-value")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Volcano plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create volcano plot: {e}")
            return plt.figure()
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Expression Heatmap",
        cmap: str = "viridis",
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a static heatmap.
        
        Args:
            data: DataFrame with expression data
            title: Chart title
            cmap: Colormap
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(
                data,
                cmap=cmap,
                center=0,
                ax=ax,
                cbar_kws={'label': 'Expression'}
            )
            
            # Set labels and title
            ax.set_title(title)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Gene")
            
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Heatmap saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create heatmap: {e}")
            return plt.figure()
    
    def create_network_plot(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "Pathway Network",
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a static network plot.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            title: Chart title
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Draw edges
            for edge in edges:
                x0, y0 = edge.get('x0', 0), edge.get('y0', 0)
                x1, y1 = edge.get('x1', 0), edge.get('y1', 0)
                ax.plot([x0, x1], [y0, y1], 'k-', alpha=0.5, linewidth=0.5)
            
            # Draw nodes
            for node in nodes:
                x, y = node.get('x', 0), node.get('y', 0)
                size = node.get('size', 100)
                color = node.get('color', 'blue')
                label = node.get('label', '')
                
                ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='white')
                
                if label:
                    ax.annotate(
                        label,
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8
                    )
            
            # Set labels and title
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Network plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create network plot: {e}")
            return plt.figure()
    
    def create_pca_plot(
        self,
        pca_data: pd.DataFrame,
        title: str = "PCA Plot",
        pc1_col: str = "PC1",
        pc2_col: str = "PC2",
        color_col: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a static PCA plot.
        
        Args:
            pca_data: DataFrame with PCA data
            title: Chart title
            pc1_col: Column for PC1
            pc2_col: Column for PC2
            color_col: Column for coloring points
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Create scatter plot
            if color_col and color_col in pca_data.columns:
                scatter = ax.scatter(
                    pca_data[pc1_col],
                    pca_data[pc2_col],
                    c=pca_data[color_col],
                    cmap='viridis',
                    alpha=0.7
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_col)
            else:
                ax.scatter(
                    pca_data[pc1_col],
                    pca_data[pc2_col],
                    alpha=0.7,
                    c='blue'
                )
            
            # Set labels and title
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"PCA plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA plot: {e}")
            return plt.figure()
    
    def create_dashboard(
        self,
        plots: Dict[str, plt.Figure],
        title: str = "Analysis Dashboard",
        layout: str = "2x2",
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a dashboard with multiple plots.
        
        Args:
            plots: Dictionary of plot names to figures
            title: Dashboard title
            layout: Layout configuration
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure object with subplots
        """
        try:
            plot_names = list(plots.keys())
            n_plots = len(plot_names)
            
            if n_plots == 0:
                return plt.figure()
            
            # Determine subplot layout
            if layout == "2x2":
                rows, cols = 2, 2
            elif layout == "1x3":
                rows, cols = 1, 3
            elif layout == "3x1":
                rows, cols = 3, 1
            else:
                rows, cols = 2, 2
            
            # Create subplots
            fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
            
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Add plots to subplots
            for i, (plot_name, plot_fig) in enumerate(plots.items()):
                if i >= rows * cols:
                    break
                
                # Copy plot to subplot
                axes[i].set_title(plot_name)
                # Note: This is a simplified version. In practice, you'd need to
                # properly copy the plot content to the subplot axes.
            
            # Set main title
            fig.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Dashboard saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return plt.figure()
    
    def export_to_png(
        self,
        fig: plt.Figure,
        output_file: str
    ) -> bool:
        """
        Export figure to PNG file.
        
        Args:
            fig: Matplotlib figure object
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Figure exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export figure: {e}")
            return False
    
    def export_to_pdf(
        self,
        fig: plt.Figure,
        output_file: str
    ) -> bool:
        """
        Export figure to PDF file.
        
        Args:
            fig: Matplotlib figure object
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fig.savefig(output_file, format='pdf', bbox_inches='tight')
            self.logger.info(f"Figure exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export figure: {e}")
            return False
    
    def export_to_svg(
        self,
        fig: plt.Figure,
        output_file: str
    ) -> bool:
        """
        Export figure to SVG file.
        
        Args:
            fig: Matplotlib figure object
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fig.savefig(output_file, format='svg', bbox_inches='tight')
            self.logger.info(f"Figure exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export figure: {e}")
            return False
