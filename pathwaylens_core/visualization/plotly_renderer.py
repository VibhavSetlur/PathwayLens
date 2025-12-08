"""
Plotly renderer for interactive visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger


class PlotlyRenderer:
    """Renderer for creating interactive Plotly visualizations."""
    
    def __init__(self):
        """Initialize the Plotly renderer."""
        self.logger = logger.bind(module="plotly_renderer")
    
    def create_dot_plot(
        self,
        data: pd.DataFrame,
        title: str = "Pathway Enrichment",
        x_col: str = "enrichment_score",
        y_col: str = "pathway_name",
        size_col: str = "overlap_count",
        color_col: str = "adjusted_p_value",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive dot plot.
        
        Args:
            data: DataFrame with pathway data
            title: Chart title
            x_col: Column for x-axis
            y_col: Column for y-axis
            size_col: Column for point size
            color_col: Column for point color
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            if data.empty:
                self.logger.warning("Empty data provided for dot plot")
                return self._create_empty_figure("No data available for Dot Plot")

            # Create scatter plot
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                size=size_col,
                color=color_col,
                hover_data={
                    x_col: ":.2f",
                    y_col: False,
                    size_col: True,
                    color_col: ":.2e"
                },
                title=title,
                labels={
                    x_col: "Enrichment Score",
                    y_col: "Pathway",
                    size_col: "Overlap Count",
                    color_col: "Adjusted P-value"
                }
            )
            
            # Add subtitle using annotation
            subtitle = f"Top {len(data)} pathways sorted by significance"
            fig.add_annotation(
                text=subtitle,
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=14, color="gray")
            )

            # Update layout
            fig.update_layout(
                width=1000,
                height=max(600, len(data) * 25),
                xaxis_title="Enrichment Score",
                yaxis_title="Pathway",
                showlegend=True,
                margin=dict(t=100)  # Make room for subtitle
            )
            
            # Add toggles (updatemenus)
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=list([
                            dict(
                                args=[{"marker.colorscale": "Viridis"}],
                                label="Viridis",
                                method="restyle"
                            ),
                            dict(
                                args=[{"marker.colorscale": "RdBu"}],
                                label="Red-Blue",
                                method="restyle"
                            ),
                            dict(
                                args=[{"marker.colorscale": "Jet"}],
                                label="Jet",
                                method="restyle"
                            )
                        ]),
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.0,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    )
                ]
            )

            # Update traces
            fig.update_traces(
                marker=dict(
                    line=dict(width=1, color='DarkSlateGrey'),
                    opacity=0.8
                ),
                hovertemplate="<b>%{y}</b><br>" +
                              "Enrichment Score: %{x:.2f}<br>" +
                              "Overlap Count: %{marker.size}<br>" +
                              "Adj. P-value: %{marker.color:.2e}<extra></extra>"
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Dot plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create dot plot: {e}")
            return self._create_empty_figure(f"Error creating Dot Plot: {str(e)}")

    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def create_volcano_plot(
        self,
        data: pd.DataFrame,
        title: str = "Volcano Plot",
        x_col: str = "log2_fold_enrichment",
        y_col: str = "neg_log10_p",
        text_col: str = "pathway_name",
        significance_threshold: float = 0.05,
        fold_change_threshold: float = 1.0,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive volcano plot.
        
        Args:
            data: DataFrame with pathway data
            title: Chart title
            x_col: Column for x-axis (log2 fold change)
            y_col: Column for y-axis (-log10 p-value)
            text_col: Column for hover text
            significance_threshold: P-value threshold
            fold_change_threshold: Fold change threshold
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            if data.empty:
                self.logger.warning("Empty data provided for volcano plot")
                return self._create_empty_figure("No data available for Volcano Plot")

            # Create scatter plot
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                hover_data={
                    x_col: ":.2f",
                    y_col: ":.2f",
                    text_col: True
                },
                title=title,
                labels={
                    x_col: "Log2 Fold Enrichment",
                    y_col: "-Log10 P-value",
                    text_col: "Pathway"
                }
            )
            
            # Add significance lines
            fig.add_hline(
                y=-np.log10(significance_threshold), 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"P = {significance_threshold}"
            )
            
            fig.add_vline(
                x=fold_change_threshold, 
                line_dash="dash", 
                line_color="blue",
                annotation_text=f"FC = {fold_change_threshold}"
            )
            
            fig.add_vline(
                x=-fold_change_threshold, 
                line_dash="dash", 
                line_color="blue"
            )
            
            # Add subtitle
            subtitle = f"Significance threshold: p < {significance_threshold}, |FC| > {fold_change_threshold}"
            fig.add_annotation(
                text=subtitle,
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=14, color="gray")
            )

            # Update layout
            fig.update_layout(
                width=1000,
                height=800,
                xaxis_title="Log2 Fold Enrichment",
                yaxis_title="-Log10 Adjusted P-value",
                showlegend=False,
                margin=dict(t=100)
            )
            
            # Update traces
            fig.update_traces(
                marker=dict(
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey'),
                    opacity=0.7
                )
            )
            

            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Volcano plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create volcano plot: {e}")
            return self._create_empty_figure(f"Error creating Volcano Plot: {str(e)}")

    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Pathway Heatmap",
        x_col: str = "database",
        y_col: str = "pathway_name",
        value_col: str = "enrichment_score",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive heatmap.
        
        Args:
            data: DataFrame with pathway data
            title: Chart title
            x_col: Column for x-axis
            y_col: Column for y-axis
            value_col: Column for heatmap values
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            if data.empty:
                self.logger.warning("Empty data provided for heatmap")
                return self._create_empty_figure("No data available for Heatmap")

            # Pivot data for heatmap if needed
            if not isinstance(data, pd.DataFrame):
                self.logger.error("Data must be a DataFrame")
                return self._create_empty_figure("Invalid data format")
                
            # Check if data is already pivoted or needs pivoting
            if not (x_col in data.columns and y_col in data.columns and value_col in data.columns):
                # Assume data is already in matrix form if columns don't match
                heatmap_data = data
            else:
                heatmap_data = data.pivot_table(
                    index=y_col, 
                    columns=x_col, 
                    values=value_col,
                    fill_value=0
                )
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>' +
                              '%{x}: %{z:.2f}<extra></extra>'
            ))
            
            # Add subtitle
            subtitle = f"Enrichment scores across {len(heatmap_data.columns)} databases"
            fig.add_annotation(
                text=subtitle,
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Database",
                yaxis_title="Pathway",
                width=1000,
                height=max(600, len(heatmap_data) * 25),
                margin=dict(t=100)
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Heatmap saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create heatmap: {e}")
            return self._create_empty_figure(f"Error creating Heatmap: {str(e)}")
    
    def create_network_plot(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "Pathway Network",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive network plot.
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            title: Chart title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            # Create network plot
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            for edge in edges:
                x0, y0 = edge.get('x0', 0), edge.get('y0', 0)
                x1, y1 = edge.get('x1', 0), edge.get('y1', 0)
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            node_x = [node.get('x', 0) for node in nodes]
            node_y = [node.get('y', 0) for node in nodes]
            node_text = [node.get('label', '') for node in nodes]
            node_sizes = [node.get('size', 10) for node in nodes]
            node_colors = [node.get('color', '#3498db') for node in nodes]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Interactive network plot",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="black", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,
                height=600
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Network plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create network plot: {e}")
            return go.Figure()
    
    def create_pca_plot(
        self,
        pca_data: pd.DataFrame,
        title: str = "PCA Plot",
        pc1_col: str = "PC1",
        pc2_col: str = "PC2",
        color_col: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive PCA plot.
        
        Args:
            pca_data: DataFrame with PCA data
            title: Chart title
            pc1_col: Column for PC1
            pc2_col: Column for PC2
            color_col: Column for coloring points
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            # Create PCA plot
            if color_col and color_col in pca_data.columns:
                fig = px.scatter(
                    pca_data,
                    x=pc1_col,
                    y=pc2_col,
                    color=color_col,
                    title=title,
                    labels={
                        pc1_col: "PC1",
                        pc2_col: "PC2"
                    }
                )
            else:
                fig = px.scatter(
                    pca_data,
                    x=pc1_col,
                    y=pc2_col,
                    title=title,
                    labels={
                        pc1_col: "PC1",
                        pc2_col: "PC2"
                    }
                )
            
            # Update layout
            fig.update_layout(
                width=800,
                height=600,
                xaxis_title="PC1",
                yaxis_title="PC2",
                showlegend=True
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"PCA plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create PCA plot: {e}")
            return go.Figure()
    
    def create_volcano_plot(
        self,
        data: pd.DataFrame,
        x_col: str = "log2FoldChange",
        y_col: str = "padj",
        gene_col: str = "gene_name",
        title: str = "Volcano Plot",
        output_file: Optional[str] = None,
        lfc_threshold: float = 1.0,
        p_threshold: float = 0.05
    ) -> Optional[str]:
        """
        Create a research-grade interactive volcano plot.
        
        Args:
            data: DataFrame containing results
            x_col: Column for log2 fold change
            y_col: Column for adjusted p-value
            gene_col: Column for gene names
            title: Plot title
            output_file: Path to save HTML file
            lfc_threshold: Log2 fold change threshold for significance
            p_threshold: Adjusted p-value threshold for significance
            
        Returns:
            Path to the generated plot file
        """
        try:
            df = data.copy()
            
            # Add significance column
            df['Significance'] = 'NS'
            df.loc[(df[x_col] > lfc_threshold) & (df[y_col] < p_threshold), 'Significance'] = 'Up'
            df.loc[(df[x_col] < -lfc_threshold) & (df[y_col] < p_threshold), 'Significance'] = 'Down'
            
            # Calculate -log10 p-value
            df['-log10(padj)'] = -np.log10(df[y_col].replace(0, np.nextafter(0, 1)))
            
            # Create plot
            fig = px.scatter(
                df,
                x=x_col,
                y='-log10(padj)',
                color='Significance',
                hover_data=[gene_col, x_col, y_col],
                title=title,
                color_discrete_map={'Up': '#e74c3c', 'Down': '#3498db', 'NS': '#bdc3c7'},
                template="plotly_white",
                opacity=0.8
            )
            
            # Add threshold lines
            fig.add_hline(y=-np.log10(p_threshold), line_dash="dash", line_color="grey", annotation_text=f"p={p_threshold}")
            fig.add_vline(x=lfc_threshold, line_dash="dash", line_color="grey", annotation_text=f"LFC={lfc_threshold}")
            fig.add_vline(x=-lfc_threshold, line_dash="dash", line_color="grey", annotation_text=f"LFC={-lfc_threshold}")
            
            # Update layout for publication quality
            fig.update_layout(
                font=dict(family="Arial", size=12),
                title_font=dict(size=20),
                xaxis_title="Log2 Fold Change",
                yaxis_title="-Log10 Adjusted P-value",
                legend_title="Regulation",
                hovermode="closest"
            )
            
            if output_file:
                fig.write_html(output_file)
                return output_file
            return fig.to_html()
            
        except Exception as e:
            self.logger.error(f"Failed to create volcano plot: {e}")
            return None.Figure()
    
    def create_dashboard(
        self,
        plots: Dict[str, go.Figure],
        title: str = "Analysis Dashboard",
        layout: str = "2x2",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create a dashboard with multiple plots.
        
        Args:
            plots: Dictionary of plot names to figures
            title: Dashboard title
            layout: Layout configuration
            output_file: Optional output file path
            
        Returns:
            Plotly figure object with subplots
        """
        try:
            plot_names = list(plots.keys())
            n_plots = len(plot_names)
            
            if n_plots == 0:
                return go.Figure()
            
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
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=plot_names[:rows*cols]
            )
            
            # Add plots to subplots
            for i, (plot_name, plot_fig) in enumerate(plots.items()):
                if i >= rows * cols:
                    break
                
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                for trace in plot_fig.data:
                    fig.add_trace(trace, row=row, col=col)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=600 * rows,
                width=800 * cols,
                showlegend=False
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Dashboard saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return go.Figure()
    
    def export_to_html(
        self,
        fig: go.Figure,
        output_file: str,
        include_plotlyjs: bool = True
    ) -> bool:
        """
        Export figure to HTML file.
        
        Args:
            fig: Plotly figure object
            output_file: Output file path
            include_plotlyjs: Whether to include Plotly.js library
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fig.write_html(
                output_file,
                include_plotlyjs=include_plotlyjs,
                full_html=True
            )
            self.logger.info(f"Figure exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export figure: {e}")
            return False
    
    def export_to_json(
        self,
        fig: go.Figure,
        output_file: str
    ) -> bool:
        """
        Export figure to JSON file.
        
        Args:
            fig: Plotly figure object
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fig.write_json(output_file)
            self.logger.info(f"Figure exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export figure: {e}")
            return False

    def create_upset_plot(
        self,
        sets: Dict[str, List[str]],
        title: str = "UpSet Plot",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create a simple UpSet-like plot using bar charts for intersections.

        Args:
            sets: Mapping of dataset name -> list of pathway IDs (or genes)
            title: Chart title
            output_file: Optional HTML output path

        Returns:
            Plotly figure
        """
        try:
            # Build intersection sizes for all non-empty combinations up to a reasonable cap
            from itertools import combinations

            set_items = {name: set(items) for name, items in sets.items()}
            names = list(set_items.keys())
            combos = []
            for r in range(1, min(len(names), 5) + 1):
                for combo in combinations(names, r):
                    inter = set.intersection(*(set_items[n] for n in combo))
                    if len(inter) > 0:
                        combos.append((combo, len(inter)))

            if not combos:
                fig = go.Figure()
                fig.add_annotation(text="No intersections", xref="paper", yref="paper", x=0.5, y=0.5)
                return fig

            # Sort by size desc
            combos.sort(key=lambda x: x[1], reverse=True)
            labels = [" ∩ ".join(c) for c, _ in combos]
            sizes = [s for _, s in combos]

            fig = go.Figure(data=[go.Bar(x=list(range(len(labels))), y=sizes)])
            fig.update_layout(
                title=title,
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(labels))),
                    ticktext=labels,
                    tickangle=45
                ),
                yaxis_title="Intersection size",
                width=max(800, 20 * len(labels)),
                height=500,
            )

            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"UpSet plot saved to {output_file}")

            return fig
        except Exception as e:
            self.logger.error(f"Failed to create UpSet plot: {e}")
            return go.Figure()
