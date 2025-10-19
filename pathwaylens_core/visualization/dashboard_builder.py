"""
Dashboard builder for creating comprehensive analysis dashboards.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class DashboardBuilder:
    """Builder for creating comprehensive analysis dashboards."""
    
    def __init__(self):
        """Initialize the dashboard builder."""
        self.logger = logger.bind(module="dashboard_builder")
    
    def create_analysis_dashboard(
        self,
        analysis_results: Dict[str, Any],
        title: str = "Pathway Analysis Dashboard",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create a comprehensive analysis dashboard.
        
        Args:
            analysis_results: Analysis results dictionary
            title: Dashboard title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object with dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Pathway Enrichment', 'Volcano Plot',
                    'Expression Heatmap', 'PCA Plot',
                    'Network Analysis', 'Summary Statistics'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "table"}]
                ]
            )
            
            # Add pathway enrichment plot
            if "pathway_results" in analysis_results:
                self._add_pathway_enrichment_plot(fig, analysis_results["pathway_results"], 1, 1)
            
            # Add volcano plot
            if "volcano_data" in analysis_results:
                self._add_volcano_plot(fig, analysis_results["volcano_data"], 1, 2)
            
            # Add expression heatmap
            if "expression_data" in analysis_results:
                self._add_expression_heatmap(fig, analysis_results["expression_data"], 2, 1)
            
            # Add PCA plot
            if "pca_data" in analysis_results:
                self._add_pca_plot(fig, analysis_results["pca_data"], 2, 2)
            
            # Add network plot
            if "network_data" in analysis_results:
                self._add_network_plot(fig, analysis_results["network_data"], 3, 1)
            
            # Add summary statistics table
            if "summary_stats" in analysis_results:
                self._add_summary_table(fig, analysis_results["summary_stats"], 3, 2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=1200,
                width=1200,
                showlegend=False
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Analysis dashboard saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis dashboard: {e}")
            return go.Figure()
    
    def create_comparison_dashboard(
        self,
        comparison_results: Dict[str, Any],
        title: str = "Comparison Analysis Dashboard",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create a comparison analysis dashboard.
        
        Args:
            comparison_results: Comparison results dictionary
            title: Dashboard title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object with dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Dataset Overlap', 'Analysis Correlation',
                    'Clustering Results', 'Pathway Concordance'
                ),
                specs=[
                    [{"type": "heatmap"}, {"type": "heatmap"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ]
            )
            
            # Add overlap heatmap
            if "overlap_analysis" in comparison_results:
                self._add_overlap_heatmap(fig, comparison_results["overlap_analysis"], 1, 1)
            
            # Add correlation heatmap
            if "correlation_analysis" in comparison_results:
                self._add_correlation_heatmap(fig, comparison_results["correlation_analysis"], 1, 2)
            
            # Add clustering plot
            if "clustering_analysis" in comparison_results:
                self._add_clustering_plot(fig, comparison_results["clustering_analysis"], 2, 1)
            
            # Add concordance plot
            if "overlap_analysis" in comparison_results:
                self._add_concordance_plot(fig, comparison_results["overlap_analysis"], 2, 2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                width=1000,
                showlegend=False
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Comparison dashboard saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison dashboard: {e}")
            return go.Figure()
    
    def create_multi_omics_dashboard(
        self,
        multi_omics_results: Dict[str, Any],
        title: str = "Multi-Omics Analysis Dashboard",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create a multi-omics analysis dashboard.
        
        Args:
            multi_omics_results: Multi-omics results dictionary
            title: Dashboard title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object with dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Genomics Results', 'Proteomics Results', 'Metabolomics Results',
                    'Integrated Analysis', 'Correlation Matrix', 'Summary'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "heatmap"}, {"type": "table"}]
                ]
            )
            
            # Add omics-specific plots
            omics_types = ["genomics", "proteomics", "metabolomics"]
            for i, omics_type in enumerate(omics_types):
                if f"{omics_type}_results" in multi_omics_results:
                    self._add_omics_plot(fig, multi_omics_results[f"{omics_type}_results"], 1, i+1, omics_type)
            
            # Add integrated analysis plot
            if "integrated_results" in multi_omics_results:
                self._add_integrated_plot(fig, multi_omics_results["integrated_results"], 2, 1)
            
            # Add correlation matrix
            if "correlation_matrix" in multi_omics_results:
                self._add_correlation_matrix(fig, multi_omics_results["correlation_matrix"], 2, 2)
            
            # Add summary table
            if "summary_stats" in multi_omics_results:
                self._add_summary_table(fig, multi_omics_results["summary_stats"], 2, 3)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                width=1200,
                showlegend=False
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Multi-omics dashboard saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-omics dashboard: {e}")
            return go.Figure()
    
    def _add_pathway_enrichment_plot(self, fig: go.Figure, pathway_results: List[Dict[str, Any]], row: int, col: int):
        """Add pathway enrichment plot to dashboard."""
        try:
            if not pathway_results:
                return
            
            # Extract data
            pathway_names = [p.get("pathway_name", p.get("pathway_id", "")) for p in pathway_results]
            enrichment_scores = [p.get("enrichment_score", 0.0) for p in pathway_results]
            p_values = [p.get("adjusted_p_value", 1.0) for p in pathway_results]
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=enrichment_scores,
                    y=pathway_names,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=p_values,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    name='Pathway Enrichment'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add pathway enrichment plot: {e}")
    
    def _add_volcano_plot(self, fig: go.Figure, volcano_data: Dict[str, Any], row: int, col: int):
        """Add volcano plot to dashboard."""
        try:
            if not volcano_data:
                return
            
            # Extract data
            x_data = volcano_data.get("x", [])
            y_data = volcano_data.get("y", [])
            gene_names = volcano_data.get("genes", [])
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=dict(size=6, color='blue', opacity=0.6),
                    text=gene_names,
                    name='Volcano Plot'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add volcano plot: {e}")
    
    def _add_expression_heatmap(self, fig: go.Figure, expression_data: Dict[str, Any], row: int, col: int):
        """Add expression heatmap to dashboard."""
        try:
            if not expression_data:
                return
            
            # Extract data
            data_matrix = expression_data.get("matrix", [])
            genes = expression_data.get("genes", [])
            samples = expression_data.get("samples", [])
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=data_matrix,
                    x=samples,
                    y=genes,
                    colorscale='RdBu',
                    showscale=False
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add expression heatmap: {e}")
    
    def _add_pca_plot(self, fig: go.Figure, pca_data: Dict[str, Any], row: int, col: int):
        """Add PCA plot to dashboard."""
        try:
            if not pca_data:
                return
            
            # Extract data
            pc1 = pca_data.get("PC1", [])
            pc2 = pca_data.get("PC2", [])
            labels = pca_data.get("labels", [])
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=pc1,
                    y=pc2,
                    mode='markers',
                    marker=dict(size=6, color='blue', opacity=0.6),
                    text=labels,
                    name='PCA Plot'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add PCA plot: {e}")
    
    def _add_network_plot(self, fig: go.Figure, network_data: Dict[str, Any], row: int, col: int):
        """Add network plot to dashboard."""
        try:
            if not network_data:
                return
            
            # Extract data
            nodes = network_data.get("nodes", [])
            edges = network_data.get("edges", [])
            
            # Add edges
            edge_x = []
            edge_y = []
            for edge in edges:
                x0, y0 = edge.get('x0', 0), edge.get('y0', 0)
                x1, y1 = edge.get('x1', 0), edge.get('y1', 0)
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines',
                    name='Network Edges'
                ),
                row=row, col=col
            )
            
            # Add nodes
            node_x = [node.get('x', 0) for node in nodes]
            node_y = [node.get('y', 0) for node in nodes]
            node_text = [node.get('label', '') for node in nodes]
            
            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    marker=dict(size=8, color='red'),
                    text=node_text,
                    textposition="middle center",
                    name='Network Nodes'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add network plot: {e}")
    
    def _add_summary_table(self, fig: go.Figure, summary_stats: Dict[str, Any], row: int, col: int):
        """Add summary statistics table to dashboard."""
        try:
            if not summary_stats:
                return
            
            # Create table data
            headers = list(summary_stats.keys())
            values = list(summary_stats.values())
            
            fig.add_trace(
                go.Table(
                    header=dict(values=headers),
                    cells=dict(values=[values])
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add summary table: {e}")
    
    def _add_overlap_heatmap(self, fig: go.Figure, overlap_data: Dict[str, Any], row: int, col: int):
        """Add overlap heatmap to dashboard."""
        try:
            if not overlap_data:
                return
            
            # Extract overlap matrix
            pairwise_overlaps = overlap_data.get("pairwise_overlaps", {})
            
            if not pairwise_overlaps:
                return
            
            # Create overlap matrix
            dataset_names = list(pairwise_overlaps.keys())
            overlap_matrix = np.zeros((len(dataset_names), len(dataset_names)))
            
            for i, dataset1 in enumerate(dataset_names):
                for j, dataset2 in enumerate(dataset_names):
                    if i == j:
                        overlap_matrix[i, j] = 1.0
                    elif dataset2 in pairwise_overlaps[dataset1]:
                        overlap_matrix[i, j] = pairwise_overlaps[dataset1][dataset2]["jaccard_index"]
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=overlap_matrix,
                    x=dataset_names,
                    y=dataset_names,
                    colorscale='Blues',
                    showscale=False
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add overlap heatmap: {e}")
    
    def _add_correlation_heatmap(self, fig: go.Figure, correlation_data: Dict[str, Any], row: int, col: int):
        """Add correlation heatmap to dashboard."""
        try:
            if not correlation_data:
                return
            
            # Extract correlation matrix
            correlation_matrix = correlation_data.get("correlation_matrix", {})
            
            if not correlation_matrix:
                return
            
            # Convert to DataFrame if needed
            if isinstance(correlation_matrix, dict):
                corr_df = pd.DataFrame(correlation_matrix)
            else:
                corr_df = correlation_matrix
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=False
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add correlation heatmap: {e}")
    
    def _add_clustering_plot(self, fig: go.Figure, clustering_data: Dict[str, Any], row: int, col: int):
        """Add clustering plot to dashboard."""
        try:
            if not clustering_data:
                return
            
            # Extract clustering data
            reduced_data = clustering_data.get("reduced_data", {})
            cluster_labels = clustering_data.get("cluster_labels", [])
            
            if not reduced_data or "pca" not in reduced_data:
                return
            
            pca_data = reduced_data["pca"]
            coordinates = pca_data["coordinates"]
            
            if len(coordinates) < 2:
                return
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    mode='markers',
                    marker=dict(size=6, color=cluster_labels, colorscale='Set1'),
                    name='Clustering Results'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add clustering plot: {e}")
    
    def _add_concordance_plot(self, fig: go.Figure, overlap_data: Dict[str, Any], row: int, col: int):
        """Add concordance plot to dashboard."""
        try:
            if not overlap_data:
                return
            
            # Extract concordance data
            pathway_concordance = overlap_data.get("pathway_concordance", {})
            
            if not pathway_concordance or "pathway_concordance" not in pathway_concordance:
                return
            
            concordance_data = pathway_concordance["pathway_concordance"]
            
            concordance_scores = []
            enrichment_scores = []
            
            for data in concordance_data.values():
                concordance_scores.append(data.get("concordance_score", 0.0))
                enrichment_scores.append(data.get("mean_enrichment_score", 0.0))
            
            if concordance_scores and enrichment_scores:
                fig.add_trace(
                    go.Scatter(
                        x=enrichment_scores,
                        y=concordance_scores,
                        mode='markers',
                        marker=dict(size=6, color='blue'),
                        name='Pathway Concordance'
                    ),
                    row=row, col=col
                )
            
        except Exception as e:
            self.logger.error(f"Failed to add concordance plot: {e}")
    
    def _add_omics_plot(self, fig: go.Figure, omics_results: Dict[str, Any], row: int, col: int, omics_type: str):
        """Add omics-specific plot to dashboard."""
        try:
            if not omics_results:
                return
            
            # Extract data
            pathways = omics_results.get("pathways", [])
            enrichment_scores = [p.get("enrichment_score", 0.0) for p in pathways]
            pathway_names = [p.get("pathway_name", p.get("pathway_id", "")) for p in pathways]
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=enrichment_scores,
                    y=pathway_names,
                    mode='markers',
                    marker=dict(size=8, color='blue'),
                    name=f'{omics_type.title()} Results'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add {omics_type} plot: {e}")
    
    def _add_integrated_plot(self, fig: go.Figure, integrated_results: Dict[str, Any], row: int, col: int):
        """Add integrated analysis plot to dashboard."""
        try:
            if not integrated_results:
                return
            
            # Extract data
            pathways = integrated_results.get("pathways", [])
            enrichment_scores = [p.get("enrichment_score", 0.0) for p in pathways]
            pathway_names = [p.get("pathway_name", p.get("pathway_id", "")) for p in pathways]
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=enrichment_scores,
                    y=pathway_names,
                    mode='markers',
                    marker=dict(size=8, color='green'),
                    name='Integrated Results'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add integrated plot: {e}")
    
    def _add_correlation_matrix(self, fig: go.Figure, correlation_matrix: Dict[str, Any], row: int, col: int):
        """Add correlation matrix to dashboard."""
        try:
            if not correlation_matrix:
                return
            
            # Extract correlation data
            if isinstance(correlation_matrix, dict):
                corr_df = pd.DataFrame(correlation_matrix)
            else:
                corr_df = correlation_matrix
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=False
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add correlation matrix: {e}")
