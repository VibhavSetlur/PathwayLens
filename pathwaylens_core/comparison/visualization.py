"""
Comparison visualization module for PathwayLens.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


class ComparisonVisualizer:
    """Visualizer for comparison analysis results."""
    
    def __init__(self):
        """Initialize the comparison visualizer."""
        self.logger = logger.bind(module="comparison_visualizer")
    
    def create_overlap_heatmap(
        self,
        overlap_data: Dict[str, Any],
        title: str = "Dataset Overlap Heatmap",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create heatmap showing overlaps between datasets.
        
        Args:
            overlap_data: Overlap analysis results
            title: Chart title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            pairwise_overlaps = overlap_data.get("pairwise_overlaps", {})
            
            if not pairwise_overlaps:
                self.logger.warning("No pairwise overlap data available")
                return go.Figure()
            
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
            fig = go.Figure(data=go.Heatmap(
                z=overlap_matrix,
                x=dataset_names,
                y=dataset_names,
                colorscale='Blues',
                text=np.round(overlap_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='Dataset 1: %{y}<br>Dataset 2: %{x}<br>Jaccard Index: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Dataset",
                yaxis_title="Dataset",
                width=600,
                height=500
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Overlap heatmap saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create overlap heatmap: {e}")
            return go.Figure()
    
    def create_correlation_heatmap(
        self,
        correlation_data: Dict[str, Any],
        title: str = "Correlation Heatmap",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create heatmap showing correlations between analyses.
        
        Args:
            correlation_data: Correlation analysis results
            title: Chart title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            correlation_matrix = correlation_data.get("correlation_matrix", {})
            
            if not correlation_matrix:
                self.logger.warning("No correlation matrix data available")
                return go.Figure()
            
            # Convert to DataFrame if needed
            if isinstance(correlation_matrix, dict):
                corr_df = pd.DataFrame(correlation_matrix)
            else:
                corr_df = correlation_matrix
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_df.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='Analysis 1: %{y}<br>Analysis 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Analysis",
                yaxis_title="Analysis",
                width=600,
                height=500
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Correlation heatmap saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create correlation heatmap: {e}")
            return go.Figure()
    
    def create_clustering_plot(
        self,
        clustering_data: Dict[str, Any],
        title: str = "Clustering Results",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create scatter plot showing clustering results.
        
        Args:
            clustering_data: Clustering analysis results
            title: Chart title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            reduced_data = clustering_data.get("reduced_data", {})
            cluster_labels = clustering_data.get("cluster_labels", [])
            analysis_names = clustering_data.get("analysis_names", [])
            
            if not reduced_data or "pca" not in reduced_data:
                self.logger.warning("No reduced data available for clustering plot")
                return go.Figure()
            
            pca_data = reduced_data["pca"]
            coordinates = pca_data["coordinates"]
            feature_names = pca_data["feature_names"]
            
            if len(coordinates) < 2:
                self.logger.warning("Insufficient data for clustering plot")
                return go.Figure()
            
            # Create scatter plot
            fig = go.Figure()
            
            # Group points by cluster
            unique_clusters = list(set(cluster_labels))
            colors = px.colors.qualitative.Set1
            
            for i, cluster in enumerate(unique_clusters):
                cluster_mask = np.array(cluster_labels) == cluster
                cluster_coords = np.array(coordinates)[cluster_mask]
                cluster_names = [feature_names[j] for j in range(len(feature_names)) if cluster_mask[j]]
                
                fig.add_trace(go.Scatter(
                    x=cluster_coords[:, 0],
                    y=cluster_coords[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    name=f'Cluster {cluster}',
                    text=cluster_names,
                    hovertemplate='%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="PC1",
                yaxis_title="PC2",
                width=600,
                height=500,
                showlegend=True
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Clustering plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create clustering plot: {e}")
            return go.Figure()
    
    def create_venn_diagram(
        self,
        overlap_data: Dict[str, Any],
        title: str = "Dataset Overlap",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create Venn diagram showing dataset overlaps.
        
        Args:
            overlap_data: Overlap analysis results
            title: Chart title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            venn_data = overlap_data.get("venn_diagram_data", {})
            
            if not venn_data or "datasets" not in venn_data:
                self.logger.warning("No Venn diagram data available")
                return go.Figure()
            
            datasets = venn_data["datasets"]
            overlaps = venn_data.get("overlaps", {})
            
            if len(datasets) > 3:
                self.logger.warning("Venn diagram not suitable for more than 3 datasets")
                return go.Figure()
            
            # For simplicity, create a bar chart showing overlap sizes
            overlap_names = []
            overlap_sizes = []
            
            for overlap_name, overlap_info in overlaps.items():
                if isinstance(overlap_info, dict) and "size" in overlap_info:
                    overlap_names.append(overlap_name.replace("_", " ∩ "))
                    overlap_sizes.append(overlap_info["size"])
            
            # Create bar chart
            fig = go.Figure(data=go.Bar(
                x=overlap_names,
                y=overlap_sizes,
                marker_color='lightblue',
                text=overlap_sizes,
                textposition='auto',
                hovertemplate='%{x}<br>Size: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Overlap",
                yaxis_title="Number of Genes",
                width=600,
                height=400,
                xaxis_tickangle=-45
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Venn diagram saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create Venn diagram: {e}")
            return go.Figure()
    
    def create_concordance_plot(
        self,
        overlap_data: Dict[str, Any],
        title: str = "Pathway Concordance",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create plot showing pathway concordance between analyses.
        
        Args:
            overlap_data: Overlap analysis results
            title: Chart title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            pathway_concordance = overlap_data.get("pathway_concordance", {})
            
            if not pathway_concordance or "pathway_concordance" not in pathway_concordance:
                self.logger.warning("No pathway concordance data available")
                return go.Figure()
            
            concordance_data = pathway_concordance["pathway_concordance"]
            
            if not concordance_data:
                self.logger.warning("No pathway concordance data available")
                return go.Figure()
            
            # Extract concordance scores
            pathway_ids = []
            concordance_scores = []
            enrichment_scores = []
            
            for pathway_id, data in concordance_data.items():
                pathway_ids.append(pathway_id)
                concordance_scores.append(data.get("concordance_score", 0.0))
                enrichment_scores.append(data.get("mean_enrichment_score", 0.0))
            
            # Create scatter plot
            fig = go.Figure(data=go.Scatter(
                x=enrichment_scores,
                y=concordance_scores,
                mode='markers',
                marker=dict(
                    size=8,
                    color=concordance_scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Concordance Score")
                ),
                text=pathway_ids,
                hovertemplate='Pathway: %{text}<br>Enrichment Score: %{x:.3f}<br>Concordance Score: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Mean Enrichment Score",
                yaxis_title="Concordance Score",
                width=600,
                height=500
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Concordance plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create concordance plot: {e}")
            return go.Figure()
    
    def create_comparison_dashboard(
        self,
        comparison_results: Dict[str, Any],
        output_dir: str = "comparison_plots"
    ) -> Dict[str, str]:
        """
        Create a comprehensive comparison dashboard.
        
        Args:
            comparison_results: Combined comparison analysis results
            output_dir: Output directory for plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            plot_files = {}
            
            # Overlap analysis plots
            if "overlap_analysis" in comparison_results:
                overlap_data = comparison_results["overlap_analysis"]
                
                # Overlap heatmap
                overlap_heatmap = self.create_overlap_heatmap(
                    overlap_data, 
                    "Dataset Overlap Heatmap",
                    os.path.join(output_dir, "overlap_heatmap.html")
                )
                plot_files["overlap_heatmap"] = os.path.join(output_dir, "overlap_heatmap.html")
                
                # Venn diagram
                venn_diagram = self.create_venn_diagram(
                    overlap_data,
                    "Dataset Overlap",
                    os.path.join(output_dir, "venn_diagram.html")
                )
                plot_files["venn_diagram"] = os.path.join(output_dir, "venn_diagram.html")
                
                # Concordance plot
                concordance_plot = self.create_concordance_plot(
                    overlap_data,
                    "Pathway Concordance",
                    os.path.join(output_dir, "concordance_plot.html")
                )
                plot_files["concordance_plot"] = os.path.join(output_dir, "concordance_plot.html")
            
            # Correlation analysis plots
            if "correlation_analysis" in comparison_results:
                correlation_data = comparison_results["correlation_analysis"]
                
                # Correlation heatmap
                correlation_heatmap = self.create_correlation_heatmap(
                    correlation_data,
                    "Analysis Correlation Heatmap",
                    os.path.join(output_dir, "correlation_heatmap.html")
                )
                plot_files["correlation_heatmap"] = os.path.join(output_dir, "correlation_heatmap.html")
            
            # Clustering analysis plots
            if "clustering_analysis" in comparison_results:
                clustering_data = comparison_results["clustering_analysis"]
                
                # Clustering plot
                clustering_plot = self.create_clustering_plot(
                    clustering_data,
                    "Analysis Clustering",
                    os.path.join(output_dir, "clustering_plot.html")
                )
                plot_files["clustering_plot"] = os.path.join(output_dir, "clustering_plot.html")
            
            self.logger.info(f"Comparison dashboard created with {len(plot_files)} plots")
            return plot_files
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison dashboard: {e}")
            return {}
    
    def create_summary_plot(
        self,
        comparison_results: Dict[str, Any],
        title: str = "Comparison Analysis Summary",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create a summary plot combining multiple comparison metrics.
        
        Args:
            comparison_results: Combined comparison analysis results
            title: Chart title
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Dataset Overlap', 'Analysis Correlation', 
                              'Clustering Results', 'Concordance Score'),
                specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Add overlap heatmap
            if "overlap_analysis" in comparison_results:
                overlap_data = comparison_results["overlap_analysis"]
                pairwise_overlaps = overlap_data.get("pairwise_overlaps", {})
                
                if pairwise_overlaps:
                    dataset_names = list(pairwise_overlaps.keys())
                    overlap_matrix = np.zeros((len(dataset_names), len(dataset_names)))
                    
                    for i, dataset1 in enumerate(dataset_names):
                        for j, dataset2 in enumerate(dataset_names):
                            if i == j:
                                overlap_matrix[i, j] = 1.0
                            elif dataset2 in pairwise_overlaps[dataset1]:
                                overlap_matrix[i, j] = pairwise_overlaps[dataset1][dataset2]["jaccard_index"]
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=overlap_matrix,
                            x=dataset_names,
                            y=dataset_names,
                            colorscale='Blues',
                            showscale=False
                        ),
                        row=1, col=1
                    )
            
            # Add correlation heatmap
            if "correlation_analysis" in comparison_results:
                correlation_data = comparison_results["correlation_analysis"]
                correlation_matrix = correlation_data.get("correlation_matrix", {})
                
                if correlation_matrix:
                    if isinstance(correlation_matrix, dict):
                        corr_df = pd.DataFrame(correlation_matrix)
                    else:
                        corr_df = correlation_matrix
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=corr_df.values,
                            x=corr_df.columns,
                            y=corr_df.index,
                            colorscale='RdBu',
                            zmid=0,
                            showscale=False
                        ),
                        row=1, col=2
                    )
            
            # Add clustering plot
            if "clustering_analysis" in comparison_results:
                clustering_data = comparison_results["clustering_analysis"]
                reduced_data = clustering_data.get("reduced_data", {})
                cluster_labels = clustering_data.get("cluster_labels", [])
                
                if reduced_data and "pca" in reduced_data:
                    pca_data = reduced_data["pca"]
                    coordinates = pca_data["coordinates"]
                    
                    if len(coordinates) >= 2:
                        unique_clusters = list(set(cluster_labels))
                        colors = px.colors.qualitative.Set1
                        
                        for i, cluster in enumerate(unique_clusters):
                            cluster_mask = np.array(cluster_labels) == cluster
                            cluster_coords = np.array(coordinates)[cluster_mask]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=cluster_coords[:, 0],
                                    y=cluster_coords[:, 1],
                                    mode='markers',
                                    marker=dict(size=6, color=colors[i % len(colors)]),
                                    name=f'Cluster {cluster}',
                                    showlegend=False
                                ),
                                row=2, col=1
                            )
            
            # Add concordance plot
            if "overlap_analysis" in comparison_results:
                overlap_data = comparison_results["overlap_analysis"]
                pathway_concordance = overlap_data.get("pathway_concordance", {})
                
                if pathway_concordance and "pathway_concordance" in pathway_concordance:
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
                                showlegend=False
                            ),
                            row=2, col=2
                        )
            
            fig.update_layout(
                title=title,
                height=800,
                width=1000,
                showlegend=False
            )
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Summary plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create summary plot: {e}")
            return go.Figure()
