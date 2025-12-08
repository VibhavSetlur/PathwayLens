"""
Visualization engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

from .schemas import (
    VisualizationResult, VisualizationParameters, PlotType, PlotMetadata,
    DashboardConfig, ExportConfig
)
from ..analysis.schemas import AnalysisResult
from ..comparison.schemas import ComparisonResult
from .plotly_renderer import PlotlyRenderer
from .multi_omics_visualizer import MultiOmicsVisualizer
from .themes import ThemeManager


class VisualizationEngine:
    """Main visualization engine for generating plots and dashboards."""
    
    def __init__(self):
        """Initialize the visualization engine."""
        self.logger = logger.bind(module="visualization_engine")
        self.renderer = PlotlyRenderer()
        self.multi_omics_visualizer = MultiOmicsVisualizer()
        self.theme_manager = ThemeManager()
        self.plot_renderers = {
            PlotType.DOT_PLOT: self._create_dot_plot,
            PlotType.VOLCANO_PLOT: self._create_volcano_plot,
            PlotType.HEATMAP: self._create_heatmap,
            PlotType.NETWORK: self._create_network_plot,
            PlotType.BAR_CHART: self._create_bar_chart,
            PlotType.SCATTER_PLOT: self._create_scatter_plot,
            PlotType.PCA_PLOT: self._create_pca_plot,
            PlotType.UMAP_PLOT: self._create_umap_plot,
            PlotType.MANHATTAN_PLOT: self._create_manhattan_plot,
            PlotType.ENRICHMENT_PLOT: self._create_enrichment_plot,
            PlotType.PATHWAY_MAP: self._create_pathway_map,
            PlotType.GENE_EXPRESSION: self._create_gene_expression_plot,
            PlotType.TIME_SERIES: self._create_time_series_plot,
            PlotType.COMPARISON_PLOT: self._create_comparison_plot,
            PlotType.CONSENSUS_PLOT: self._create_consensus_plot,
            PlotType.UPSET_PLOT: self._create_upset_plot,
            PlotType.MULTI_OMICS_HEATMAP: self._create_multi_omics_heatmap,
            PlotType.MULTI_OMICS_NETWORK: self._create_multi_omics_network,
            PlotType.MULTI_OMICS_SANKEY: self._create_multi_omics_sankey
        }
    
    async def _create_multi_omics_heatmap(
        self,
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]],
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create multi-omics heatmap visualization."""
        if isinstance(data, dict) and 'omics_data' in data:
            omics_data = data['omics_data']
        elif isinstance(data, dict) and 'multi_omics_results' in data:
            omics_data = data['multi_omics_results']
        else:
            return await self._create_heatmap(data, parameters)
        
        return self.multi_omics_visualizer.create_multi_omics_heatmap(
            omics_data, max_pathways=parameters.max_pathways
        )
    
    async def _create_multi_omics_network(
        self,
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]],
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create multi-omics network visualization."""
        if isinstance(data, dict) and 'cross_omics_network' in data:
            network = data['cross_omics_network']
        else:
            return await self._create_network_plot(data, parameters)
        
        return self.multi_omics_visualizer.create_multi_omics_network(
            network, layout=parameters.network_layout
        )
    
    async def _create_multi_omics_sankey(
        self,
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]],
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create Sankey diagram for multi-omics data flow."""
        if isinstance(data, dict) and 'cross_omics_mapping' in data:
            mapping = data['cross_omics_mapping']
        else:
            return go.Figure().add_annotation(
                text="Multi-omics mapping data required for Sankey diagram",
                xref="paper", yref="paper", x=0.5, y=0.5
            )
        
        return self.multi_omics_visualizer.create_multi_omics_sankey(mapping)
    
    async def visualize(
        self,
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]],
        parameters: VisualizationParameters,
        output_dir: Optional[str] = None
    ) -> VisualizationResult:
        """
        Generate visualizations based on input data and parameters.
        
        Args:
            data: Analysis or comparison results to visualize
            parameters: Visualization parameters
            output_dir: Output directory for generated files
            
        Returns:
            VisualizationResult with generated plots
        """
        self.logger.info(f"Starting visualization generation with {len(parameters.plot_types)} plot types")
        
        try:
            # Prepare output directory
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path.cwd() / "visualizations"
                output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            generated_plots = {}
            plot_metadata = {}
            total_size = 0
            
            for plot_type in parameters.plot_types:
                self.logger.info(f"Generating {plot_type.value} plot")
                
                try:
                    plot_file, metadata = await self._generate_plot(
                        data, plot_type, parameters, output_path
                    )
                    
                    generated_plots[plot_type] = plot_file
                    plot_metadata[plot_type] = metadata
                    
                    # Calculate file size
                    if Path(plot_file).exists():
                        total_size += Path(plot_file).stat().st_size
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate {plot_type.value} plot: {e}")
                    continue
            
            # Generate dashboard if requested
            dashboard_file = None
            if parameters.interactive and len(generated_plots) > 1:
                dashboard_file = await self._generate_dashboard(
                    generated_plots, plot_metadata, parameters, output_path
                )
            
            # Calculate quality metrics
            plot_quality = self._calculate_plot_quality(plot_metadata)
            data_coverage = self._calculate_data_coverage(data, plot_metadata)
            
            # Create visualization result
            result = VisualizationResult(
                job_id=f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                input_file=str(data) if isinstance(data, str) else "in_memory_data",
                parameters=parameters,
                generated_plots=generated_plots,
                plot_metadata=plot_metadata,
                total_plots=len(generated_plots),
                total_size=total_size,
                processing_time=0.0,  # Would be calculated in actual implementation
                plot_quality=plot_quality,
                data_coverage=data_coverage,
                created_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat(),
                dashboard_file=dashboard_file
            )
            
            self.logger.info(f"Visualization generation completed: {len(generated_plots)} plots generated")
            return result
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise
    
    async def _generate_plot(
        self,
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]],
        plot_type: PlotType,
        parameters: VisualizationParameters,
        output_path: Path
    ) -> Tuple[str, PlotMetadata]:
        """Generate a single plot."""
        # Get plot renderer
        renderer = self.plot_renderers.get(plot_type)
        if not renderer:
            raise ValueError(f"No renderer available for plot type: {plot_type}")
        
        # Generate plot
        fig = await renderer(data, parameters)
        
        # Apply theme
        if parameters.theme:
            fig = self.theme_manager.apply_plotly_theme(fig, parameters.theme)
        
        # Save plot
        # Save plot
        filename_base = f"{plot_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        generated_files = []
        
        for fmt in parameters.output_formats:
            fmt = fmt.lower()
            plot_file = output_path / f"{filename_base}.{fmt}"
            
            if fmt == "html":
                fig.write_html(str(plot_file))
            elif fmt in ["png", "jpg", "jpeg", "webp"]:
                # High-res raster
                fig.write_image(
                    str(plot_file), 
                    width=parameters.figure_size[0], 
                    height=parameters.figure_size[1],
                    scale=3  # 3x scale for high DPI (approx 300 DPI)
                )
            elif fmt in ["svg", "pdf", "eps"]:
                # Vector formats
                fig.write_image(
                    str(plot_file), 
                    width=parameters.figure_size[0], 
                    height=parameters.figure_size[1]
                )
            
            generated_files.append(str(plot_file))
            
        # Use the first generated file as the primary one for metadata
        plot_file = generated_files[0] if generated_files else ""
        
        # Create metadata
        metadata = PlotMetadata(
            plot_type=plot_type,
            title=fig.layout.title.text if fig.layout.title else f"{plot_type.value.replace('_', ' ').title()}",
            description=f"Generated {plot_type.value} plot",
            num_data_points=self._count_data_points(data, plot_type),
            num_pathways=self._count_pathways(data),
            num_genes=self._count_genes(data),
            significance_threshold=parameters.significance_threshold,
            num_significant=self._count_significant(data, parameters.significance_threshold),
            theme=parameters.theme,
            color_scheme=parameters.color_scheme,
            figure_size=parameters.figure_size,
            interactive=parameters.interactive,
            output_formats=parameters.output_formats,
            file_size=Path(plot_file).stat().st_size if Path(plot_file).exists() else 0,
            created_at=datetime.now().isoformat(),
            processing_time=0.0
        )
        
        return str(plot_file), metadata
    
    async def _create_dot_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create dot plot for pathway enrichment results."""
        # Extract pathway data
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for plotting
        df = pd.DataFrame(pathway_data)
        
        # Filter by significance
        df = df[df['adjusted_p_value'] <= parameters.significance_threshold]
        
        # Limit number of pathways
        df = df.head(parameters.max_pathways)
        
        # Create dot plot
        fig = go.Figure()
        
        # Add dots for each database
        databases = df['database'].unique()
        colors = px.colors.qualitative.Set1[:len(databases)]
        
        for i, database in enumerate(databases):
            db_data = df[df['database'] == database]
            
            fig.add_trace(go.Scatter(
                x=db_data['overlap_count'],
                y=db_data['pathway_name'],
                mode='markers',
                marker=dict(
                    size=db_data['overlap_count'],
                    color=colors[i],
                    opacity=0.7,
                    sizemode='diameter',
                    sizemin=5
                ),
                text=db_data.apply(lambda row: f"P-value: {row['adjusted_p_value']:.2e}<br>"
                                             f"Overlap: {row['overlap_count']}/{row['pathway_count']}<br>"
                                             f"Database: {row['database']}", axis=1),
                hovertemplate='%{text}<extra></extra>',
                name=database
            ))
        
        fig.update_layout(
            title="Pathway Enrichment Dot Plot",
            xaxis_title="Number of Overlapping Genes",
            yaxis_title="Pathway",
            showlegend=True,
            height=max(400, len(df) * 20)
        )
        
        return fig
    
    async def _create_volcano_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create volcano plot for pathway enrichment results."""
        # Extract pathway data
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for plotting
        df = pd.DataFrame(pathway_data)
        
        # Calculate -log10(p-value)
        df['neg_log10_p'] = -np.log10(df['adjusted_p_value'] + 1e-300)
        
        # Calculate fold enrichment (log2)
        df['log2_fold_enrichment'] = np.log2(df['enrichment_score'] + 1e-300)
        
        # Create volcano plot
        fig = go.Figure()
        
        # Color points based on significance and fold change
        colors = []
        for _, row in df.iterrows():
            if (row['adjusted_p_value'] <= parameters.significance_threshold and 
                abs(row['log2_fold_enrichment']) >= np.log2(parameters.fold_change_threshold)):
                colors.append('red')  # Significant and high fold change
            elif row['adjusted_p_value'] <= parameters.significance_threshold:
                colors.append('orange')  # Significant but low fold change
            else:
                colors.append('gray')  # Not significant
        
        fig.add_trace(go.Scatter(
            x=df['log2_fold_enrichment'],
            y=df['neg_log10_p'],
            mode='markers',
            marker=dict(
                color=colors,
                size=8,
                opacity=0.7
            ),
            text=df['pathway_name'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Log2 Fold Enrichment: %{x:.2f}<br>' +
                         '-Log10 P-value: %{y:.2f}<br>' +
                         'Adjusted P-value: ' + df['adjusted_p_value'].apply(lambda x: f'{x:.2e}') + '<extra></extra>',
            name='Pathways'
        ))
        
        # Add significance lines
        fig.add_hline(y=-np.log10(parameters.significance_threshold), 
                     line_dash="dash", line_color="red", 
                     annotation_text=f"P = {parameters.significance_threshold}")
        
        fig.add_vline(x=np.log2(parameters.fold_change_threshold), 
                     line_dash="dash", line_color="blue",
                     annotation_text=f"FC = {parameters.fold_change_threshold}")
        
        fig.add_vline(x=-np.log2(parameters.fold_change_threshold), 
                     line_dash="dash", line_color="blue")
        
        fig.update_layout(
            title="Volcano Plot - Pathway Enrichment",
            xaxis_title="Log2 Fold Enrichment",
            yaxis_title="-Log10 Adjusted P-value",
            height=600
        )
        
        return fig
    
    async def _create_heatmap(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create heatmap for pathway enrichment results."""
        # Extract pathway data
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for heatmap
        df = pd.DataFrame(pathway_data)
        
        # Pivot table for heatmap
        heatmap_data = df.pivot_table(
            values='enrichment_score',
            index='pathway_name',
            columns='database',
            fill_value=0
        )
        
        # Limit number of pathways
        heatmap_data = heatmap_data.head(parameters.max_pathways)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         'Database: %{x}<br>' +
                         'Enrichment Score: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Pathway Enrichment Heatmap",
            xaxis_title="Database",
            yaxis_title="Pathway",
            height=max(400, len(heatmap_data) * 20)
        )
        
        return fig
    
    async def _create_network_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create network plot for pathway relationships."""
        # This is a simplified network plot - in practice, would use actual pathway networks
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for network
        df = pd.DataFrame(pathway_data)
        df = df[df['adjusted_p_value'] <= parameters.significance_threshold]
        df = df.head(parameters.max_pathways)
        
        # Create simple network layout
        n_pathways = len(df)
        angles = np.linspace(0, 2*np.pi, n_pathways, endpoint=False)
        
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Create network plot
        fig = go.Figure()
        
        # Add nodes (pathways)
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=df['overlap_count'] * 2,
                color=df['enrichment_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Enrichment Score")
            ),
            text=df['pathway_name'],
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Overlap: ' + df['overlap_count'].astype(str) + '<br>' +
                         'Enrichment: ' + df['enrichment_score'].round(2).astype(str) + '<extra></extra>',
            name='Pathways'
        ))
        
        # Add edges (simplified - would use actual pathway relationships)
        for i in range(n_pathways):
            for j in range(i+1, n_pathways):
                # Simple edge based on shared genes (simplified)
                if abs(df.iloc[i]['enrichment_score'] - df.iloc[j]['enrichment_score']) < 0.5:
                    fig.add_trace(go.Scatter(
                        x=[x_pos[i], x_pos[j]],
                        y=[y_pos[i], y_pos[j]],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        fig.update_layout(
            title="Pathway Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    async def _create_bar_chart(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create bar chart for pathway enrichment results."""
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for bar chart
        df = pd.DataFrame(pathway_data)
        df = df[df['adjusted_p_value'] <= parameters.significance_threshold]
        df = df.head(parameters.max_pathways)
        
        # Sort by enrichment score
        df = df.sort_values('enrichment_score', ascending=True)
        
        # Create bar chart
        fig = go.Figure(data=go.Bar(
            x=df['enrichment_score'],
            y=df['pathway_name'],
            orientation='h',
            marker=dict(
                color=df['enrichment_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Enrichment Score")
            ),
            text=df['enrichment_score'].round(2),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' +
                         'Enrichment Score: %{x:.2f}<br>' +
                         'P-value: ' + df['adjusted_p_value'].apply(lambda x: f'{x:.2e}') + '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Pathway Enrichment Bar Chart",
            xaxis_title="Enrichment Score",
            yaxis_title="Pathway",
            height=max(400, len(df) * 30)
        )
        
        return fig
    
    async def _create_scatter_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create scatter plot for pathway enrichment results."""
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for scatter plot
        df = pd.DataFrame(pathway_data)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add points for each database
        databases = df['database'].unique()
        colors = px.colors.qualitative.Set1[:len(databases)]
        
        for i, database in enumerate(databases):
            db_data = df[df['database'] == database]
            
            fig.add_trace(go.Scatter(
                x=db_data['overlap_count'],
                y=db_data['enrichment_score'],
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=8,
                    opacity=0.7
                ),
                text=db_data['pathway_name'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Overlap: %{x}<br>' +
                             'Enrichment: %{y:.2f}<br>' +
                             'Database: ' + database + '<extra></extra>',
                name=database
            ))
        
        fig.update_layout(
            title="Pathway Enrichment Scatter Plot",
            xaxis_title="Number of Overlapping Genes",
            yaxis_title="Enrichment Score",
            height=600
        )
        
        return fig
    
    async def _create_pca_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create PCA plot for pathway enrichment results."""
        # This is a simplified PCA plot - in practice, would use actual PCA
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for PCA
        df = pd.DataFrame(pathway_data)
        
        # Create simplified 2D representation
        x_pos = df['enrichment_score'] * np.cos(df['overlap_count'] * 0.1)
        y_pos = df['enrichment_score'] * np.sin(df['overlap_count'] * 0.1)
        
        # Create PCA plot
        fig = go.Figure(data=go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=df['overlap_count'],
                color=df['adjusted_p_value'],
                colorscale='Viridis_r',
                showscale=True,
                colorbar=dict(title="Adjusted P-value")
            ),
            text=df['pathway_name'],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
                         'PC1: %{x:.2f}<br>' +
                         'PC2: %{y:.2f}<extra></extra>',
            name='Pathways'
        ))
        
        fig.update_layout(
            title="PCA Plot - Pathway Enrichment",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            height=600
        )
        
        return fig
    
    async def _create_umap_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create UMAP plot for pathway enrichment results."""
        # This is a simplified UMAP plot - in practice, would use actual UMAP
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for UMAP
        df = pd.DataFrame(pathway_data)
        
        # Create simplified 2D representation
        x_pos = df['enrichment_score'] * np.random.normal(1, 0.1, len(df))
        y_pos = df['overlap_count'] * np.random.normal(1, 0.1, len(df))
        
        # Create UMAP plot
        fig = go.Figure(data=go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=df['overlap_count'],
                color=df['adjusted_p_value'],
                colorscale='Viridis_r',
                showscale=True,
                colorbar=dict(title="Adjusted P-value")
            ),
            text=df['pathway_name'],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
                         'UMAP1: %{x:.2f}<br>' +
                         'UMAP2: %{y:.2f}<extra></extra>',
            name='Pathways'
        ))
        
        fig.update_layout(
            title="UMAP Plot - Pathway Enrichment",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            height=600
        )
        
        return fig

    async def _create_upset_plot(
        self,
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]],
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create UpSet plot from comparison results or raw sets."""
        try:
            # If data is a ComparisonResult, derive sets of pathways per dataset
            if isinstance(data, ComparisonResult):
                sets: Dict[str, List[str]] = {}
                # Rebuild per-dataset pathway sets from database_results
                # ComparisonResult may not carry raw per-dataset results, so expect optional attachment under extra fields
                extra = getattr(data, "extra", None)
                if extra and isinstance(extra, dict) and "dataset_pathways" in extra:
                    sets = {k: list(v) for k, v in extra["dataset_pathways"].items()}
                else:
                    # Fallback: use pathway_concordance to infer presence across datasets
                    presence: Dict[str, set] = {}
                    for pc in getattr(data, "pathway_concordance", []) or []:
                        # pc.p_values has keys for dataset names where pathway considered
                        for ds in pc.p_values.keys():
                            presence.setdefault(ds, set()).add(f"{pc.database}_{pc.pathway_id}")
                    sets = {k: list(v) for k, v in presence.items()}
            elif isinstance(data, dict) and "sets" in data:
                sets = {k: list(v) for k, v in data["sets"].items()}
            else:
                sets = {}

            return self.renderer.create_upset_plot(sets, title="UpSet - Dataset intersections")
        except Exception as e:
            self.logger.error(f"Failed to create UpSet plot: {e}")
            return go.Figure()
    
    async def _create_manhattan_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create Manhattan plot for pathway enrichment results."""
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for Manhattan plot
        df = pd.DataFrame(pathway_data)
        
        # Calculate -log10(p-value)
        df['neg_log10_p'] = -np.log10(df['adjusted_p_value'] + 1e-300)
        
        # Create Manhattan plot
        fig = go.Figure()
        
        # Add points for each database
        databases = df['database'].unique()
        colors = px.colors.qualitative.Set1[:len(databases)]
        
        for i, database in enumerate(databases):
            db_data = df[df['database'] == database]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(db_data))),
                y=db_data['neg_log10_p'],
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=8,
                    opacity=0.7
                ),
                text=db_data['pathway_name'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Position: %{x}<br>' +
                             '-Log10 P-value: %{y:.2f}<extra></extra>',
                name=database
            ))
        
        # Add significance line
        fig.add_hline(y=-np.log10(parameters.significance_threshold), 
                     line_dash="dash", line_color="red", 
                     annotation_text=f"P = {parameters.significance_threshold}")
        
        fig.update_layout(
            title="Manhattan Plot - Pathway Enrichment",
            xaxis_title="Pathway Index",
            yaxis_title="-Log10 Adjusted P-value",
            height=600
        )
        
        return fig
    
    async def _create_enrichment_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create enrichment plot (similar to GSEA enrichment plot)."""
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for enrichment plot
        df = pd.DataFrame(pathway_data)
        df = df[df['adjusted_p_value'] <= parameters.significance_threshold]
        df = df.head(parameters.max_pathways)
        
        # Create enrichment plot
        fig = go.Figure()
        
        # Add enrichment curves for each database
        databases = df['database'].unique()
        colors = px.colors.qualitative.Set1[:len(databases)]
        
        for i, database in enumerate(databases):
            db_data = df[df['database'] == database]
            
            # Create enrichment curve (simplified)
            x_pos = np.linspace(0, 1, len(db_data))
            y_pos = np.cumsum(db_data['enrichment_score']) / np.sum(db_data['enrichment_score'])
            
            fig.add_trace(go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='lines+markers',
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                name=database
            ))
        
        fig.update_layout(
            title="Enrichment Plot",
            xaxis_title="Rank in Ordered Gene List",
            yaxis_title="Running Enrichment Score",
            height=600
        )
        
        return fig
    
    async def _create_pathway_map(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create pathway map visualization."""
        # This is a simplified pathway map - in practice, would use actual pathway maps
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for pathway map
        df = pd.DataFrame(pathway_data)
        df = df[df['adjusted_p_value'] <= parameters.significance_threshold]
        df = df.head(parameters.max_pathways)
        
        # Create pathway map
        fig = go.Figure()
        
        # Add pathway nodes
        fig.add_trace(go.Scatter(
            x=np.random.uniform(0, 10, len(df)),
            y=np.random.uniform(0, 10, len(df)),
            mode='markers+text',
            marker=dict(
                size=df['overlap_count'] * 3,
                color=df['enrichment_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Enrichment Score")
            ),
            text=df['pathway_name'],
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Overlap: ' + df['overlap_count'].astype(str) + '<br>' +
                         'Enrichment: ' + df['enrichment_score'].round(2).astype(str) + '<extra></extra>',
            name='Pathways'
        ))
        
        fig.update_layout(
            title="Pathway Map",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    async def _create_gene_expression_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create gene expression plot."""
        # This is a simplified gene expression plot
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for gene expression plot
        df = pd.DataFrame(pathway_data)
        df = df[df['adjusted_p_value'] <= parameters.significance_threshold]
        df = df.head(parameters.max_pathways)
        
        # Create gene expression plot
        fig = go.Figure()
        
        # Add expression data for each pathway
        for _, row in df.iterrows():
            # Simulate gene expression data
            genes = row['overlapping_genes'][:10]  # Limit to first 10 genes
            expression = np.random.normal(row['enrichment_score'], 0.5, len(genes))
            
            fig.add_trace(go.Scatter(
                x=genes,
                y=expression,
                mode='markers+lines',
                name=row['pathway_name'],
                hovertemplate='<b>%{x}</b><br>' +
                             'Expression: %{y:.2f}<br>' +
                             'Pathway: ' + row['pathway_name'] + '<extra></extra>'
            ))
        
        fig.update_layout(
            title="Gene Expression Plot",
            xaxis_title="Gene",
            yaxis_title="Expression Level",
            height=600
        )
        
        return fig
    
    async def _create_time_series_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create time series plot."""
        # This is a simplified time series plot
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for time series plot
        df = pd.DataFrame(pathway_data)
        df = df[df['adjusted_p_value'] <= parameters.significance_threshold]
        df = df.head(parameters.max_pathways)
        
        # Create time series plot
        fig = go.Figure()
        
        # Add time series data for each pathway
        time_points = np.linspace(0, 24, 10)  # 10 time points over 24 hours
        
        for _, row in df.iterrows():
            # Simulate time series data
            expression = row['enrichment_score'] * (1 + 0.5 * np.sin(time_points * np.pi / 12))
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=expression,
                mode='lines+markers',
                name=row['pathway_name'],
                hovertemplate='<b>' + row['pathway_name'] + '</b><br>' +
                             'Time: %{x:.1f}h<br>' +
                             'Expression: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Time Series Plot",
            xaxis_title="Time (hours)",
            yaxis_title="Expression Level",
            height=600
        )
        
        return fig
    
    async def _create_comparison_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create comparison plot for multiple datasets."""
        if isinstance(data, ComparisonResult):
            # Create comparison plot
            fig = go.Figure()
            
            # Add comparison data (simplified)
            fig.add_trace(go.Scatter(
                x=[1, 2, 3, 4],
                y=[0.5, 0.7, 0.3, 0.9],
                mode='markers+lines',
                name='Dataset Comparison',
                hovertemplate='<b>Dataset %{x}</b><br>' +
                             'Score: %{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Dataset Comparison",
                xaxis_title="Dataset",
                yaxis_title="Comparison Score",
                height=600
            )
            
            return fig
        else:
            # Fallback to scatter plot
            return await self._create_scatter_plot(data, parameters)
    
    async def _create_consensus_plot(
        self, 
        data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], 
        parameters: VisualizationParameters
    ) -> go.Figure:
        """Create consensus plot for multiple databases."""
        pathway_data = self._extract_pathway_data(data)
        
        if not pathway_data:
            return go.Figure().add_annotation(text="No pathway data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data for consensus plot
        df = pd.DataFrame(pathway_data)
        
        # Create consensus plot
        fig = go.Figure()
        
        # Add consensus data
        fig.add_trace(go.Scatter(
            x=df['enrichment_score'],
            y=df['adjusted_p_value'],
            mode='markers',
            marker=dict(
                size=df['overlap_count'],
                color=df['enrichment_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Enrichment Score")
            ),
            text=df['pathway_name'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Enrichment: %{x:.2f}<br>' +
                         'P-value: %{y:.2e}<extra></extra>',
            name='Consensus Pathways'
        ))
        
        fig.update_layout(
            title="Consensus Plot",
            xaxis_title="Enrichment Score",
            yaxis_title="Adjusted P-value",
            height=600
        )
        
        return fig
    
    def _extract_pathway_data(self, data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract pathway data from analysis or comparison results."""
        pathway_data = []
        
        if isinstance(data, AnalysisResult):
            # Extract from analysis result
            for db_name, db_result in data.database_results.items():
                for pathway in db_result.pathways:
                    pathway_data.append({
                        'pathway_id': pathway.pathway_id,
                        'pathway_name': pathway.pathway_name,
                        'database': db_name,
                        'p_value': pathway.p_value,
                        'adjusted_p_value': pathway.adjusted_p_value,
                        'enrichment_score': pathway.enrichment_score or 0.0,
                        'overlap_count': pathway.overlap_count,
                        'pathway_count': pathway.pathway_count,
                        'overlapping_genes': pathway.overlapping_genes
                    })
        
        elif isinstance(data, ComparisonResult):
            # Extract from comparison result
            for pathway in data.pathway_concordance:
                pathway_data.append({
                    'pathway_id': pathway.pathway_id,
                    'pathway_name': pathway.pathway_name,
                    'database': pathway.database,
                    'p_value': np.mean(list(pathway.p_values.values())),
                    'adjusted_p_value': np.mean(list(pathway.adjusted_p_values.values())),
                    'enrichment_score': np.mean(list(pathway.effect_sizes.values())) if pathway.effect_sizes else 0.0,
                    'overlap_count': pathway.pathway_size,
                    'pathway_count': pathway.pathway_size,
                    'overlapping_genes': []
                })
        
        elif isinstance(data, dict):
            # Extract from dictionary
            if 'pathways' in data:
                pathway_data = data['pathways']
            elif 'database_results' in data:
                for db_name, db_result in data['database_results'].items():
                    for pathway in db_result.get('pathways', []):
                        pathway_data.append({
                            'pathway_id': pathway.get('pathway_id', ''),
                            'pathway_name': pathway.get('pathway_name', ''),
                            'database': db_name,
                            'p_value': pathway.get('p_value', 1.0),
                            'adjusted_p_value': pathway.get('adjusted_p_value', 1.0),
                            'enrichment_score': pathway.get('enrichment_score', 0.0),
                            'overlap_count': pathway.get('overlap_count', 0),
                            'pathway_count': pathway.get('pathway_count', 0),
                            'overlapping_genes': pathway.get('overlapping_genes', [])
                        })
        
        return pathway_data
    
    def _count_data_points(self, data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], plot_type: PlotType) -> int:
        """Count data points for a specific plot type."""
        pathway_data = self._extract_pathway_data(data)
        return len(pathway_data)
    
    def _count_pathways(self, data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]]) -> int:
        """Count total pathways in data."""
        pathway_data = self._extract_pathway_data(data)
        return len(set(p['pathway_id'] for p in pathway_data))
    
    def _count_genes(self, data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]]) -> int:
        """Count total genes in data."""
        pathway_data = self._extract_pathway_data(data)
        all_genes = set()
        for pathway in pathway_data:
            all_genes.update(pathway.get('overlapping_genes', []))
        return len(all_genes)
    
    def _count_significant(self, data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], threshold: float) -> int:
        """Count significant pathways."""
        pathway_data = self._extract_pathway_data(data)
        return sum(1 for pathway in pathway_data if pathway['adjusted_p_value'] <= threshold)
    
    async def _generate_dashboard(
        self,
        generated_plots: Dict[PlotType, str],
        plot_metadata: Dict[PlotType, PlotMetadata],
        parameters: VisualizationParameters,
        output_path: Path
    ) -> str:
        """Generate interactive dashboard."""
        dashboard_file = output_path / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Create simple dashboard HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PathwayLens Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .plot-container {{ margin: 20px 0; }}
                .plot-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>PathwayLens Visualization Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for plot_type, plot_file in generated_plots.items():
            if Path(plot_file).exists():
                html_content += f"""
                <div class="plot-container">
                    <div class="plot-title">{plot_type.value.replace('_', ' ').title()}</div>
                    <iframe src="{Path(plot_file).name}" width="100%" height="600"></iframe>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        return str(dashboard_file)
    
    def _calculate_plot_quality(self, plot_metadata: Dict[PlotType, PlotMetadata]) -> float:
        """Calculate overall plot quality score."""
        if not plot_metadata:
            return 0.0
        
        # Simple quality metric based on data coverage and significance
        quality_scores = []
        for metadata in plot_metadata.values():
            data_coverage = min(metadata.num_data_points / 100, 1.0)  # Normalize to 0-1
            significance_ratio = min(metadata.num_significant / max(metadata.num_pathways, 1), 1.0)
            quality = (data_coverage + significance_ratio) / 2
            quality_scores.append(quality)
        
        return np.mean(quality_scores)
    
    def _calculate_data_coverage(self, data: Union[AnalysisResult, ComparisonResult, Dict[str, Any]], plot_metadata: Dict[PlotType, PlotMetadata]) -> float:
        """Calculate data coverage score."""
        if not plot_metadata:
            return 0.0
        
        # Calculate coverage based on number of data points used
        total_data_points = sum(metadata.num_data_points for metadata in plot_metadata.values())
        max_possible_points = len(plot_metadata) * 1000  # Assume max 1000 points per plot
        
        return min(total_data_points / max_possible_points, 1.0)
