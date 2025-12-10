"""
Report generator for creating comprehensive analysis reports.
"""

import os
import json
import base64
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from .export_manager import ExportManager
from ..analysis.schemas import AnalysisResult
from ..comparison.schemas import ComparisonResult


class ReportGenerator:
    """Generator for comprehensive analysis reports with interactive visualizations."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Base output directory for reports
        """
        self.logger = logger.bind(module="report_generator")
        self.output_dir = output_dir
        self.export_manager = ExportManager(output_dir)
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"Report directory ensured: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create report directory: {e}")
    
    def generate_analysis_report(
        self,
        analysis_result: AnalysisResult,
        job_metadata: Dict[str, Any],
        include_interactive: bool = True,
        include_static: bool = True,
        theme: str = "light"
    ) -> Dict[str, str]:
        """
        Generate a comprehensive analysis report.
        
        Args:
            analysis_result: Analysis results
            job_metadata: Job metadata including parameters and timestamps
            include_interactive: Whether to include interactive visualizations
            include_static: Whether to include static visualizations
            theme: Report theme ('light', 'dark', 'scientific')
            
        Returns:
            Dictionary mapping report types to file paths
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = job_metadata.get("job_id", "unknown")
            report_base = f"analysis_report_{job_id}_{timestamp}"
            
            # Create report directory
            report_dir = os.path.join(self.output_dir, report_base)
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate visualizations
            visualizations = self._generate_analysis_visualizations(
                analysis_result, report_dir, include_interactive, include_static
            )
            
            # Generate HTML report
            html_report_path = self._generate_html_report(
                analysis_result, job_metadata, visualizations, report_dir, theme
            )
            
            # Generate data exports
            data_exports = self._export_analysis_data(analysis_result, report_dir)
            
            # Generate summary
            summary_path = self._generate_report_summary(
                analysis_result, job_metadata, visualizations, data_exports, report_dir
            )
            
            # Create ZIP archive
            zip_path = self._create_report_archive(report_dir, report_base)
            
            return {
                "html_report": html_report_path,
                "summary": summary_path,
                "archive": zip_path,
                "visualizations": visualizations,
                "data_exports": data_exports
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis report: {e}")
            return {}
    
    def generate_comparison_report(
        self,
        comparison_result: ComparisonResult,
        job_metadata: Dict[str, Any],
        include_interactive: bool = True,
        include_static: bool = True,
        theme: str = "light"
    ) -> Dict[str, str]:
        """
        Generate a comprehensive comparison report.
        
        Args:
            comparison_result: Comparison results
            job_metadata: Job metadata including parameters and timestamps
            include_interactive: Whether to include interactive visualizations
            include_static: Whether to include static visualizations
            theme: Report theme ('light', 'dark', 'scientific')
            
        Returns:
            Dictionary mapping report types to file paths
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = job_metadata.get("job_id", "unknown")
            report_base = f"comparison_report_{job_id}_{timestamp}"
            
            # Create report directory
            report_dir = os.path.join(self.output_dir, report_base)
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate visualizations
            visualizations = self._generate_comparison_visualizations(
                comparison_result, report_dir, include_interactive, include_static
            )
            
            # Generate HTML report
            html_report_path = self._generate_comparison_html_report(
                comparison_result, job_metadata, visualizations, report_dir, theme
            )
            
            # Generate data exports
            data_exports = self._export_comparison_data(comparison_result, report_dir)
            
            # Generate summary
            summary_path = self._generate_comparison_summary(
                comparison_result, job_metadata, visualizations, data_exports, report_dir
            )
            
            # Create ZIP archive
            zip_path = self._create_report_archive(report_dir, report_base)
            
            return {
                "html_report": html_report_path,
                "summary": summary_path,
                "archive": zip_path,
                "visualizations": visualizations,
                "data_exports": data_exports
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison report: {e}")
            return {}
    
    def _generate_analysis_visualizations(
        self,
        analysis_result: AnalysisResult,
        report_dir: str,
        include_interactive: bool,
        include_static: bool
    ) -> Dict[str, str]:
        """Generate visualizations for analysis results."""
        try:
            visualizations = {}
            
            # Generate dot plot
            if analysis_result.database_results:
                dot_plot = self._create_dot_plot(analysis_result)
                if dot_plot:
                    if include_interactive:
                        html_path = self.export_manager.export_plotly_figure(
                            dot_plot, os.path.join(report_dir, "dot_plot"), "html"
                        )
                        if html_path:
                            visualizations["dot_plot_interactive"] = html_path
                    
                    if include_static:
                        png_path = self.export_manager.export_plotly_figure(
                            dot_plot, os.path.join(report_dir, "dot_plot"), "png"
                        )
                        if png_path:
                            visualizations["dot_plot_static"] = png_path
            
            # Generate volcano plot if available
            if hasattr(analysis_result, 'volcano_data') and analysis_result.volcano_data:
                volcano_plot = self._create_volcano_plot(analysis_result)
                if volcano_plot:
                    if include_interactive:
                        html_path = self.export_manager.export_plotly_figure(
                            volcano_plot, os.path.join(report_dir, "volcano_plot"), "html"
                        )
                        if html_path:
                            visualizations["volcano_plot_interactive"] = html_path
                    
                    if include_static:
                        png_path = self.export_manager.export_plotly_figure(
                            volcano_plot, os.path.join(report_dir, "volcano_plot"), "png"
                        )
                        if png_path:
                            visualizations["volcano_plot_static"] = png_path
            
            # Generate network plot
            if analysis_result.consensus_results:
                network_plot = self._create_network_plot(analysis_result)
                if network_plot:
                    if include_interactive:
                        html_path = self.export_manager.export_plotly_figure(
                            network_plot, os.path.join(report_dir, "network_plot"), "html"
                        )
                        if html_path:
                            visualizations["network_plot_interactive"] = html_path
                    
                    if include_static:
                        png_path = self.export_manager.export_plotly_figure(
                            network_plot, os.path.join(report_dir, "network_plot"), "png"
                        )
                        if png_path:
                            visualizations["network_plot_static"] = png_path
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis visualizations: {e}")
            return {}
    
    def _generate_comparison_visualizations(
        self,
        comparison_result: ComparisonResult,
        report_dir: str,
        include_interactive: bool,
        include_static: bool
    ) -> Dict[str, str]:
        """Generate visualizations for comparison results."""
        try:
            visualizations = {}
            
            # Generate correlation plot
            if comparison_result.correlation_results:
                corr_plot = self._create_correlation_plot(comparison_result)
                if corr_plot:
                    if include_interactive:
                        html_path = self.export_manager.export_plotly_figure(
                            corr_plot, os.path.join(report_dir, "correlation_plot"), "html"
                        )
                        if html_path:
                            visualizations["correlation_plot_interactive"] = html_path
                    
                    if include_static:
                        png_path = self.export_manager.export_plotly_figure(
                            corr_plot, os.path.join(report_dir, "correlation_plot"), "png"
                        )
                        if png_path:
                            visualizations["correlation_plot_static"] = png_path
            
            # Generate overlap plot
            if comparison_result.overlap_statistics:
                overlap_plot = self._create_overlap_plot(comparison_result)
                if overlap_plot:
                    if include_interactive:
                        html_path = self.export_manager.export_plotly_figure(
                            overlap_plot, os.path.join(report_dir, "overlap_plot"), "html"
                        )
                        if html_path:
                            visualizations["overlap_plot_interactive"] = html_path
                    
                    if include_static:
                        png_path = self.export_manager.export_plotly_figure(
                            overlap_plot, os.path.join(report_dir, "overlap_plot"), "png"
                        )
                        if png_path:
                            visualizations["overlap_plot_static"] = png_path
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison visualizations: {e}")
            return {}
    
    def _create_dot_plot(self, analysis_result: AnalysisResult) -> Optional[go.Figure]:
        """Create a dot plot for pathway enrichment results."""
        try:
            # Extract pathway data
            pathways = []
            for db_name, db_results in analysis_result.database_results.items():
                for pathway in db_results:
                    pathways.append({
                        'pathway_name': pathway.pathway_name,
                        'database': db_name,
                        'p_value': pathway.p_value,
                        'adjusted_p_value': pathway.adjusted_p_value,
                        'gene_ratio': pathway.gene_ratio,
                        'overlap_count': pathway.overlap_count
                    })
            
            if not pathways:
                return None
            
            df = pd.DataFrame(pathways)
            
            # Create dot plot
            fig = go.Figure()
            
            for db in df['database'].unique():
                db_data = df[df['database'] == db]
                fig.add_trace(go.Scatter(
                    x=db_data['gene_ratio'],
                    y=-np.log10(db_data['adjusted_p_value']),
                    mode='markers',
                    name=db,
                    text=db_data['pathway_name'],
                    hovertemplate='<b>%{text}</b><br>' +
                                'Gene Ratio: %{x}<br>' +
                                '-log10(p-value): %{y}<br>' +
                                '<extra></extra>',
                    marker=dict(
                        size=db_data['overlap_count'],
                        sizemode='diameter',
                        sizeref=2.*max(df['overlap_count'])/(40.**2),
                        sizemin=4
                    )
                ))
            
            fig.update_layout(
                title="Pathway Enrichment Dot Plot",
                xaxis_title="Gene Ratio",
                yaxis_title="-log10(Adjusted P-value)",
                hovermode='closest',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create dot plot: {e}")
            return None
    
    def _create_volcano_plot(self, analysis_result: AnalysisResult) -> Optional[go.Figure]:
        """Create a volcano plot if differential expression data is available."""
        try:
            # This would require differential expression data
            # For now, return None as it's not always available
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create volcano plot: {e}")
            return None
    
    def _create_network_plot(self, analysis_result: AnalysisResult) -> Optional[go.Figure]:
        """Create a network plot showing pathway relationships."""
        try:
            # This would require pathway interaction data
            # For now, return None as it's not always available
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create network plot: {e}")
            return None
    
    def _create_correlation_plot(self, comparison_result: ComparisonResult) -> Optional[go.Figure]:
        """Create a correlation plot for comparison results."""
        try:
            # Extract correlation data
            corr_data = comparison_result.correlation_results
            
            if not corr_data:
                return None
            
            # Create correlation heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.get('correlation_matrix', []),
                x=corr_data.get('dataset_names', []),
                y=corr_data.get('dataset_names', []),
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="Dataset Correlation Heatmap",
                xaxis_title="Datasets",
                yaxis_title="Datasets"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create correlation plot: {e}")
            return None
    
    def _create_overlap_plot(self, comparison_result: ComparisonResult) -> Optional[go.Figure]:
        """Create an overlap plot for comparison results."""
        try:
            # Extract overlap data
            overlap_data = comparison_result.overlap_statistics
            
            if not overlap_data:
                return None
            
            # Create Venn diagram or UpSet plot
            # For now, create a simple bar chart
            datasets = list(overlap_data.keys())
            overlap_counts = [overlap_data[ds].get('overlap_count', 0) for ds in datasets]
            
            fig = go.Figure(data=go.Bar(
                x=datasets,
                y=overlap_counts,
                text=overlap_counts,
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Pathway Overlap Between Datasets",
                xaxis_title="Datasets",
                yaxis_title="Number of Overlapping Pathways"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create overlap plot: {e}")
            return None
    
    def _generate_html_report(
        self,
        analysis_result: AnalysisResult,
        job_metadata: Dict[str, Any],
        visualizations: Dict[str, str],
        report_dir: str,
        theme: str
    ) -> str:
        """Generate HTML report with embedded visualizations."""
        try:
            html_content = self._create_html_template(
                analysis_result, job_metadata, visualizations, theme
            )
            
            html_path = os.path.join(report_dir, "analysis_report.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {html_path}")
            return html_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return ""
    
    def _generate_comparison_html_report(
        self,
        comparison_result: ComparisonResult,
        job_metadata: Dict[str, Any],
        visualizations: Dict[str, str],
        report_dir: str,
        theme: str
    ) -> str:
        """Generate HTML report for comparison results."""
        try:
            html_content = self._create_comparison_html_template(
                comparison_result, job_metadata, visualizations, theme
            )
            
            html_path = os.path.join(report_dir, "comparison_report.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Comparison HTML report generated: {html_path}")
            return html_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison HTML report: {e}")
            return ""
    
    def _create_html_template(
        self,
        analysis_result: AnalysisResult,
        job_metadata: Dict[str, Any],
        visualizations: Dict[str, str],
        theme: str
    ) -> str:
        """Create HTML template for analysis report."""
        try:
            # Get theme CSS
            theme_css = self._get_theme_css(theme)
            
            # Create summary statistics
            total_pathways = sum(len(db_results) for db_results in analysis_result.database_results.values())
            significant_pathways = sum(
                len([p for p in db_results if p.adjusted_p_value < 0.05])
                for db_results in analysis_result.database_results.values()
            )
            
            # Embed visualizations
            viz_embeds = ""
            for viz_name, viz_path in visualizations.items():
                if viz_path.endswith('.html'):
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        viz_content = f.read()
                    viz_embeds += f'<div class="visualization" id="{viz_name}">{viz_content}</div>\n'
            
            html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PathwayLens Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        {theme_css}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: var(--bg-color);
            color: var(--text-color);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: var(--header-bg);
            border-radius: 8px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: var(--primary-color);
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-color);
        }}
        .visualization {{
            margin: 30px 0;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        .metadata h3 {{
            color: var(--primary-color);
            margin-top: 0;
        }}
        .metadata-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metadata-table th,
        .metadata-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        .metadata-table th {{
            background: var(--header-bg);
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 PathwayLens Analysis Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="summary">
        <div class="summary-card">
            <h3>Total Pathways</h3>
            <div class="value">{total_pathways}</div>
        </div>
        <div class="summary-card">
            <h3>Significant Pathways</h3>
            <div class="value">{significant_pathways}</div>
        </div>
        <div class="summary-card">
            <h3>Input Genes</h3>
            <div class="value">{analysis_result.input_gene_count or 'N/A'}</div>
        </div>
        <div class="summary-card">
            <h3>Species</h3>
            <div class="value">{analysis_result.species}</div>
        </div>
    </div>
    
    {viz_embeds}
    
    <div class="metadata">
        <h3>Analysis Metadata</h3>
        <table class="metadata-table">
            <tr><th>Job ID</th><td>{job_metadata.get('job_id', 'N/A')}</td></tr>
            <tr><th>Analysis Type</th><td>{analysis_result.analysis_type}</td></tr>
            <tr><th>Species</th><td>{analysis_result.species}</td></tr>
            <tr><th>Created At</th><td>{job_metadata.get('created_at', 'N/A')}</td></tr>
            <tr><th>Completed At</th><td>{job_metadata.get('completed_at', 'N/A')}</td></tr>
        </table>
    </div>
</body>
</html>
            """
            
            return html_template
            
        except Exception as e:
            self.logger.error(f"Failed to create HTML template: {e}")
            return ""
    
    def _create_comparison_html_template(
        self,
        comparison_result: ComparisonResult,
        job_metadata: Dict[str, Any],
        visualizations: Dict[str, str],
        theme: str
    ) -> str:
        """Create HTML template for comparison report."""
        try:
            # Similar to analysis template but for comparison results
            theme_css = self._get_theme_css(theme)
            
            # Embed visualizations
            viz_embeds = ""
            for viz_name, viz_path in visualizations.items():
                if viz_path.endswith('.html'):
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        viz_content = f.read()
                    viz_embeds += f'<div class="visualization" id="{viz_name}">{viz_content}</div>\n'
            
            html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PathwayLens Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        {theme_css}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: var(--bg-color);
            color: var(--text-color);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: var(--header-bg);
            border-radius: 8px;
        }}
        .visualization {{
            margin: 30px 0;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        .metadata h3 {{
            color: var(--primary-color);
            margin-top: 0;
        }}
        .metadata-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metadata-table th,
        .metadata-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        .metadata-table th {{
            background: var(--header-bg);
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 PathwayLens Comparison Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    {viz_embeds}
    
    <div class="metadata">
        <h3>Comparison Metadata</h3>
        <table class="metadata-table">
            <tr><th>Job ID</th><td>{job_metadata.get('job_id', 'N/A')}</td></tr>
            <tr><th>Comparison Type</th><td>{comparison_result.comparison_type}</td></tr>
            <tr><th>Created At</th><td>{job_metadata.get('created_at', 'N/A')}</td></tr>
            <tr><th>Completed At</th><td>{job_metadata.get('completed_at', 'N/A')}</td></tr>
        </table>
    </div>
</body>
</html>
            """
            
            return html_template
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison HTML template: {e}")
            return ""
    
    def _get_theme_css(self, theme: str) -> str:
        """Get CSS for the specified theme."""
        themes = {
            "light": """
                :root {
                    --bg-color: #ffffff;
                    --text-color: #333333;
                    --header-bg: #f8f9fa;
                    --card-bg: #ffffff;
                    --primary-color: #007bff;
                    --accent-color: #28a745;
                    --border-color: #dee2e6;
                }
            """,
            "dark": """
                :root {
                    --bg-color: #1a1a1a;
                    --text-color: #ffffff;
                    --header-bg: #2d2d2d;
                    --card-bg: #2d2d2d;
                    --primary-color: #4dabf7;
                    --accent-color: #51cf66;
                    --border-color: #495057;
                }
            """,
            "scientific": """
                :root {
                    --bg-color: #f8f9fa;
                    --text-color: #212529;
                    --header-bg: #e9ecef;
                    --card-bg: #ffffff;
                    --primary-color: #6f42c1;
                    --accent-color: #dc3545;
                    --border-color: #ced4da;
                }
            """
        }
        
        return themes.get(theme, themes["light"])
    
    def _export_analysis_data(self, analysis_result: AnalysisResult, report_dir: str) -> Dict[str, str]:
        """Export analysis data in various formats."""
        try:
            data_exports = {}
            
            # Export pathway results
            if analysis_result.database_results:
                for db_name, db_results in analysis_result.database_results.items():
                    df = pd.DataFrame([
                        {
                            'pathway_id': p.pathway_id,
                            'pathway_name': p.pathway_name,
                            'p_value': p.p_value,
                            'adjusted_p_value': p.adjusted_p_value,
                            'gene_ratio': p.gene_ratio,
                            'overlap_count': p.overlap_count
                        }
                        for p in db_results
                    ])
                    
                    csv_path = self.export_manager.export_dataframe(
                        df, os.path.join(report_dir, f"pathways_{db_name}"), "csv"
                    )
                    if csv_path:
                        data_exports[f"pathways_{db_name}"] = csv_path
            
            # Export consensus results
            if analysis_result.consensus_results:
                df = pd.DataFrame([
                    {
                        'pathway_id': p.pathway_id,
                        'pathway_name': p.pathway_name,
                        'consensus_score': p.consensus_score,
                        'databases': ', '.join(p.databases)
                    }
                    for p in analysis_result.consensus_results
                ])
                
                csv_path = self.export_manager.export_dataframe(
                    df, os.path.join(report_dir, "consensus_results"), "csv"
                )
                if csv_path:
                    data_exports["consensus_results"] = csv_path
            
            return data_exports
            
        except Exception as e:
            self.logger.error(f"Failed to export analysis data: {e}")
            return {}
    
    def _export_comparison_data(self, comparison_result: ComparisonResult, report_dir: str) -> Dict[str, str]:
        """Export comparison data in various formats."""
        try:
            data_exports = {}
            
            # Export overlap statistics
            if comparison_result.overlap_statistics:
                overlap_data = []
                for dataset, stats in comparison_result.overlap_statistics.items():
                    overlap_data.append({
                        'dataset': dataset,
                        'overlap_count': stats.get('overlap_count', 0),
                        'total_pathways': stats.get('total_pathways', 0),
                        'overlap_percentage': stats.get('overlap_percentage', 0)
                    })
                
                df = pd.DataFrame(overlap_data)
                csv_path = self.export_manager.export_dataframe(
                    df, os.path.join(report_dir, "overlap_statistics"), "csv"
                )
                if csv_path:
                    data_exports["overlap_statistics"] = csv_path
            
            return data_exports
            
        except Exception as e:
            self.logger.error(f"Failed to export comparison data: {e}")
            return {}
    
    def _generate_report_summary(
        self,
        analysis_result: AnalysisResult,
        job_metadata: Dict[str, Any],
        visualizations: Dict[str, str],
        data_exports: Dict[str, str],
        report_dir: str
    ) -> str:
        """Generate a summary of the report contents."""
        try:
            summary = {
                "report_type": "analysis",
                "generated_at": datetime.now().isoformat(),
                "job_metadata": job_metadata,
                "analysis_summary": {
                    "total_pathways": sum(len(db_results) for db_results in analysis_result.database_results.values()),
                    "significant_pathways": sum(
                        len([p for p in db_results if p.adjusted_p_value < 0.05])
                        for db_results in analysis_result.database_results.values()
                    ),
                    "input_gene_count": analysis_result.input_gene_count,
                    "species": analysis_result.species
                },
                "visualizations": list(visualizations.keys()),
                "data_exports": list(data_exports.keys()),
                "files": {
                    "visualizations": visualizations,
                    "data_exports": data_exports
                }
            }
            
            summary_path = os.path.join(report_dir, "report_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Report summary generated: {summary_path}")
            return summary_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate report summary: {e}")
            return ""
    
    def _generate_comparison_summary(
        self,
        comparison_result: ComparisonResult,
        job_metadata: Dict[str, Any],
        visualizations: Dict[str, str],
        data_exports: Dict[str, str],
        report_dir: str
    ) -> str:
        """Generate a summary of the comparison report contents."""
        try:
            summary = {
                "report_type": "comparison",
                "generated_at": datetime.now().isoformat(),
                "job_metadata": job_metadata,
                "comparison_summary": {
                    "comparison_type": comparison_result.comparison_type,
                    "datasets_compared": len(comparison_result.input_analysis_ids) if comparison_result.input_analysis_ids else 0
                },
                "visualizations": list(visualizations.keys()),
                "data_exports": list(data_exports.keys()),
                "files": {
                    "visualizations": visualizations,
                    "data_exports": data_exports
                }
            }
            
            summary_path = os.path.join(report_dir, "comparison_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Comparison report summary generated: {summary_path}")
            return summary_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison summary: {e}")
            return ""
    
    def _create_report_archive(self, report_dir: str, report_base: str) -> str:
        """Create a ZIP archive of the report."""
        try:
            import zipfile
            
            zip_path = os.path.join(self.output_dir, f"{report_base}.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(report_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, report_dir)
                        zipf.write(file_path, arcname)
            
            self.logger.info(f"Report archive created: {zip_path}")
            return zip_path
            
        except Exception as e:
            self.logger.error(f"Failed to create report archive: {e}")
            return ""
