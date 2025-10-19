"""
Export manager for visualization outputs.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from loguru import logger
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class ExportManager:
    """Manager for exporting visualization outputs in various formats."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the export manager.
        
        Args:
            output_dir: Base output directory
        """
        self.logger = logger.bind(module="export_manager")
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"Output directory ensured: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
    
    def export_plotly_figure(
        self,
        fig: go.Figure,
        filename: str,
        format: str = "html",
        include_plotlyjs: bool = True
    ) -> str:
        """
        Export Plotly figure to file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Export format ('html', 'json', 'png', 'pdf', 'svg')
            include_plotlyjs: Whether to include Plotly.js library (HTML only)
            
        Returns:
            Path to exported file
        """
        try:
            # Ensure filename has correct extension
            if not filename.endswith(f'.{format}'):
                filename = f"{filename}.{format}"
            
            output_path = os.path.join(self.output_dir, filename)
            
            if format == "html":
                fig.write_html(output_path, include_plotlyjs=include_plotlyjs)
            elif format == "json":
                fig.write_json(output_path)
            elif format == "png":
                fig.write_image(output_path)
            elif format == "pdf":
                fig.write_image(output_path, format="pdf")
            elif format == "svg":
                fig.write_image(output_path, format="svg")
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Plotly figure exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export Plotly figure: {e}")
            return ""
    
    def export_matplotlib_figure(
        self,
        fig: plt.Figure,
        filename: str,
        format: str = "png",
        dpi: int = 300
    ) -> str:
        """
        Export matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure object
            filename: Output filename
            format: Export format ('png', 'pdf', 'svg', 'eps')
            dpi: DPI for raster formats
            
        Returns:
            Path to exported file
        """
        try:
            # Ensure filename has correct extension
            if not filename.endswith(f'.{format}'):
                filename = f"{filename}.{format}"
            
            output_path = os.path.join(self.output_dir, filename)
            
            fig.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
            
            self.logger.info(f"Matplotlib figure exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export matplotlib figure: {e}")
            return ""
    
    def export_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = "csv",
        index: bool = False
    ) -> str:
        """
        Export DataFrame to file.
        
        Args:
            df: DataFrame to export
            filename: Output filename
            format: Export format ('csv', 'excel', 'json', 'tsv')
            index: Whether to include index
            
        Returns:
            Path to exported file
        """
        try:
            # Ensure filename has correct extension
            if not filename.endswith(f'.{format}'):
                filename = f"{filename}.{format}"
            
            output_path = os.path.join(self.output_dir, filename)
            
            if format == "csv":
                df.to_csv(output_path, index=index)
            elif format == "excel":
                df.to_excel(output_path, index=index)
            elif format == "json":
                df.to_json(output_path, orient='records', indent=2)
            elif format == "tsv":
                df.to_csv(output_path, sep='\t', index=index)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"DataFrame exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export DataFrame: {e}")
            return ""
    
    def export_analysis_results(
        self,
        results: Dict[str, Any],
        filename: str,
        format: str = "json"
    ) -> str:
        """
        Export analysis results to file.
        
        Args:
            results: Analysis results dictionary
            filename: Output filename
            format: Export format ('json', 'yaml')
            
        Returns:
            Path to exported file
        """
        try:
            # Ensure filename has correct extension
            if not filename.endswith(f'.{format}'):
                filename = f"{filename}.{format}"
            
            output_path = os.path.join(self.output_dir, filename)
            
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format == "yaml":
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Analysis results exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export analysis results: {e}")
            return ""
    
    def export_network_data(
        self,
        network_data: Dict[str, Any],
        filename: str,
        format: str = "json"
    ) -> str:
        """
        Export network data to file.
        
        Args:
            network_data: Network data dictionary
            filename: Output filename
            format: Export format ('json', 'graphml', 'gexf')
            
        Returns:
            Path to exported file
        """
        try:
            # Ensure filename has correct extension
            if not filename.endswith(f'.{format}'):
                filename = f"{filename}.{format}"
            
            output_path = os.path.join(self.output_dir, filename)
            
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(network_data, f, indent=2, default=str)
            elif format == "graphml":
                import networkx as nx
                G = network_data.get("graph")
                if G:
                    nx.write_graphml(G, output_path)
            elif format == "gexf":
                import networkx as nx
                G = network_data.get("graph")
                if G:
                    nx.write_gexf(G, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Network data exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export network data: {e}")
            return ""
    
    def export_visualization_batch(
        self,
        visualizations: Dict[str, Any],
        base_filename: str,
        formats: List[str] = ["html", "png"]
    ) -> Dict[str, str]:
        """
        Export multiple visualizations in batch.
        
        Args:
            visualizations: Dictionary of visualization names to objects
            base_filename: Base filename for outputs
            formats: List of export formats
            
        Returns:
            Dictionary mapping visualization names to exported file paths
        """
        try:
            exported_files = {}
            
            for viz_name, viz_obj in visualizations.items():
                viz_files = {}
                
                for format in formats:
                    filename = f"{base_filename}_{viz_name}"
                    
                    if isinstance(viz_obj, go.Figure):
                        file_path = self.export_plotly_figure(viz_obj, filename, format)
                    elif isinstance(viz_obj, plt.Figure):
                        file_path = self.export_matplotlib_figure(viz_obj, filename, format)
                    else:
                        self.logger.warning(f"Unsupported visualization type for {viz_name}")
                        continue
                    
                    if file_path:
                        viz_files[format] = file_path
                
                exported_files[viz_name] = viz_files
            
            self.logger.info(f"Batch export completed: {len(exported_files)} visualizations")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Failed to export visualization batch: {e}")
            return {}
    
    def export_dashboard(
        self,
        dashboard: go.Figure,
        filename: str,
        formats: List[str] = ["html", "png"]
    ) -> Dict[str, str]:
        """
        Export dashboard in multiple formats.
        
        Args:
            dashboard: Dashboard figure object
            filename: Output filename
            formats: List of export formats
            
        Returns:
            Dictionary mapping formats to exported file paths
        """
        try:
            exported_files = {}
            
            for format in formats:
                file_path = self.export_plotly_figure(dashboard, filename, format)
                if file_path:
                    exported_files[format] = file_path
            
            self.logger.info(f"Dashboard exported in {len(exported_files)} formats")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Failed to export dashboard: {e}")
            return {}
    
    def create_export_summary(
        self,
        exported_files: Dict[str, Any],
        filename: str = "export_summary.json"
    ) -> str:
        """
        Create a summary of exported files.
        
        Args:
            exported_files: Dictionary of exported files
            filename: Summary filename
            
        Returns:
            Path to summary file
        """
        try:
            summary = {
                "export_timestamp": pd.Timestamp.now().isoformat(),
                "output_directory": self.output_dir,
                "exported_files": exported_files,
                "total_files": sum(len(files) if isinstance(files, dict) else 1 for files in exported_files.values())
            }
            
            output_path = os.path.join(self.output_dir, filename)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Export summary created: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to create export summary: {e}")
            return ""
    
    def cleanup_old_files(
        self,
        max_age_days: int = 30,
        pattern: str = "*"
    ) -> int:
        """
        Clean up old files in output directory.
        
        Args:
            max_age_days: Maximum age of files to keep (in days)
            pattern: File pattern to match
            
        Returns:
            Number of files cleaned up
        """
        try:
            import glob
            import time
            
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            pattern_path = os.path.join(self.output_dir, pattern)
            files = glob.glob(pattern_path)
            
            cleaned_count = 0
            
            for file_path in files:
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old files")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files: {e}")
            return 0
    
    def get_output_directory(self) -> str:
        """
        Get the output directory path.
        
        Returns:
            Output directory path
        """
        return self.output_dir
    
    def set_output_directory(self, output_dir: str):
        """
        Set the output directory path.
        
        Args:
            output_dir: New output directory path
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        self.logger.info(f"Output directory set to: {output_dir}")
    
    def list_exported_files(self) -> List[str]:
        """
        List all exported files in the output directory.
        
        Returns:
            List of file paths
        """
        try:
            files = []
            for root, dirs, filenames in os.walk(self.output_dir):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
            
            self.logger.info(f"Found {len(files)} exported files")
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to list exported files: {e}")
            return []
