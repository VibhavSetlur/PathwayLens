"""
Visualization functions for comparing analysis results across databases.
"""

from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from loguru import logger

from pathwaylens_core.analysis.schemas import DatabaseResult, PathwayResult
from pathwaylens_core.visualization.plotly_renderer import PlotlyRenderer

class ComparisonVisualizer:
    """Visualizer for comparison results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.renderer = PlotlyRenderer()
        self.logger = logger.bind(module="comparison_visualizer")
        
    def plot_database_comparison(self, results: Dict[str, DatabaseResult]) -> Optional[str]:
        """
        Plot comparison of significant pathways count across databases.
        
        Args:
            results: Dictionary of database results
            
        Returns:
            Path to the generated plot file
        """
        try:
            data = []
            for db_name, result in results.items():
                data.append({
                    "Database": db_name,
                    "Significant Pathways": result.significant_pathways,
                    "Total Pathways": result.total_pathways
                })
                
            df = pd.DataFrame(data)
            
            fig = px.bar(
                df,
                x="Database",
                y="Significant Pathways",
                title="Significant Pathways per Database",
                text="Significant Pathways",
                color="Database"
            )
            
            fig.update_layout(
                xaxis_title="Database",
                yaxis_title="Number of Significant Pathways",
                showlegend=False
            )
            
            output_file = self.output_dir / "database_comparison.html"
            fig.write_html(str(output_file))
            self.logger.info(f"Database comparison plot saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to plot database comparison: {e}")
            return None

    def plot_pathway_overlap(self, results: Dict[str, DatabaseResult]) -> Optional[str]:
        """
        Plot overlap of significant pathways across databases using UpSet plot.
        
        Args:
            results: Dictionary of database results
            
        Returns:
            Path to the generated plot file
        """
        try:
            sets = {}
            for db_name, result in results.items():
                # Use pathway names for overlap as IDs might differ across DBs
                # Ideally we should map them, but names are a good proxy for now
                pathways = [p.pathway_name for p in result.pathways if p.adjusted_p_value < 0.05] # Assuming 0.05 threshold or use result.significant_pathways logic
                sets[db_name] = pathways
                
            output_file = self.output_dir / "pathway_overlap.html"
            self.renderer.create_upset_plot(
                sets=sets,
                title="Pathway Overlap across Databases",
                output_file=str(output_file)
            )
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to plot pathway overlap: {e}")
            return None

    def plot_enrichment_consistency(self, results: Dict[str, DatabaseResult]) -> Optional[str]:
        """
        Plot consistency of enrichment scores across databases.
        Currently supports pairwise comparison if 2 databases, or multi-panel.
        For simplicity, we'll do a scatter plot of P-values for common pathways.
        
        Args:
            results: Dictionary of database results
            
        Returns:
            Path to the generated plot file
        """
        try:
            if len(results) < 2:
                self.logger.warning("Need at least 2 databases for consistency plot")
                return None
                
            # Find common pathways (by name)
            common_pathways = set()
            first = True
            for result in results.values():
                names = {p.pathway_name for p in result.pathways}
                if first:
                    common_pathways = names
                    first = False
                else:
                    common_pathways &= names
            
            if not common_pathways:
                self.logger.warning("No common pathways found for consistency plot")
                return None
                
            # Create dataframe for common pathways
            data = []
            for name in common_pathways:
                row = {"Pathway": name}
                for db_name, result in results.items():
                    # Find pathway object
                    pathway = next((p for p in result.pathways if p.pathway_name == name), None)
                    if pathway:
                        row[f"{db_name} -log10(p-adj)"] = -1 * pd.np.log10(pathway.adjusted_p_value) if pathway.adjusted_p_value > 0 else 50 # Cap at 50
                data.append(row)
                
            df = pd.DataFrame(data)
            
            # If 2 databases, simple scatter
            db_names = list(results.keys())
            if len(db_names) == 2:
                x_col = f"{db_names[0]} -log10(p-adj)"
                y_col = f"{db_names[1]} -log10(p-adj)"
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    hover_data=["Pathway"],
                    title=f"Enrichment Consistency: {db_names[0]} vs {db_names[1]}"
                )
                
                # Add diagonal line
                max_val = max(df[x_col].max(), df[y_col].max())
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=max_val, y1=max_val,
                    line=dict(color="Red", dash="dash")
                )
                
            else:
                # Matrix scatter plot
                cols = [f"{db} -log10(p-adj)" for db in db_names]
                fig = px.scatter_matrix(
                    df,
                    dimensions=cols,
                    hover_data=["Pathway"],
                    title="Enrichment Consistency Matrix"
                )
                
            output_file = self.output_dir / "enrichment_consistency.html"
            fig.write_html(str(output_file))
            self.logger.info(f"Enrichment consistency plot saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to plot enrichment consistency: {e}")
            return None

    def plot_gene_overlap(self, gene_lists: Dict[str, List[str]]) -> Optional[str]:
        """
        Plot overlap of gene lists (or any string lists) using UpSet plot.
        
        Args:
            gene_lists: Dictionary of label -> list of genes
            
        Returns:
            Path to the generated plot file
        """
        try:
            output_file = self.output_dir / "overlap_plot.html"
            self.renderer.create_upset_plot(
                sets=gene_lists,
                title="Overlap Analysis",
                output_file=str(output_file)
            )
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to plot gene overlap: {e}")
            return None
