"""
Engine for comparing multiple datasets or analysis results.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from pathlib import Path
from loguru import logger
import asyncio

from .schemas import (
    AnalysisResult, AnalysisParameters, DatabaseResult, PathwayResult
)
from .engine import AnalysisEngine
from ..visualization.comparison import ComparisonVisualizer

class ComparisonEngine:
    """Engine for comparing datasets."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.logger = logger.bind(module="comparison_engine")
        self.output_dir = output_dir
        self.visualizer = ComparisonVisualizer(output_dir) if output_dir else None
        self.analysis_engine = AnalysisEngine()

    async def compare_genes(
        self, 
        gene_lists: Dict[str, List[str]],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple gene lists (Venn/UpSet).
        
        Creates output structure:
        comparison_output/
        ├── run.json
        ├── gene_comparison/
        │   ├── overlap_matrix.tsv
        │   ├── unique_genes.tsv
        │   └── shared_genes.tsv
        ├── visualizations/
        │   ├── upset_plot.html
        │   └── venn_diagram.html (if ≤3 sets)
        └── summary/
            └── comparison_report.txt
        """
        out_dir = output_dir or self.output_dir
        if not out_dir:
            raise ValueError("Output directory must be specified")
            
        self.logger.info(f"Comparing {len(gene_lists)} gene lists")
        
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        gene_comp_dir = out_path / "gene_comparison"
        gene_comp_dir.mkdir(exist_ok=True)
        
        viz_dir = out_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        summary_dir = out_path / "summary"
        summary_dir.mkdir(exist_ok=True)
        
        # Calculate overlaps and statistics
        all_genes_union = set().union(*gene_lists.values())
        all_genes_intersection = set.intersection(*[set(g) for g in gene_lists.values()]) if gene_lists else set()
        
        stats = {
            "total_genes_per_list": {label: len(genes) for label, genes in gene_lists.items()},
            "union_size": len(all_genes_union),
            "intersection_size": len(all_genes_intersection)
        }
        
        # 1. Generate overlap matrix
        overlap_matrix = []
        labels = list(gene_lists.keys())
        for i, label1 in enumerate(labels):
            row = {"list": label1}
            for label2 in labels:
                overlap = len(set(gene_lists[label1]) & set(gene_lists[label2]))
                row[label2] = overlap
            overlap_matrix.append(row)
        
        df_overlap = pd.DataFrame(overlap_matrix)
        overlap_file = gene_comp_dir / "overlap_matrix.tsv"
        df_overlap.to_csv(overlap_file, sep='\t', index=False)
        
        # 2. Generate unique genes per list
        unique_genes_data = []
        for label, genes in gene_lists.items():
            other_genes = set().union(*[set(g) for l, g in gene_lists.items() if l != label])
            unique = set(genes) - other_genes
            for gene in unique:
                unique_genes_data.append({"list": label, "gene": gene})
        
        if unique_genes_data:
            df_unique = pd.DataFrame(unique_genes_data)
            unique_file = gene_comp_dir / "unique_genes.tsv"
            df_unique.to_csv(unique_file, sep='\t', index=False)
        else:
            unique_file = None # Ensure unique_file is defined even if no unique genes
        
        # 3. Generate shared genes
        if all_genes_intersection:
            shared_data = [{"gene": gene} for gene in all_genes_intersection]
            df_shared = pd.DataFrame(shared_data)
            shared_file = gene_comp_dir / "shared_genes.tsv"
            df_shared.to_csv(shared_file, sep='\t', index=False)
        else:
            shared_file = None # Ensure shared_file is defined even if no shared genes
        
        # 4. Generate visualizations
        plots = {}
        if self.visualizer:
            self.visualizer.output_dir = viz_dir
            
            # UpSet plot
            upset_file = self.visualizer.plot_gene_overlap(gene_lists)
            if upset_file:
                plots['upset'] = upset_file
        
        # 5. Generate summary report
        report_lines = [
            "Gene List Comparison Report",
            "=" * 50,
            f"\nNumber of lists compared: {len(gene_lists)}",
            f"Total unique genes (union): {len(all_genes_union)}",
            f"Shared genes (intersection): {len(all_genes_intersection)}",
            "\nGenes per list:",
        ]
        for label, genes in gene_lists.items():
            report_lines.append(f"  {label}: {len(genes)} genes")
        
        report_file = summary_dir / "comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # 6. Generate run.json
        run_data = {
            "comparison_type": "genes",
            "timestamp": pd.Timestamp.now().isoformat(),
            "lists_compared": list(gene_lists.keys()),
            "statistics": stats,
            "output_files": {
                "overlap_matrix": str(overlap_file),
                "unique_genes": str(unique_file) if unique_genes_data else None,
                "shared_genes": str(shared_file) if all_genes_intersection else None,
                "report": str(report_file)
            }
        }
        
        import json
        json_file = out_path / "run.json"
        with open(json_file, 'w') as f:
            json.dump(run_data, f, indent=2)
        
        return {
            "stats": stats,
            "plots": plots,
            "output_dir": str(out_path)
        }

    async def compare_datasets(
        self,
        datasets: Dict[str, str], # label -> file path
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run analysis on multiple datasets and compare results.
        
        Args:
            datasets: Dictionary of label -> input file path
            parameters: Analysis parameters to use for all datasets
            output_dir: Output directory
            
        Returns:
            Combined analysis results and comparison stats
        """
        out_dir = output_dir or self.output_dir
        if not out_dir:
            raise ValueError("Output directory must be specified")
            
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Run analysis for each dataset
        for label, file_path in datasets.items():
            self.logger.info(f"Running analysis for {label} ({file_path})")
            
            # Create sub-directory for this dataset
            sub_dir = out_path / label
            
            # Run analysis
            # We use analyze_sync for simplicity here, or await analyze
            result = await self.analysis_engine.analyze(
                input_data=file_path,
                parameters=parameters,
                output_dir=str(sub_dir)
            )
            
            results[label] = result
            
        # Compare results
        comparison_results = await self.compare_pathways(results, out_dir)
        
        return {
            "individual_results": results,
            "comparison": comparison_results
        }

    async def compare_pathways(
        self,
        results: Dict[str, AnalysisResult],
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare pathway analysis results.
        
        Args:
            results: Dictionary of label -> AnalysisResult
            output_dir: Output directory
            
        Returns:
            Comparison stats
        """
        out_dir = output_dir or self.output_dir
        if not out_dir:
            raise ValueError("Output directory must be specified")
            
        self.logger.info("Comparing pathway results")
        
        # Extract significant pathways for each dataset
        sig_pathways = {}
        for label, result in results.items():
            # Aggregate pathways from all databases in the result
            # Or we can compare per database. Let's aggregate for high-level view first.
            # Actually, comparing per database is more rigorous.
            # Let's assume we compare the first database or aggregate unique pathway names.
            
            all_sig_pathways = set()
            for db_res in result.database_results.values():
                for p in db_res.pathways:
                    if p.adjusted_p_value < result.parameters.significance_threshold:
                        all_sig_pathways.add(p.pathway_name)
            
            sig_pathways[label] = list(all_sig_pathways)
            
        # Generate plots
        plots = {}
        if self.visualizer:
            self.visualizer.output_dir = Path(out_dir)
            
            # UpSet of significant pathways
            upset_file = self.visualizer.plot_gene_overlap(sig_pathways) # Reusing gene overlap plot for pathway names
            plots["pathway_overlap"] = upset_file
            
            # Consistency plot (if applicable)
            # We need to extract a common metric (e.g. p-value) for common pathways
            # This requires more complex data structure extraction.
            # For now, let's stick to overlap.
            
        return {
            "significant_pathways_counts": {l: len(p) for l, p in sig_pathways.items()},
            "plots": plots
        }
