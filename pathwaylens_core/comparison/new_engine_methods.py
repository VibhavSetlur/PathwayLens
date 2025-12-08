"""
Comparison engine methods for stage-based comparisons with labels.
These methods will be integrated into ComparisonEngine class in engine.py
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


async def compare_gene_lists(
    self,
    gene_lists: Dict[str, List[str]],
    parameters,  # ComparisonParameters
    output_dir: str
) -> Dict[str, Any]:
    """
    Compare gene lists at the gene-stage with labels.
    
    Args:
        gene_lists: Dict mapping labels to lists of genes
        parameters: Comparison parameters with labels  
        output_dir: Output directory for results
        
    Returns:
        Comparison results with visualizations
    """
    self.logger.info(f"Comparing {len(gene_lists)} labeled gene lists")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate overlap statistics for all pairs
    overlap_stats = {}
    labels = list(gene_lists.keys())
    
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            genes1 = set(gene_lists[label1])
            genes2 = set(gene_lists[label2])
            
            # Calculate overlap
            overlap = genes1 & genes2
            union = genes1 | genes2
            jaccard = len(overlap) / len(union) if union else 0.0
            
            overlap_stats[f"{label1}_vs_{label2}"] = {
                "label1": label1,
                "label2": label2,
                "genes1_count": len(genes1),
                "genes2_count": len(genes2),
                "overlap_count": len(overlap),
                "jaccard_index": jaccard,
                "overlapping_genes": list(overlap),
                "unique_to_label1": list(genes1 - genes2),
                "unique_to_label2": list(genes2 - genes1)
            }
    
    # Create visualizations
    plots = {}
    
    # Generate Venn diagram for 2-3 way comparisons
    if len(gene_lists) in [2, 3]:
        try:
            from matplotlib_venn import venn2, venn3
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if len(gene_lists) == 2:
                venn2([set(gene_lists[labels[0]]), set(gene_lists[labels[1]])],
                      set_labels=labels, ax=ax)
            else:
                venn3([set(gene_lists[labels[0]]), set(gene_lists[labels[1]]), set(gene_lists[labels[2]])],
                      set_labels=labels, ax=ax)
            
            plt.title("Gene List Overlap")
            venn_path = Path(output_dir) / "gene_overlap_venn.png"
            plt.savefig(venn_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['venn'] = str(venn_path)
        except ImportError:
            self.logger.warning("matplotlib-venn not installed - skipping Venn diagram")
    
    # Generate UpSet plot for 2+ way comparisons
    if len(gene_lists) >= 2:
        try:
            from upsetplot import from_contents, UpSet
            
            # Build membership data - from_contents expects dict of sets
            upset_data = from_contents({label: set(genes) for label, genes in gene_lists.items()})
            
            upset = UpSet(upset_data, subset_size='count', show_counts=True)
            upset.plot()
            
            upset_path = Path(output_dir) / "gene_overlap_upset.png"
            plt.savefig(upset_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['upset'] = str(upset_path)
        except ImportError:
            self.logger.warning("upsetplot not installed - skipping UpSet plot")
    
    # Save results
    results_path = Path(output_dir) / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'comparison_stage': 'gene',
            'labels': labels,
            'gene_counts': {label: len(genes) for label, genes in gene_lists.items()},
            'overlap_statistics': overlap_stats,
            'plots': plots
        }, f, indent=2)
    
    self.logger.info(f"Gene list comparison completed. Results saved to {output_dir}")
    
    return {
        'overlap_statistics': overlap_stats,
        'plots': plots,
        'results_file': str(results_path)
    }


async def compare_pathway_stage(
    self,
    dataset_map: Dict[str, str],
    parameters,  # ComparisonParameters
    output_dir: str,
    omic_type=None,
    data_type=None,
    tool="auto"
) -> Dict[str, Any]:
    """
    Compare datasets at pathway-stage with flexible input handling.
    
    Handles two input types:
    1. Pathway enrichment results (JSON files from prior analysis)
    2. Gene lists (will run enrichment first, then compare)
    
    Args:
        dataset_map: Dict mapping labels to file paths
        parameters: Comparison parameters
        output_dir: Output directory
        omic_type: Omic type (required if running enrichment)
        data_type: Data type (required if running enrichment)
        tool: Tool used to generate data
        
    Returns:
        Comparison results
    """
    self.logger.info(f"Pathway-stage comparison for {len(dataset_map)} labeled datasets")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if we need to run enrichment first
    if parameters.run_enrichment_first:
        self.logger.info("Running pathway enrichment on gene lists before comparison")
        
        # Run enrichment for each gene list
        enrichment_results = {}
        
        from ..analysis.engine import AnalysisEngine
        from ..analysis.schemas import AnalysisParameters, AnalysisType, DatabaseType
        
        analysis_engine = AnalysisEngine()
        
        for label, file_path in dataset_map.items():
            # Read gene list
            with open(file_path, 'r') as f:
                genes = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            self.logger.info(f"Running enrichment for {label} ({len(genes)} genes)")
            
            # Create analysis parameters
            analysis_params = AnalysisParameters(
                analysis_type=AnalysisType.ORA,
                omic_type=omic_type,
                data_type=data_type,
                databases=[DatabaseType(db) for db in parameters.databases],
                species=parameters.species,
                tool=tool
            )
            
            # Run analysis
            result = analysis_engine.analyze_sync(
                input_data=file_path,
                parameters=analysis_params,
                output_dir=f"{output_dir}/{label}_enrichment"
            )
            
            enrichment_results[label] = result
        
        # Now compare the enrichment results
        return await self._compare_enrichment_results(
            enrichment_results, parameters, output_dir
        )
    
    else:
        # Load existing enrichment results
        self.logger.info("Loading existing pathway enrichment results")
        
        enrichment_results = {}
        for label, file_path in dataset_map.items():
            with open(file_path, 'r') as f:
                result_data = json.load(f)
                # Convert to AnalysisResult object
                from ..analysis.schemas import AnalysisResult
                enrichment_results[label] = AnalysisResult(**result_data)
        
        return await self._compare_enrichment_results(
            enrichment_results, parameters, output_dir
        )


async def _compare_enrichment_results(
    self,
    enrichment_results: Dict[str, Any],
    parameters,  # ComparisonParameters
    output_dir: str
) -> Dict[str, Any]:
    """
    Compare pathway enrichment results from multiple labeled datasets.
    
    Args:
        enrichment_results: Dict mapping labels to AnalysisResult objects
        parameters: Comparison parameters
        output_dir: Output directory
        
    Returns:
        Comparison results with statistics and visualizations
    """
    self.logger.info(f"Comparing pathway enrichment for {len(enrichment_results)} labeled datasets")
    
    # Extract pathway data for each labeled dataset
    pathway_data = {}
    all_pathways = set()
    
    for label, result in enrichment_results.items():
        pathways = {}
        for db_name, db_result in result.database_results.items():
            for pathway in db_result.pathways:
                pathway_key = f"{db_name}:{pathway.pathway_id}"
                pathways[pathway_key] = {
                    'pathway_id': pathway.pathway_id,
                    'pathway_name': pathway.pathway_name,
                    'p_value': pathway.p_value,
                    'adjusted_p_value': pathway.adjusted_p_value,
                    'enrichment_score': pathway.enrichment_score,
                    'genes': set(pathway.overlapping_genes)
                }
                all_pathways.add(pathway_key)
        
        pathway_data[label] = pathways
    
    # Calculate pathway overlap statistics
    overlap_stats = {}
    labels = list(pathway_data.keys())
    
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            pathways1 = set(pathway_data[label1].keys())
            pathways2 = set(pathway_data[label2].keys())
            
            overlap = pathways1 & pathways2
            union = pathways1 | pathways2
            jaccard = len(overlap) / len(union) if union else 0.0
            
            overlap_stats[f"{label1}_vs_{label2}"] = {
                "label1": label1,
                "label2": label2,
                "pathways1_count": len(pathways1),
                "pathways2_count": len(pathways2),
                "overlap_count": len(overlap),
                "jaccard_index": jaccard,
                "overlapping_pathways": list(overlap),
                "unique_to_label1": list(pathways1 - pathways2),
                "unique_to_label2": list(pathways2 - pathways1)
            }
    
    # Calculate enrichment score correlations
    correlation_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    
    for label1 in labels:
        for label2 in labels:
            if label1 == label2:
                correlation_matrix.loc[label1, label2] = 1.0
            else:
                # Find common pathways
                common = set(pathway_data[label1].keys()) & set(pathway_data[label2].keys())
                
                if len(common) >= 2:
                    scores1 = [pathway_data[label1][p]['enrichment_score'] 
                              for p in common 
                              if pathway_data[label1][p]['enrichment_score'] is not None]
                    scores2 = [pathway_data[label2][p]['enrichment_score'] 
                              for p in common 
                              if pathway_data[label2][p]['enrichment_score'] is not None]
                    
                    if len(scores1) >= 2 and len(scores2) >= 2:
                        from scipy.stats import pearsonr
                        corr, _ = pearsonr(scores1, scores2)
                        correlation_matrix.loc[label1, label2] = corr
                    else:
                        correlation_matrix.loc[label1, label2] = 0.0
                else:
                    correlation_matrix.loc[label1, label2] = 0.0
    
    # Save results
    results = {
        'comparison_stage': 'pathway',
        'labels': labels,
        'pathway_counts': {label: len(pathways) for label, pathways in pathway_data.items()},
        'overlap_statistics': overlap_stats,
        'correlation_matrix': correlation_matrix.to_dict()
    }
    
    results_path = Path(output_dir) / "pathway_comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save correlation matrix as CSV
    corr_path = Path(output_dir) / "correlation_matrix.csv"
    correlation_matrix.to_csv(corr_path)
    
    # Create visualizations
    plots = {}
    
    # Correlation heatmap
    try:
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='RdBu_r', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Pathway Enrichment Score Correlation")
        
        heatmap_path = Path(output_dir) / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['correlation_heatmap'] = str(heatmap_path)
    except ImportError:
        self.logger.warning("seaborn not installed - skipping correlation heatmap")
    
    results['plots'] = plots
    
    self.logger.info(f"Pathway comparison completed. Results saved to {output_dir}")
    
    return results
