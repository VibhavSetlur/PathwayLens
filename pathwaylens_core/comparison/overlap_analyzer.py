"""
Overlap analyzer for comparing datasets and pathway results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from collections import Counter


class OverlapAnalyzer:
    """Analyzer for calculating overlaps between datasets and pathway results."""
    
    def __init__(self):
        """Initialize the overlap analyzer."""
        self.logger = logger.bind(module="overlap_analyzer")
    
    def analyze_gene_overlap(
        self,
        datasets: Dict[str, List[str]],
        min_overlap: int = 1
    ) -> Dict[str, Any]:
        """
        Analyze gene overlap between multiple datasets.
        
        Args:
            datasets: Dictionary of dataset name -> gene list
            min_overlap: Minimum overlap size to report
            
        Returns:
            Dictionary with overlap analysis results
        """
        self.logger.info(f"Analyzing gene overlap between {len(datasets)} datasets")
        
        try:
            # Convert to sets for efficient operations
            dataset_sets = {name: set(genes) for name, genes in datasets.items()}
            
            # Calculate pairwise overlaps
            pairwise_overlaps = self._calculate_pairwise_overlaps(dataset_sets, min_overlap)
            
            # Calculate multi-way overlaps
            multi_way_overlaps = self._calculate_multi_way_overlaps(dataset_sets, min_overlap)
            
            # Calculate overlap statistics
            overlap_stats = self._calculate_overlap_statistics(dataset_sets)
            
            # Generate Venn diagram data
            venn_data = self._generate_venn_data(dataset_sets)
            
            result = {
                "pairwise_overlaps": pairwise_overlaps,
                "multi_way_overlaps": multi_way_overlaps,
                "overlap_statistics": overlap_stats,
                "venn_diagram_data": venn_data,
                "total_datasets": len(datasets),
                "min_overlap": min_overlap
            }
            
            self.logger.info(f"Gene overlap analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Gene overlap analysis failed: {e}")
            return {}
    
    def analyze_pathway_overlap(
        self,
        pathway_results: Dict[str, List[Dict[str, Any]]],
        significance_threshold: float = 0.05,
        min_overlap: int = 1
    ) -> Dict[str, Any]:
        """
        Analyze pathway overlap between multiple analysis results.
        
        Args:
            pathway_results: Dictionary of analysis name -> pathway results
            significance_threshold: P-value threshold for significant pathways
            min_overlap: Minimum overlap size to report
            
        Returns:
            Dictionary with pathway overlap analysis results
        """
        self.logger.info(f"Analyzing pathway overlap between {len(pathway_results)} analyses")
        
        try:
            # Extract significant pathways
            significant_pathways = self._extract_significant_pathways(
                pathway_results, significance_threshold
            )
            
            # Calculate pathway overlaps
            pathway_overlaps = self._calculate_pathway_overlaps(
                significant_pathways, min_overlap
            )
            
            # Calculate pathway concordance
            pathway_concordance = self._calculate_pathway_concordance(pathway_results)
            
            # Calculate pathway similarity
            pathway_similarity = self._calculate_pathway_similarity(significant_pathways)
            
            result = {
                "pathway_overlaps": pathway_overlaps,
                "pathway_concordance": pathway_concordance,
                "pathway_similarity": pathway_similarity,
                "significant_pathways": significant_pathways,
                "total_analyses": len(pathway_results),
                "significance_threshold": significance_threshold,
                "min_overlap": min_overlap
            }
            
            self.logger.info(f"Pathway overlap analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Pathway overlap analysis failed: {e}")
            return {}
    
    def _calculate_pairwise_overlaps(
        self,
        dataset_sets: Dict[str, set],
        min_overlap: int
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate pairwise overlaps between datasets."""
        pairwise_overlaps = {}
        
        dataset_names = list(dataset_sets.keys())
        for i, dataset1 in enumerate(dataset_names):
            pairwise_overlaps[dataset1] = {}
            for dataset2 in dataset_names[i+1:]:
                overlap = dataset_sets[dataset1].intersection(dataset_sets[dataset2])
                
                if len(overlap) >= min_overlap:
                    pairwise_overlaps[dataset1][dataset2] = {
                        "overlap_count": len(overlap),
                        "overlap_genes": list(overlap),
                        "dataset1_size": len(dataset_sets[dataset1]),
                        "dataset2_size": len(dataset_sets[dataset2]),
                        "jaccard_index": len(overlap) / len(dataset_sets[dataset1].union(dataset_sets[dataset2])),
                        "overlap_coefficient": len(overlap) / min(len(dataset_sets[dataset1]), len(dataset_sets[dataset2]))
                    }
        
        return pairwise_overlaps
    
    def _calculate_multi_way_overlaps(
        self,
        dataset_sets: Dict[str, set],
        min_overlap: int
    ) -> Dict[str, Any]:
        """Calculate multi-way overlaps between datasets."""
        multi_way_overlaps = {}
        
        dataset_names = list(dataset_sets.keys())
        
        # Calculate overlaps for all combinations
        from itertools import combinations
        
        for r in range(2, len(dataset_names) + 1):
            for combo in combinations(dataset_names, r):
                combo_name = "_".join(combo)
                overlap = set.intersection(*[dataset_sets[name] for name in combo])
                
                if len(overlap) >= min_overlap:
                    multi_way_overlaps[combo_name] = {
                        "datasets": list(combo),
                        "overlap_count": len(overlap),
                        "overlap_genes": list(overlap),
                        "union_size": len(set.union(*[dataset_sets[name] for name in combo])),
                        "jaccard_index": len(overlap) / len(set.union(*[dataset_sets[name] for name in combo]))
                    }
        
        return multi_way_overlaps
    
    def _calculate_overlap_statistics(
        self,
        dataset_sets: Dict[str, set]
    ) -> Dict[str, Any]:
        """Calculate overlap statistics."""
        stats = {}
        
        # Overall statistics
        all_genes = set.union(*dataset_sets.values())
        common_genes = set.intersection(*dataset_sets.values())
        
        stats["total_unique_genes"] = len(all_genes)
        stats["common_genes"] = len(common_genes)
        stats["common_genes_list"] = list(common_genes)
        
        # Dataset-specific statistics
        stats["dataset_sizes"] = {name: len(genes) for name, genes in dataset_sets.items()}
        
        # Coverage statistics
        stats["coverage_per_dataset"] = {}
        for name, genes in dataset_sets.items():
            coverage = len(genes) / len(all_genes)
            stats["coverage_per_dataset"][name] = coverage
        
        return stats
    
    def _generate_venn_data(
        self,
        dataset_sets: Dict[str, set]
    ) -> Dict[str, Any]:
        """Generate data for Venn diagram visualization."""
        if len(dataset_sets) > 5:
            self.logger.warning("Venn diagram not recommended for more than 5 datasets")
            return {}
        
        venn_data = {
            "datasets": list(dataset_sets.keys()),
            "sizes": {name: len(genes) for name, genes in dataset_sets.items()},
            "overlaps": {}
        }
        
        # Calculate all possible overlaps
        from itertools import combinations, chain
        
        dataset_names = list(dataset_sets.keys())
        for r in range(1, len(dataset_names) + 1):
            for combo in combinations(dataset_names, r):
                if r == 1:
                    # Single dataset
                    name = combo[0]
                    venn_data["overlaps"][name] = {
                        "genes": list(dataset_sets[name]),
                        "size": len(dataset_sets[name])
                    }
                else:
                    # Multi-dataset overlap
                    combo_name = "_".join(combo)
                    overlap = set.intersection(*[dataset_sets[name] for name in combo])
                    
                    # Subtract overlaps with larger combinations
                    for larger_combo in combinations(dataset_names, r + 1):
                        if all(name in larger_combo for name in combo):
                            larger_overlap = set.intersection(*[dataset_sets[name] for name in larger_combo])
                            overlap = overlap - larger_overlap
                    
                    venn_data["overlaps"][combo_name] = {
                        "genes": list(overlap),
                        "size": len(overlap),
                        "datasets": list(combo)
                    }
        
        return venn_data
    
    def _extract_significant_pathways(
        self,
        pathway_results: Dict[str, List[Dict[str, Any]]],
        significance_threshold: float
    ) -> Dict[str, List[str]]:
        """Extract significant pathways from analysis results."""
        significant_pathways = {}
        
        for analysis_name, pathways in pathway_results.items():
            significant = []
            for pathway in pathways:
                if pathway.get("adjusted_p_value", 1.0) <= significance_threshold:
                    significant.append(pathway.get("pathway_id", ""))
            
            significant_pathways[analysis_name] = [p for p in significant if p]
        
        return significant_pathways
    
    def _calculate_pathway_overlaps(
        self,
        significant_pathways: Dict[str, List[str]],
        min_overlap: int
    ) -> Dict[str, Any]:
        """Calculate overlaps between significant pathways."""
        return self.analyze_gene_overlap(significant_pathways, min_overlap)
    
    def _calculate_pathway_concordance(
        self,
        pathway_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate pathway concordance between analyses."""
        concordance = {}
        
        # Get all unique pathway IDs
        all_pathways = set()
        for pathways in pathway_results.values():
            for pathway in pathways:
                all_pathways.add(pathway.get("pathway_id", ""))
        
        all_pathways = {p for p in all_pathways if p}
        
        # Calculate concordance for each pathway
        pathway_concordance = {}
        for pathway_id in all_pathways:
            pathway_scores = {}
            for analysis_name, pathways in pathway_results.items():
                for pathway in pathways:
                    if pathway.get("pathway_id") == pathway_id:
                        pathway_scores[analysis_name] = {
                            "p_value": pathway.get("p_value", 1.0),
                            "adjusted_p_value": pathway.get("adjusted_p_value", 1.0),
                            "enrichment_score": pathway.get("enrichment_score", 0.0)
                        }
                        break
            
            if len(pathway_scores) > 1:
                # Calculate concordance metrics
                p_values = [s["adjusted_p_value"] for s in pathway_scores.values()]
                enrichment_scores = [s["enrichment_score"] for s in pathway_scores.values()]
                
                concordance_score = 1.0 - np.std(p_values) / np.mean(p_values) if np.mean(p_values) > 0 else 0.0
                enrichment_concordance = 1.0 - np.std(enrichment_scores) / (np.mean(enrichment_scores) + 1e-10)
                
                pathway_concordance[pathway_id] = {
                    "pathway_scores": pathway_scores,
                    "concordance_score": concordance_score,
                    "enrichment_concordance": enrichment_concordance,
                    "mean_p_value": np.mean(p_values),
                    "mean_enrichment_score": np.mean(enrichment_scores)
                }
        
        concordance["pathway_concordance"] = pathway_concordance
        concordance["overall_concordance"] = np.mean([p["concordance_score"] for p in pathway_concordance.values()])
        
        return concordance
    
    def _calculate_pathway_similarity(
        self,
        significant_pathways: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Calculate pathway similarity between analyses."""
        similarity = {}
        
        # Calculate pairwise similarities
        analysis_names = list(significant_pathways.keys())
        for i, analysis1 in enumerate(analysis_names):
            similarity[analysis1] = {}
            for analysis2 in analysis_names[i+1:]:
                pathways1 = set(significant_pathways[analysis1])
                pathways2 = set(significant_pathways[analysis2])
                
                if len(pathways1) > 0 or len(pathways2) > 0:
                    jaccard = len(pathways1.intersection(pathways2)) / len(pathways1.union(pathways2))
                    overlap_coef = len(pathways1.intersection(pathways2)) / min(len(pathways1), len(pathways2)) if min(len(pathways1), len(pathways2)) > 0 else 0
                    
                    similarity[analysis1][analysis2] = {
                        "jaccard_index": jaccard,
                        "overlap_coefficient": overlap_coef,
                        "shared_pathways": len(pathways1.intersection(pathways2)),
                        "unique_to_1": len(pathways1 - pathways2),
                        "unique_to_2": len(pathways2 - pathways1)
                    }
        
        return similarity
