"""
Comparison engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from loguru import logger
import json
from pathlib import Path
import matplotlib.pyplot as plt

from .schemas import (
    ComparisonResult, ComparisonParameters, ComparisonType,
    OverlapStatistics, CorrelationResult, ClusteringResult,
    PathwayConcordance
)
from ..analysis.schemas import AnalysisResult, DatabaseResult


class ComparisonEngine:
    """Engine for comparing multiple datasets and analysis results."""
    
    def __init__(self):
        """Initialize the comparison engine."""
        self.logger = logger.bind(module="comparison_engine")
    
    async def compare(
        self,
        analysis_results: List[AnalysisResult],
        parameters: ComparisonParameters,
        output_dir: Optional[str] = None
    ) -> ComparisonResult:
        """
        Perform comprehensive comparison of multiple analysis results.
        
        Args:
            analysis_results: List of analysis results to compare
            parameters: Comparison parameters
            output_dir: Output directory for results
            
        Returns:
            ComparisonResult with comparison analysis
        """
        self.logger.info(f"Starting comparison analysis with {len(analysis_results)} datasets")
        
        try:
            # Validate inputs
            if len(analysis_results) < 2:
                raise ValueError("At least 2 analysis results are required for comparison")
            
            # Extract comparison data
            comparison_data = self._extract_comparison_data(analysis_results, parameters)
            
            # Initialize default results
            results = {
                'overlap_statistics': {},
                'correlation_results': {},
                'clustering_results': None,
                'pathway_concordance': []
            }

            # Perform comparison based on type
            if parameters.comparison_type == ComparisonType.GENE_OVERLAP:
                results.update(await self._perform_gene_overlap_analysis(comparison_data, parameters))
            elif parameters.comparison_type == ComparisonType.PATHWAY_OVERLAP:
                results.update(await self._perform_pathway_overlap_analysis(comparison_data, parameters))
            elif parameters.comparison_type == ComparisonType.PATHWAY_CONCORDANCE:
                results.update(await self._perform_pathway_concordance_analysis(comparison_data, parameters))
            elif parameters.comparison_type == ComparisonType.ENRICHMENT_CORRELATION:
                results.update(await self._perform_enrichment_correlation_analysis(comparison_data, parameters))
            elif parameters.comparison_type == ComparisonType.DATASET_CLUSTERING:
                results.update(await self._perform_dataset_clustering_analysis(comparison_data, parameters))
            elif parameters.comparison_type == ComparisonType.COMPREHENSIVE:
                results.update(await self._perform_comprehensive_analysis(comparison_data, parameters))
            else:
                raise ValueError(f"Unsupported comparison type: {parameters.comparison_type}")
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(results, analysis_results)
            
            # Create comparison result
            comparison_result = ComparisonResult(
                job_id=f"comparison_{len(analysis_results)}_datasets",
                comparison_type=parameters.comparison_type,
                parameters=parameters,
                input_files=[result.input_file for result in analysis_results],
                num_datasets=len(analysis_results),
                **results,
                **summary_stats
            )
            
            self.logger.info("Comparison analysis completed successfully")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Comparison analysis failed: {e}")
            raise
    
    def _extract_comparison_data(
        self, 
        analysis_results: List[AnalysisResult], 
        parameters: ComparisonParameters
    ) -> Dict[str, Any]:
        """Extract data needed for comparison analysis."""
        comparison_data = {
            'datasets': {},
            'all_genes': set(),
            'all_pathways': set(),
            'pathway_data': {}
        }
        
        for i, result in enumerate(analysis_results):
            dataset_name = f"dataset_{i+1}"
            
            # Extract genes
            genes = set()
            for db_result in result.database_results.values():
                for pathway in db_result.pathways:
                    genes.update(pathway.overlapping_genes)
            
            comparison_data['datasets'][dataset_name] = {
                'genes': genes,
                'analysis_result': result,
                'database_results': result.database_results
            }
            
            comparison_data['all_genes'].update(genes)
            
            # Extract pathway data
            for db_name, db_result in result.database_results.items():
                for pathway in db_result.pathways:
                    pathway_key = f"{db_name}_{pathway.pathway_id}"
                    comparison_data['all_pathways'].add(pathway_key)
                    
                    if pathway_key not in comparison_data['pathway_data']:
                        comparison_data['pathway_data'][pathway_key] = {
                            'pathway_id': pathway.pathway_id,
                            'pathway_name': pathway.pathway_name,
                            'database': db_name,
                            'datasets': {}
                        }
                    
                    comparison_data['pathway_data'][pathway_key]['datasets'][dataset_name] = {
                        'p_value': pathway.p_value,
                        'adjusted_p_value': pathway.adjusted_p_value,
                        'enrichment_score': pathway.enrichment_score,
                        'overlap_count': pathway.overlap_count,
                        'pathway_count': pathway.pathway_count,
                        'overlapping_genes': set(pathway.overlapping_genes)
                    }
        
        return comparison_data
    
    async def _perform_gene_overlap_analysis(
        self, 
        comparison_data: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> Dict[str, Any]:
        """Perform gene overlap analysis."""
        self.logger.info("Performing gene overlap analysis")
        
        overlap_statistics = {}
        datasets = comparison_data['datasets']
        dataset_names = list(datasets.keys())
        
        for i, dataset1 in enumerate(dataset_names):
            for dataset2 in dataset_names[i+1:]:
                genes1 = datasets[dataset1]['genes']
                genes2 = datasets[dataset2]['genes']
                
                overlap_stats = self._calculate_gene_overlap(
                    dataset1, genes1, dataset2, genes2
                )
                
                key = f"{dataset1}_{dataset2}"
                overlap_statistics[key] = overlap_stats
        
        return {'overlap_statistics': overlap_statistics}

    def build_upset_sets_from_genes(
        self,
        comparison_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Build per-dataset gene sets for UpSet plotting."""
        return {name: list(info['genes']) for name, info in comparison_data['datasets'].items()}

    def build_upset_sets_from_pathways(
        self,
        comparison_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Build per-dataset pathway sets for UpSet plotting."""
        sets: Dict[str, List[str]] = {}
        for name, info in comparison_data['datasets'].items():
            pathways = set()
            for db_result in info['database_results'].values():
                for pathway in db_result.pathways:
                    pathways.add(f"{db_result.database.value}_{pathway.pathway_id}")
            sets[name] = list(pathways)
        return sets
    
    def _calculate_gene_overlap(
        self, 
        dataset1_name: str, 
        genes1: set, 
        dataset2_name: str, 
        genes2: set
    ) -> OverlapStatistics:
        """Calculate gene overlap statistics between two datasets."""
        total_genes1 = len(genes1)
        total_genes2 = len(genes2)
        overlapping_genes = genes1 & genes2
        overlap_count = len(overlapping_genes)
        
        # Calculate overlap percentage
        union_size = len(genes1 | genes2)
        overlap_percentage = overlap_count / union_size if union_size > 0 else 0.0
        
        # Calculate Jaccard index
        jaccard_index = overlap_count / union_size if union_size > 0 else 0.0
        
        # Get unique genes
        unique_genes1 = genes1 - genes2
        unique_genes2 = genes2 - genes1
        
        return OverlapStatistics(
            dataset1=dataset1_name,
            dataset2=dataset2_name,
            total_genes_dataset1=total_genes1,
            total_genes_dataset2=total_genes2,
            overlapping_genes=overlap_count,
            overlap_percentage=overlap_percentage,
            jaccard_index=jaccard_index,
            total_pathways_dataset1=0,  # Will be filled in pathway overlap
            total_pathways_dataset2=0,
            overlapping_pathways=0,
            pathway_overlap_percentage=0.0,
            pathway_jaccard_index=0.0,
            genes_dataset1=list(genes1),
            genes_dataset2=list(genes2),
            overlapping_gene_list=list(overlapping_genes),
            unique_genes_dataset1=list(unique_genes1),
            unique_genes_dataset2=list(unique_genes2)
        )
    
    async def _perform_pathway_overlap_analysis(
        self, 
        comparison_data: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> Dict[str, Any]:
        """Perform pathway overlap analysis."""
        self.logger.info("Performing pathway overlap analysis")
        
        overlap_statistics = {}
        datasets = comparison_data['datasets']
        dataset_names = list(datasets.keys())
        
        for i, dataset1 in enumerate(dataset_names):
            for dataset2 in dataset_names[i+1:]:
                # Get pathways for each dataset
                pathways1 = self._get_pathways_for_dataset(datasets[dataset1])
                pathways2 = self._get_pathways_for_dataset(datasets[dataset2])
                
                overlap_stats = self._calculate_pathway_overlap(
                    dataset1, pathways1, dataset2, pathways2
                )
                
                key = f"{dataset1}_{dataset2}"
                overlap_statistics[key] = overlap_stats
        
        return {'overlap_statistics': overlap_statistics}
    
    def _get_pathways_for_dataset(self, dataset_info: Dict[str, Any]) -> set:
        """Get pathway IDs for a dataset."""
        pathways = set()
        for db_result in dataset_info['database_results'].values():
            for pathway in db_result.pathways:
                pathways.add(f"{db_result.database.value}_{pathway.pathway_id}")
        return pathways
    
    def _calculate_pathway_overlap(
        self, 
        dataset1_name: str, 
        pathways1: set, 
        dataset2_name: str, 
        pathways2: set
    ) -> OverlapStatistics:
        """Calculate pathway overlap statistics between two datasets."""
        total_pathways1 = len(pathways1)
        total_pathways2 = len(pathways2)
        overlapping_pathways = pathways1 & pathways2
        overlap_count = len(overlapping_pathways)
        
        # Calculate pathway overlap percentage
        union_size = len(pathways1 | pathways2)
        pathway_overlap_percentage = overlap_count / union_size if union_size > 0 else 0.0
        
        # Calculate pathway Jaccard index
        pathway_jaccard_index = overlap_count / union_size if union_size > 0 else 0.0
        
        return OverlapStatistics(
            dataset1=dataset1_name,
            dataset2=dataset2_name,
            total_genes_dataset1=0,  # Will be filled in gene overlap
            total_genes_dataset2=0,
            overlapping_genes=0,
            overlap_percentage=0.0,
            jaccard_index=0.0,
            total_pathways_dataset1=total_pathways1,
            total_pathways_dataset2=total_pathways2,
            overlapping_pathways=overlap_count,
            pathway_overlap_percentage=pathway_overlap_percentage,
            pathway_jaccard_index=pathway_jaccard_index,
            genes_dataset1=[],
            genes_dataset2=[],
            overlapping_gene_list=[],
            unique_genes_dataset1=[],
            unique_genes_dataset2=[]
        )
    
    async def _perform_pathway_concordance_analysis(
        self, 
        comparison_data: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> Dict[str, Any]:
        """Perform pathway concordance analysis."""
        self.logger.info("Performing pathway concordance analysis")
        
        pathway_concordance = []
        pathway_data = comparison_data['pathway_data']
        
        for pathway_key, pathway_info in pathway_data.items():
            datasets = pathway_info['datasets']
            
            if len(datasets) < 2:
                continue
            
            concordance_result = self._calculate_pathway_concordance(
                pathway_info, datasets, parameters
            )
            
            if concordance_result:
                pathway_concordance.append(concordance_result)
        
        # Sort by concordance score
        pathway_concordance.sort(key=lambda x: x.concordance_score, reverse=True)
        
        return {'pathway_concordance': pathway_concordance}
    
    def _calculate_pathway_concordance(
        self, 
        pathway_info: Dict[str, Any], 
        datasets: Dict[str, Dict[str, Any]], 
        parameters: ComparisonParameters
    ) -> Optional[PathwayConcordance]:
        """Calculate concordance for a single pathway across datasets."""
        dataset_names = list(datasets.keys())
        num_datasets = len(dataset_names)
        
        if num_datasets < 2:
            return None
        
        # Extract p-values and effect sizes
        p_values = {}
        adjusted_p_values = {}
        effect_sizes = {}
        
        for dataset_name, dataset_info in datasets.items():
            p_values[dataset_name] = dataset_info['p_value']
            adjusted_p_values[dataset_name] = dataset_info['adjusted_p_value']
            if dataset_info['enrichment_score'] is not None:
                effect_sizes[dataset_name] = dataset_info['enrichment_score']
        
        # Calculate significance rate
        significant_count = sum(
            1 for p_val in adjusted_p_values.values() 
            if p_val <= parameters.significance_threshold
        )
        significance_rate = significant_count / num_datasets
        
        # Calculate concordance score
        concordance_score = self._calculate_concordance_score(
            p_values, effect_sizes, parameters
        )
        
        return PathwayConcordance(
            pathway_id=pathway_info['pathway_id'],
            pathway_name=pathway_info['pathway_name'],
            database=pathway_info['database'],
            concordance_score=concordance_score,
            num_datasets=num_datasets,
            num_significant=significant_count,
            significance_rate=significance_rate,
            p_values=p_values,
            adjusted_p_values=adjusted_p_values,
            effect_sizes=effect_sizes,
            pathway_size=datasets[dataset_names[0]]['pathway_count'],
            pathway_category=None
        )
    
    def _calculate_concordance_score(
        self, 
        p_values: Dict[str, float], 
        effect_sizes: Dict[str, float], 
        parameters: ComparisonParameters
    ) -> float:
        """Calculate concordance score for a pathway."""
        # Significance concordance
        significant_count = sum(
            1 for p_val in p_values.values() 
            if p_val <= parameters.significance_threshold
        )
        significance_concordance = significant_count / len(p_values)
        
        # Effect direction concordance
        if effect_sizes and len(effect_sizes) >= 2:
            positive_count = sum(1 for es in effect_sizes.values() if es > 1.0)
            direction_concordance = max(positive_count, len(effect_sizes) - positive_count) / len(effect_sizes)
        else:
            direction_concordance = 1.0
        
        # Combine concordance measures
        concordance_score = (significance_concordance + direction_concordance) / 2
        
        return concordance_score
    
    async def _perform_enrichment_correlation_analysis(
        self, 
        comparison_data: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> Dict[str, Any]:
        """Perform enrichment correlation analysis."""
        self.logger.info("Performing enrichment correlation analysis")
        
        correlation_results = {}
        datasets = comparison_data['datasets']
        dataset_names = list(datasets.keys())
        
        for i, dataset1 in enumerate(dataset_names):
            for dataset2 in dataset_names[i+1:]:
                correlation_result = self._calculate_enrichment_correlation(
                    dataset1, datasets[dataset1], dataset2, datasets[dataset2], parameters
                )
                
                key = f"{dataset1}_{dataset2}"
                correlation_results[key] = correlation_result
        
        return {'correlation_results': correlation_results}
    
    def _calculate_enrichment_correlation(
        self, 
        dataset1_name: str, 
        dataset1_info: Dict[str, Any], 
        dataset2_name: str, 
        dataset2_info: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> CorrelationResult:
        """Calculate correlation between enrichment profiles."""
        # Extract enrichment scores for common pathways
        scores1 = {}
        scores2 = {}
        
        for db_result in dataset1_info['database_results'].values():
            for pathway in db_result.pathways:
                key = f"{db_result.database.value}_{pathway.pathway_id}"
                if pathway.enrichment_score is not None:
                    scores1[key] = pathway.enrichment_score
        
        for db_result in dataset2_info['database_results'].values():
            for pathway in db_result.pathways:
                key = f"{db_result.database.value}_{pathway.pathway_id}"
                if pathway.enrichment_score is not None:
                    scores2[key] = pathway.enrichment_score
        
        # Find common pathways
        common_pathways = set(scores1.keys()) & set(scores2.keys())
        
        if len(common_pathways) < 2:
            return CorrelationResult(
                dataset1=dataset1_name,
                dataset2=dataset2_name,
                correlation=0.0,
                p_value=1.0,
                confidence_interval=[0.0, 0.0],
                sample_size=0,
                degrees_of_freedom=0,
                is_significant=False,
                significance_level=parameters.significance_threshold
            )
        
        # Extract scores for common pathways
        scores1_list = [scores1[p] for p in common_pathways]
        scores2_list = [scores2[p] for p in common_pathways]
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(scores1_list, scores2_list)
        
        # Calculate confidence interval
        n = len(common_pathways)
        df = n - 2
        
        if df > 0:
            se = np.sqrt((1 - correlation**2) / df)
            t_critical = stats.t.ppf(0.975, df)
            margin_error = t_critical * se
            
            confidence_interval = [
                correlation - margin_error,
                correlation + margin_error
            ]
        else:
            confidence_interval = [correlation, correlation]
        
        is_significant = p_value <= parameters.significance_threshold
        
        return CorrelationResult(
            dataset1=dataset1_name,
            dataset2=dataset2_name,
            correlation=correlation,
            p_value=p_value,
            confidence_interval=confidence_interval,
            sample_size=n,
            degrees_of_freedom=df,
            is_significant=is_significant,
            significance_level=parameters.significance_threshold
        )
    
    async def _perform_dataset_clustering_analysis(
        self, 
        comparison_data: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> Dict[str, Any]:
        """Perform dataset clustering analysis."""
        self.logger.info("Performing dataset clustering analysis")
        
        # Create feature matrix for clustering
        feature_matrix = self._create_clustering_features(comparison_data, parameters)
        
        if feature_matrix is None or feature_matrix.shape[0] < 2:
            return {'clustering_results': None}
        
        # Perform clustering
        clustering_result = self._perform_clustering(
            feature_matrix, comparison_data, parameters
        )
        
        return {'clustering_results': clustering_result}
    
    def _create_clustering_features(
        self, 
        comparison_data: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> Optional[np.ndarray]:
        """Create feature matrix for clustering."""
        datasets = comparison_data['datasets']
        dataset_names = list(datasets.keys())
        
        if len(dataset_names) < 2:
            return None
        
        # Extract features for each dataset
        features = []
        
        for dataset_name in dataset_names:
            dataset_info = datasets[dataset_name]
            
            # Basic features
            num_genes = len(dataset_info['genes'])
            num_pathways = sum(
                len(db_result.pathways) 
                for db_result in dataset_info['database_results'].values()
            )
            num_significant = sum(
                sum(1 for pathway in db_result.pathways if pathway.adjusted_p_value <= 0.05)
                for db_result in dataset_info['database_results'].values()
            )
            
            # Enrichment score features
            enrichment_scores = []
            for db_result in dataset_info['database_results'].values():
                for pathway in db_result.pathways:
                    if pathway.enrichment_score is not None:
                        enrichment_scores.append(pathway.enrichment_score)
            
            mean_enrichment = np.mean(enrichment_scores) if enrichment_scores else 0.0
            std_enrichment = np.std(enrichment_scores) if enrichment_scores else 0.0
            
            # P-value features
            p_values = []
            for db_result in dataset_info['database_results'].values():
                for pathway in db_result.pathways:
                    p_values.append(pathway.p_value)
            
            mean_p_value = np.mean(p_values) if p_values else 1.0
            median_p_value = np.median(p_values) if p_values else 1.0
            
            # Combine features
            feature_vector = [
                num_genes, num_pathways, num_significant,
                mean_enrichment, std_enrichment,
                mean_p_value, median_p_value
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _perform_clustering(
        self, 
        feature_matrix: np.ndarray, 
        comparison_data: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> ClusteringResult:
        """Perform clustering analysis."""
        n_datasets = feature_matrix.shape[0]
        dataset_names = list(comparison_data['datasets'].keys())
        
        # Determine number of clusters
        max_clusters = min(n_datasets, 5)
        best_score = -1
        best_n_clusters = 2
        best_labels = None
        
        # Try different numbers of clusters
        for n_clusters in range(2, max_clusters + 1):
            if parameters.clustering_method == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clusterer.fit_predict(feature_matrix)
            else:  # hierarchical
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=parameters.linkage_method
                )
                labels = clusterer.fit_predict(feature_matrix)
            
            # Calculate silhouette score
            if n_clusters > 1:
                score = silhouette_score(feature_matrix, labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
        
        # Use best clustering
        if best_labels is None:
            best_labels = np.zeros(n_datasets, dtype=int)
            best_n_clusters = 1
        
        # Calculate cluster statistics
        cluster_sizes = [np.sum(best_labels == i) for i in range(best_n_clusters)]
        cluster_datasets = {
            i: [dataset_names[j] for j in range(n_datasets) if best_labels[j] == i]
            for i in range(best_n_clusters)
        }
        
        # Calculate quality metrics
        if best_n_clusters > 1:
            silhouette = silhouette_score(feature_matrix, best_labels)
        else:
            silhouette = 0.0
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = 0.0
        for i in range(best_n_clusters):
            cluster_points = feature_matrix[best_labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)
        
        # Calculate Calinski-Harabasz score
        if best_n_clusters > 1 and n_datasets > best_n_clusters:
            from sklearn.metrics import calinski_harabasz_score
            ch_score = calinski_harabasz_score(feature_matrix, best_labels)
        else:
            ch_score = 0.0
        
        return ClusteringResult(
            method=parameters.clustering_method,
            distance_metric=parameters.distance_metric,
            linkage_method=parameters.linkage_method,
            num_clusters=best_n_clusters,
            cluster_labels=best_labels.tolist(),
            cluster_centers=[],  # Would need to calculate for KMeans
            silhouette_score=silhouette,
            inertia=inertia,
            calinski_harabasz_score=ch_score,
            cluster_sizes=cluster_sizes,
            cluster_datasets=cluster_datasets
        )
    
    async def _perform_comprehensive_analysis(
        self, 
        comparison_data: Dict[str, Any], 
        parameters: ComparisonParameters
    ) -> Dict[str, Any]:
        """Perform comprehensive comparison analysis."""
        self.logger.info("Performing comprehensive comparison analysis")
        
        # Perform all analysis types
        gene_overlap = await self._perform_gene_overlap_analysis(comparison_data, parameters)
        pathway_overlap = await self._perform_pathway_overlap_analysis(comparison_data, parameters)
        pathway_concordance = await self._perform_pathway_concordance_analysis(comparison_data, parameters)
        enrichment_correlation = await self._perform_enrichment_correlation_analysis(comparison_data, parameters)
        dataset_clustering = await self._perform_dataset_clustering_analysis(comparison_data, parameters)
        
        # Combine results
        results = {
            **gene_overlap,
            **pathway_overlap,
            **pathway_concordance,
            **enrichment_correlation,
            **dataset_clustering
        }
        
        return results
    
    def _calculate_summary_statistics(
        self, 
        results: Dict[str, Any], 
        analysis_results: List[AnalysisResult]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for comparison results."""
        # Calculate average overlap
        overlap_stats = results.get('overlap_statistics', {})
        if overlap_stats:
            overlaps = [stats.overlap_percentage for stats in overlap_stats.values()]
            average_overlap = np.mean(overlaps) if overlaps else 0.0
        else:
            average_overlap = 0.0
        
        # Calculate average correlation
        correlation_results = results.get('correlation_results', {})
        if correlation_results:
            correlations = [result.correlation for result in correlation_results.values()]
            average_correlation = np.mean(correlations) if correlations else 0.0
        else:
            average_correlation = 0.0
        
        # Calculate number of significant pathways
        pathway_concordance = results.get('pathway_concordance', [])
        num_significant_pathways = sum(
            1 for pathway in pathway_concordance
            if pathway.significance_rate >= 0.5  # At least 50% of datasets significant
        )
        
        # Calculate total and unique genes
        all_genes = set()
        for result in analysis_results:
            for db_result in result.database_results.values():
                for pathway in db_result.pathways:
                    all_genes.update(pathway.overlapping_genes)
        
        total_genes = len(all_genes)
        unique_genes = total_genes  # This would need more sophisticated calculation
        
        # Calculate quality metrics
        overall_quality = (average_overlap + abs(average_correlation)) / 2
        reproducibility = np.mean([pathway.concordance_score for pathway in pathway_concordance]) if pathway_concordance else 0.0
        
        return {
            'average_overlap': average_overlap,
            'average_correlation': average_correlation,
            'num_significant_pathways': num_significant_pathways,
            'total_genes': total_genes,
            'unique_genes': unique_genes,
            'overall_quality': overall_quality,
            'reproducibility': reproducibility,
            'created_at': pd.Timestamp.now().isoformat(),
            'completed_at': pd.Timestamp.now().isoformat(),
            'processing_time': 0.0,  # Would be calculated in actual implementation
            'output_files': {},
            'warnings': [],
            'errors': []
        }
    
    
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
    
    
        async def compare_counts_stage(
            self,
            dataset_map: Dict[str, str],
            parameters,  # ComparisonParameters
            output_dir: str,
            method: str = "deseq2"
        ) -> Dict[str, Any]:
            """
            Compare datasets at counts-stage using Differential Expression methods.
            
            Args:
                dataset_map: Dict mapping labels to file paths
                parameters: Comparison parameters
                output_dir: Output directory
                method: DE method (deseq2, edger, limma, simple)
                
            Returns:
                Comparison results
            """
            self.logger.info(f"Counts-stage comparison using {method} for {len(dataset_map)} datasets")
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            results = {
                'comparison_stage': 'counts',
                'method': method,
                'datasets': list(dataset_map.keys()),
                'de_results': {}
            }
            
            # Placeholder for actual DE implementation
            # In a real implementation, this would call R scripts or Python DE libraries
            
            if method == "simple":
                # Implement simple t-test or fold change here if possible
                self.logger.info("Running simple comparison (logFC)")
                # ... implementation ...
            elif method in ["deseq2", "edger", "limma"]:
                self.logger.info(f"Preparing to run {method} (requires R/Bioconductor)")
                # Check for R availability, etc.
                # For now, we'll just log that it's a placeholder
                self.logger.warning(f"{method} execution is not fully implemented in this version. Please use external tools for DE and provide gene lists.")
            else:
                self.logger.warning(f"Unknown method: {method}")
                
            # Save placeholder results
            results_path = Path(output_dir) / "de_comparison_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            return results
