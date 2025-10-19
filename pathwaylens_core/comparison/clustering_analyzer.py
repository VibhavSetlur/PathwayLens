"""
Clustering analyzer for grouping similar datasets and pathway results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class ClusteringAnalyzer:
    """Analyzer for clustering datasets and pathway results."""
    
    def __init__(self):
        """Initialize the clustering analyzer."""
        self.logger = logger.bind(module="clustering_analyzer")
    
    def cluster_pathway_profiles(
        self,
        pathway_results: Dict[str, List[Dict[str, Any]]],
        clustering_method: str = "kmeans",
        n_clusters: Optional[int] = None,
        max_clusters: int = 10,
        significance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Cluster pathway enrichment profiles.
        
        Args:
            pathway_results: Dictionary of analysis name -> pathway results
            clustering_method: Clustering method ('kmeans', 'hierarchical', 'gmm', 'dbscan')
            n_clusters: Number of clusters (auto-determined if None)
            max_clusters: Maximum number of clusters for auto-determination
            significance_threshold: P-value threshold for significant pathways
            
        Returns:
            Dictionary with clustering results
        """
        self.logger.info(f"Clustering pathway profiles using {clustering_method} method")
        
        try:
            # Create pathway enrichment matrix
            enrichment_matrix = self._create_enrichment_matrix(pathway_results, significance_threshold)
            
            if enrichment_matrix.empty:
                self.logger.warning("No pathways found for clustering analysis")
                return {}
            
            # Determine optimal number of clusters
            if n_clusters is None:
                n_clusters = self._determine_optimal_clusters(
                    enrichment_matrix, max_clusters, clustering_method
                )
            
            # Perform clustering
            cluster_result = self._perform_clustering(
                enrichment_matrix, clustering_method, n_clusters
            )
            
            # Calculate cluster quality metrics
            quality_metrics = self._calculate_cluster_quality(
                enrichment_matrix, cluster_result["cluster_labels"]
            )
            
            # Analyze cluster characteristics
            cluster_characteristics = self._analyze_cluster_characteristics(
                enrichment_matrix, cluster_result["cluster_labels"]
            )
            
            # Perform dimensionality reduction for visualization
            reduced_data = self._perform_dimensionality_reduction(enrichment_matrix)
            
            result = {
                "enrichment_matrix": enrichment_matrix.to_dict(),
                "cluster_labels": cluster_result["cluster_labels"],
                "cluster_centers": cluster_result["cluster_centers"],
                "n_clusters": n_clusters,
                "clustering_method": clustering_method,
                "quality_metrics": quality_metrics,
                "cluster_characteristics": cluster_characteristics,
                "reduced_data": reduced_data,
                "analysis_names": list(pathway_results.keys())
            }
            
            self.logger.info(f"Pathway clustering completed with {n_clusters} clusters")
            return result
            
        except Exception as e:
            self.logger.error(f"Pathway clustering failed: {e}")
            return {}
    
    def cluster_expression_profiles(
        self,
        expression_data: Dict[str, pd.DataFrame],
        clustering_method: str = "kmeans",
        n_clusters: Optional[int] = None,
        max_clusters: int = 10,
        feature_selection: str = "variance"
    ) -> Dict[str, Any]:
        """
        Cluster gene expression profiles.
        
        Args:
            expression_data: Dictionary of dataset name -> expression matrix
            clustering_method: Clustering method
            n_clusters: Number of clusters (auto-determined if None)
            max_clusters: Maximum number of clusters for auto-determination
            feature_selection: Feature selection method ('variance', 'correlation', 'all')
            
        Returns:
            Dictionary with expression clustering results
        """
        self.logger.info(f"Clustering expression profiles using {clustering_method} method")
        
        try:
            # Align and preprocess expression data
            processed_data = self._preprocess_expression_data(expression_data, feature_selection)
            
            if processed_data.empty:
                self.logger.warning("No expression data available for clustering")
                return {}
            
            # Determine optimal number of clusters
            if n_clusters is None:
                n_clusters = self._determine_optimal_clusters(
                    processed_data, max_clusters, clustering_method
                )
            
            # Perform clustering
            cluster_result = self._perform_clustering(
                processed_data, clustering_method, n_clusters
            )
            
            # Calculate cluster quality metrics
            quality_metrics = self._calculate_cluster_quality(
                processed_data, cluster_result["cluster_labels"]
            )
            
            # Analyze cluster characteristics
            cluster_characteristics = self._analyze_cluster_characteristics(
                processed_data, cluster_result["cluster_labels"]
            )
            
            # Perform dimensionality reduction
            reduced_data = self._perform_dimensionality_reduction(processed_data)
            
            result = {
                "expression_matrix": processed_data.to_dict(),
                "cluster_labels": cluster_result["cluster_labels"],
                "cluster_centers": cluster_result["cluster_centers"],
                "n_clusters": n_clusters,
                "clustering_method": clustering_method,
                "quality_metrics": quality_metrics,
                "cluster_characteristics": cluster_characteristics,
                "reduced_data": reduced_data,
                "dataset_names": list(expression_data.keys())
            }
            
            self.logger.info(f"Expression clustering completed with {n_clusters} clusters")
            return result
            
        except Exception as e:
            self.logger.error(f"Expression clustering failed: {e}")
            return {}
    
    def _create_enrichment_matrix(
        self,
        pathway_results: Dict[str, List[Dict[str, Any]]],
        significance_threshold: float
    ) -> pd.DataFrame:
        """Create enrichment matrix from pathway results."""
        try:
            # Collect all pathway IDs
            all_pathways = set()
            for pathways in pathway_results.values():
                for pathway in pathways:
                    pathway_id = pathway.get("pathway_id")
                    if pathway_id:
                        all_pathways.add(pathway_id)
            
            all_pathways = sorted(list(all_pathways))
            
            # Create matrix
            enrichment_matrix = pd.DataFrame(index=all_pathways)
            
            for analysis_name, pathways in pathway_results.items():
                # Create pathway lookup
                pathway_lookup = {p.get("pathway_id"): p for p in pathways}
                
                # Extract enrichment scores
                enrichment_scores = []
                for pathway_id in all_pathways:
                    if pathway_id in pathway_lookup:
                        pathway = pathway_lookup[pathway_id]
                        p_value = pathway.get("adjusted_p_value", 1.0)
                        
                        if p_value <= significance_threshold:
                            score = pathway.get("enrichment_score", 0.0)
                            if score == 0.0:
                                score = -np.log10(max(p_value, 1e-300))
                            enrichment_scores.append(score)
                        else:
                            enrichment_scores.append(0.0)
                    else:
                        enrichment_scores.append(0.0)
                
                enrichment_matrix[analysis_name] = enrichment_scores
            
            return enrichment_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to create enrichment matrix: {e}")
            return pd.DataFrame()
    
    def _preprocess_expression_data(
        self,
        expression_data: Dict[str, pd.DataFrame],
        feature_selection: str
    ) -> pd.DataFrame:
        """Preprocess expression data for clustering."""
        try:
            # Align datasets by common genes
            common_genes = None
            for name, data in expression_data.items():
                if common_genes is None:
                    common_genes = set(data.index)
                else:
                    common_genes = common_genes.intersection(set(data.index))
            
            if not common_genes:
                return pd.DataFrame()
            
            # Create combined matrix
            combined_data = pd.DataFrame()
            for name, data in expression_data.items():
                common_data = data.loc[list(common_genes)]
                # Use mean expression across samples
                mean_expression = common_data.mean(axis=1)
                combined_data[name] = mean_expression
            
            # Feature selection
            if feature_selection == "variance":
                # Select top variable features
                variances = combined_data.var(axis=1)
                top_features = variances.nlargest(min(1000, len(combined_data))).index
                combined_data = combined_data.loc[top_features]
            elif feature_selection == "correlation":
                # Select features with high correlation variance
                corr_matrix = combined_data.corr()
                corr_vars = corr_matrix.var(axis=1)
                top_features = corr_vars.nlargest(min(1000, len(combined_data))).index
                combined_data = combined_data.loc[:, top_features]
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess expression data: {e}")
            return pd.DataFrame()
    
    def _determine_optimal_clusters(
        self,
        data: pd.DataFrame,
        max_clusters: int,
        clustering_method: str
    ) -> int:
        """Determine optimal number of clusters using elbow method and silhouette analysis."""
        try:
            if data.shape[0] < 2:
                return 1
            
            max_clusters = min(max_clusters, data.shape[0])
            if max_clusters < 2:
                return 1
            
            # Calculate metrics for different numbers of clusters
            silhouette_scores = []
            inertias = []
            
            for k in range(2, max_clusters + 1):
                try:
                    if clustering_method == "kmeans":
                        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                    elif clustering_method == "hierarchical":
                        clusterer = AgglomerativeClustering(n_clusters=k)
                    elif clustering_method == "gmm":
                        clusterer = GaussianMixture(n_components=k, random_state=42)
                    else:
                        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                    
                    cluster_labels = clusterer.fit_predict(data.T)
                    
                    # Calculate silhouette score
                    silhouette_avg = silhouette_score(data.T, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                    
                    # Calculate inertia (for KMeans)
                    if hasattr(clusterer, 'inertia_'):
                        inertias.append(clusterer.inertia_)
                    else:
                        inertias.append(0)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to cluster with k={k}: {e}")
                    silhouette_scores.append(0)
                    inertias.append(0)
            
            # Find optimal k using silhouette score
            if silhouette_scores:
                optimal_k = np.argmax(silhouette_scores) + 2
            else:
                optimal_k = 2
            
            self.logger.info(f"Optimal number of clusters determined: {optimal_k}")
            return optimal_k
            
        except Exception as e:
            self.logger.error(f"Failed to determine optimal clusters: {e}")
            return 2
    
    def _perform_clustering(
        self,
        data: pd.DataFrame,
        clustering_method: str,
        n_clusters: int
    ) -> Dict[str, Any]:
        """Perform clustering on the data."""
        try:
            if data.shape[0] < 2:
                return {
                    "cluster_labels": [0] * data.shape[1],
                    "cluster_centers": []
                }
            
            if clustering_method == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(data.T)
                cluster_centers = clusterer.cluster_centers_.tolist()
                
            elif clustering_method == "hierarchical":
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clusterer.fit_predict(data.T)
                cluster_centers = self._calculate_cluster_centers(data, cluster_labels)
                
            elif clustering_method == "gmm":
                clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(data.T)
                cluster_centers = clusterer.means_.tolist()
                
            elif clustering_method == "dbscan":
                clusterer = DBSCAN(eps=0.5, min_samples=2)
                cluster_labels = clusterer.fit_predict(data.T)
                cluster_centers = self._calculate_cluster_centers(data, cluster_labels)
                
            else:
                # Default to KMeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(data.T)
                cluster_centers = clusterer.cluster_centers_.tolist()
            
            return {
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": cluster_centers
            }
            
        except Exception as e:
            self.logger.error(f"Failed to perform clustering: {e}")
            return {
                "cluster_labels": [0] * data.shape[1],
                "cluster_centers": []
            }
    
    def _calculate_cluster_centers(
        self,
        data: pd.DataFrame,
        cluster_labels: List[int]
    ) -> List[List[float]]:
        """Calculate cluster centers for non-centroid clustering methods."""
        try:
            unique_labels = list(set(cluster_labels))
            cluster_centers = []
            
            for label in unique_labels:
                if label == -1:  # Skip noise points for DBSCAN
                    continue
                    
                cluster_mask = np.array(cluster_labels) == label
                cluster_data = data.iloc[:, cluster_mask]
                
                if cluster_data.shape[1] > 0:
                    center = cluster_data.mean(axis=1).values.tolist()
                    cluster_centers.append(center)
            
            return cluster_centers
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cluster centers: {e}")
            return []
    
    def _calculate_cluster_quality(
        self,
        data: pd.DataFrame,
        cluster_labels: List[int]
    ) -> Dict[str, float]:
        """Calculate cluster quality metrics."""
        try:
            if len(set(cluster_labels)) < 2:
                return {"silhouette_score": 0.0, "calinski_harabasz_score": 0.0, "davies_bouldin_score": float('inf')}
            
            # Remove noise points (-1) for quality calculations
            valid_mask = np.array(cluster_labels) != -1
            if np.sum(valid_mask) < 2:
                return {"silhouette_score": 0.0, "calinski_harabasz_score": 0.0, "davies_bouldin_score": float('inf')}
            
            valid_data = data.iloc[:, valid_mask]
            valid_labels = np.array(cluster_labels)[valid_mask]
            
            # Calculate quality metrics
            silhouette_avg = silhouette_score(valid_data.T, valid_labels)
            calinski_harabasz = calinski_harabasz_score(valid_data.T, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_data.T, valid_labels)
            
            return {
                "silhouette_score": silhouette_avg,
                "calinski_harabasz_score": calinski_harabasz,
                "davies_bouldin_score": davies_bouldin
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cluster quality: {e}")
            return {"silhouette_score": 0.0, "calinski_harabasz_score": 0.0, "davies_bouldin_score": float('inf')}
    
    def _analyze_cluster_characteristics(
        self,
        data: pd.DataFrame,
        cluster_labels: List[int]
    ) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""
        try:
            cluster_characteristics = {}
            unique_labels = list(set(cluster_labels))
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                cluster_mask = np.array(cluster_labels) == label
                cluster_data = data.iloc[:, cluster_mask]
                cluster_names = data.columns[cluster_mask].tolist()
                
                if cluster_data.shape[1] > 0:
                    characteristics = {
                        "size": cluster_data.shape[1],
                        "members": cluster_names,
                        "mean_profile": cluster_data.mean(axis=1).to_dict(),
                        "std_profile": cluster_data.std(axis=1).to_dict(),
                        "top_features": self._get_top_features(cluster_data)
                    }
                    
                    cluster_characteristics[str(label)] = characteristics
            
            return cluster_characteristics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze cluster characteristics: {e}")
            return {}
    
    def _get_top_features(
        self,
        cluster_data: pd.DataFrame,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top features for a cluster."""
        try:
            # Calculate mean expression for each feature
            mean_expression = cluster_data.mean(axis=1)
            
            # Get top features
            top_features = mean_expression.nlargest(top_n)
            
            return [
                {"feature": feature, "mean_expression": float(expr)}
                for feature, expr in top_features.items()
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get top features: {e}")
            return []
    
    def _perform_dimensionality_reduction(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform dimensionality reduction for visualization."""
        try:
            reduced_data = {}
            
            # PCA
            if data.shape[0] > 1 and data.shape[1] > 1:
                pca = PCA(n_components=min(2, data.shape[1]))
                pca_result = pca.fit_transform(data.T)
                
                reduced_data["pca"] = {
                    "coordinates": pca_result.tolist(),
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "feature_names": data.columns.tolist()
                }
            
            # t-SNE (if data is not too large)
            if data.shape[1] <= 50:
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, data.shape[1]-1))
                    tsne_result = tsne.fit_transform(data.T)
                    
                    reduced_data["tsne"] = {
                        "coordinates": tsne_result.tolist(),
                        "feature_names": data.columns.tolist()
                    }
                except Exception as e:
                    self.logger.warning(f"t-SNE failed: {e}")
            
            return reduced_data
            
        except Exception as e:
            self.logger.error(f"Failed to perform dimensionality reduction: {e}")
            return {}
