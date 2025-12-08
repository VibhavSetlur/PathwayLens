"""
Network renderer for pathway network visualizations.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class NetworkRenderer:
    """Renderer for creating pathway network visualizations."""
    
    def __init__(self):
        """Initialize the network renderer."""
        self.logger = logger.bind(module="network_renderer")
    
    def create_pathway_network(
        self,
        pathways: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        title: str = "Pathway Network",
        layout: str = "spring",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a pathway network visualization.
        
        Args:
            pathways: List of pathway dictionaries
            interactions: List of pathway interaction dictionaries
            title: Network title
            layout: Layout algorithm ('spring', 'circular', 'hierarchical')
            output_file: Optional output file path
            
        Returns:
            Dictionary with network data and visualization
        """
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add pathway nodes
            for pathway in pathways:
                pathway_id = pathway.get("pathway_id", "")
                pathway_name = pathway.get("pathway_name", pathway_id)
                
                G.add_node(
                    pathway_id,
                    name=pathway_name,
                    size=pathway.get("gene_count", 10),
                    color=pathway.get("color", "blue"),
                    p_value=pathway.get("p_value", 1.0)
                )
            
            # Add interactions
            for interaction in interactions:
                pathway1 = interaction.get("pathway1", "")
                pathway2 = interaction.get("pathway2", "")
                weight = interaction.get("weight", 1.0)
                
                if pathway1 in G.nodes and pathway2 in G.nodes:
                    G.add_edge(pathway1, pathway2, weight=weight)
            
            # Calculate layout
            if layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "hierarchical":
                pos = nx.hierarchical_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Create visualization data
            network_data = {
                "graph": G,
                "positions": pos,
                "nodes": list(G.nodes()),
                "edges": list(G.edges()),
                "layout": layout,
                "title": title
            }
            
            if output_file:
                self._save_network_plot(network_data, output_file)
            
            self.logger.info(f"Pathway network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return network_data
            
        except Exception as e:
            self.logger.error(f"Failed to create pathway network: {e}")
            return {}

    def create_enrichment_map(
        self,
        pathway_results: List[Any],
        similarity_cutoff: float = 0.375,
        similarity_metric: str = "jaccard",
        title: str = "Enrichment Map",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an Enrichment Map (network of pathways).
        
        Args:
            pathway_results: List of PathwayResult objects or dictionaries
            similarity_cutoff: Threshold for edge creation (default 0.375)
            similarity_metric: Metric for similarity ('jaccard', 'overlap', 'combined')
            title: Plot title
            output_file: Output file path
            
        Returns:
            Network data dictionary
        """
        try:
            # Extract pathway data
            pathways = []
            for p in pathway_results:
                # Handle PathwayResult object or dict
                if hasattr(p, 'pathway_id'):
                    p_id = p.pathway_id
                    p_name = p.pathway_name
                    genes = set(p.overlapping_genes) if hasattr(p, 'overlapping_genes') else set()
                    p_val = p.adjusted_p_value if hasattr(p, 'adjusted_p_value') else 1.0
                    size = p.pathway_count if hasattr(p, 'pathway_count') else len(genes)
                else:
                    p_id = p.get('pathway_id', '')
                    p_name = p.get('pathway_name', p_id)
                    genes = set(p.get('overlapping_genes', []))
                    p_val = p.get('adjusted_p_value', 1.0)
                    size = p.get('pathway_count', len(genes))
                
                pathways.append({
                    'pathway_id': p_id,
                    'pathway_name': p_name,
                    'genes': genes,
                    'p_value': p_val,
                    'gene_count': size,
                    # Color by significance (simple heuristic for now)
                    'color': 'red' if p_val < 0.05 else 'blue' 
                })
            
            # Calculate interactions (edges)
            interactions = []
            n = len(pathways)
            
            for i in range(n):
                for j in range(i + 1, n):
                    p1 = pathways[i]
                    p2 = pathways[j]
                    
                    genes1 = p1['genes']
                    genes2 = p2['genes']
                    
                    if not genes1 or not genes2:
                        continue
                        
                    intersection = len(genes1.intersection(genes2))
                    union = len(genes1.union(genes2))
                    min_size = min(len(genes1), len(genes2))
                    
                    if similarity_metric == "jaccard":
                        score = intersection / union if union > 0 else 0
                    elif similarity_metric == "overlap":
                        score = intersection / min_size if min_size > 0 else 0
                    elif similarity_metric == "combined":
                        jaccard = intersection / union if union > 0 else 0
                        overlap = intersection / min_size if min_size > 0 else 0
                        score = (jaccard + overlap) / 2
                    else:
                        score = 0
                        
                    if score >= similarity_cutoff:
                        interactions.append({
                            'pathway1': p1['pathway_id'],
                            'pathway2': p2['pathway_id'],
                            'weight': score,
                            'type': 'gene_overlap'
                        })
            
            self.logger.info(f"Generated {len(interactions)} edges for Enrichment Map")
            
            # Reuse create_pathway_network
            return self.create_pathway_network(
                pathways=pathways,
                interactions=interactions,
                title=title,
                layout="spring", # Force spring layout for enrichment maps usually
                output_file=output_file
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create Enrichment Map: {e}")
            return {}
    
    def create_gene_network(
        self,
        genes: List[str],
        interactions: List[Dict[str, Any]],
        title: str = "Gene Network",
        layout: str = "spring",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a gene network visualization.
        
        Args:
            genes: List of gene IDs
            interactions: List of gene interaction dictionaries
            title: Network title
            layout: Layout algorithm
            output_file: Optional output file path
            
        Returns:
            Dictionary with network data and visualization
        """
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add gene nodes
            for gene in genes:
                G.add_node(gene, name=gene, size=10, color="blue")
            
            # Add interactions
            for interaction in interactions:
                gene1 = interaction.get("gene1", "")
                gene2 = interaction.get("gene2", "")
                weight = interaction.get("weight", 1.0)
                interaction_type = interaction.get("type", "unknown")
                
                if gene1 in G.nodes and gene2 in G.nodes:
                    G.add_edge(gene1, gene2, weight=weight, type=interaction_type)
            
            # Calculate layout
            if layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Create visualization data
            network_data = {
                "graph": G,
                "positions": pos,
                "nodes": list(G.nodes()),
                "edges": list(G.edges()),
                "layout": layout,
                "title": title
            }
            
            if output_file:
                self._save_network_plot(network_data, output_file)
            
            self.logger.info(f"Gene network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return network_data
            
        except Exception as e:
            self.logger.error(f"Failed to create gene network: {e}")
            return {}
    
    def create_interactive_network_plot(
        self,
        network_data: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive network plot using Plotly.
        
        Args:
            network_data: Network data dictionary
            output_file: Optional output file path
            
        Returns:
            Plotly figure object
        """
        try:
            G = network_data["graph"]
            pos = network_data["positions"]
            
            # Create edge traces
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                edge_info.append(f"{edge[0]} - {edge[1]}")
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            node_sizes = []
            node_colors = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_data = G.nodes[node]
                node_text.append(node_data.get("name", node))
                node_info.append(f"ID: {node}<br>Name: {node_data.get('name', node)}<br>Size: {node_data.get('size', 10)}")
                node_sizes.append(node_data.get("size", 10))
                node_colors.append(node_data.get("color", "blue"))
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                hovertext=node_info,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=network_data.get("title", "Network"),
                              title_font_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Interactive network plot",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(color="black", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              width=800,
                              height=600
                          ))
            
            if output_file:
                fig.write_html(output_file)
                self.logger.info(f"Interactive network plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive network plot: {e}")
            return go.Figure()
    
    def create_static_network_plot(
        self,
        network_data: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a static network plot using matplotlib.
        
        Args:
            network_data: Network data dictionary
            output_file: Optional output file path
            
        Returns:
            Matplotlib figure object
        """
        try:
            G = network_data["graph"]
            pos = network_data["positions"]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                alpha=0.5,
                width=0.5,
                edge_color='gray'
            )
            
            # Draw nodes
            node_sizes = [G.nodes[node].get("size", 300) for node in G.nodes()]
            node_colors = [G.nodes[node].get("color", "blue") for node in G.nodes()]
            
            nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                alpha=0.7,
                edgecolors='white',
                linewidths=1
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=8,
                font_weight='bold'
            )
            
            ax.set_title(network_data.get("title", "Network"), fontsize=16)
            ax.axis('off')
            
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Static network plot saved to {output_file}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create static network plot: {e}")
            return plt.figure()
    
    def calculate_network_metrics(
        self,
        network_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate network metrics.
        
        Args:
            network_data: Network data dictionary
            
        Returns:
            Dictionary with network metrics
        """
        try:
            G = network_data["graph"]
            
            metrics = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "average_clustering": nx.average_clustering(G),
                "transitivity": nx.transitivity(G),
                "average_shortest_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else None,
                "diameter": nx.diameter(G) if nx.is_connected(G) else None,
                "radius": nx.radius(G) if nx.is_connected(G) else None,
                "center": nx.center(G) if nx.is_connected(G) else None,
                "periphery": nx.periphery(G) if nx.is_connected(G) else None
            }
            
            # Calculate centrality measures
            centrality_measures = {
                "degree_centrality": nx.degree_centrality(G),
                "betweenness_centrality": nx.betweenness_centrality(G),
                "closeness_centrality": nx.closeness_centrality(G),
                "eigenvector_centrality": nx.eigenvector_centrality(G)
            }
            
            metrics["centrality_measures"] = centrality_measures
            
            self.logger.info(f"Network metrics calculated for {metrics['nodes']} nodes and {metrics['edges']} edges")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate network metrics: {e}")
            return {}
    
    def find_communities(
        self,
        network_data: Dict[str, Any],
        algorithm: str = "louvain"
    ) -> Dict[str, Any]:
        """
        Find communities in the network.
        
        Args:
            network_data: Network data dictionary
            algorithm: Community detection algorithm
            
        Returns:
            Dictionary with community information
        """
        try:
            G = network_data["graph"]
            
            if algorithm == "louvain":
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.louvain_communities(G)
            elif algorithm == "greedy":
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.greedy_modularity_communities(G)
            else:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.louvain_communities(G)
            
            # Convert to dictionary
            community_dict = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_dict[node] = i
            
            # Calculate modularity
            modularity = nx_comm.modularity(G, communities)
            
            result = {
                "communities": list(communities),
                "community_dict": community_dict,
                "modularity": modularity,
                "algorithm": algorithm,
                "n_communities": len(communities)
            }
            
            self.logger.info(f"Found {len(communities)} communities with modularity {modularity:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to find communities: {e}")
            return {}
    
    def _save_network_plot(
        self,
        network_data: Dict[str, Any],
        output_file: str
    ) -> bool:
        """
        Save network plot to file.
        
        Args:
            network_data: Network data dictionary
            output_file: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if output_file.endswith('.html'):
                # Save as interactive HTML
                fig = self.create_interactive_network_plot(network_data)
                fig.write_html(output_file)
            else:
                # Save as static image
                fig = self.create_static_network_plot(network_data)
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
            
            self.logger.info(f"Network plot saved to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save network plot: {e}")
            return False
