"""
Multi-omics visualization methods for PathwayLens.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Union
from loguru import logger

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Network visualizations will be limited.")


class MultiOmicsVisualizer:
    """Visualization utilities for multi-omics data."""
    
    def __init__(self):
        """Initialize multi-omics visualizer."""
        self.logger = logger.bind(module="multi_omics_visualizer")
    
    def create_multi_omics_heatmap(
        self,
        omics_data: Dict[str, Dict[str, Any]],
        pathway_names: Optional[List[str]] = None,
        max_pathways: int = 50
    ) -> go.Figure:
        """
        Create multi-omics heatmap visualization.
        
        Args:
            omics_data: Dictionary mapping omics types to pathway results
            pathway_names: List of pathway names to include (None = all)
            max_pathways: Maximum number of pathways to display
            
        Returns:
            Plotly figure with heatmap
        """
        # Collect all pathways across omics types
        all_pathways = set()
        omics_types = list(omics_data.keys())
        
        for omics_type, results in omics_data.items():
            if isinstance(results, dict) and 'pathways' in results:
                for pathway in results['pathways']:
                    if isinstance(pathway, dict):
                        all_pathways.add(pathway.get('pathway_id', pathway.get('pathway_name', '')))
                    else:
                        all_pathways.add(getattr(pathway, 'pathway_id', ''))
        
        # Filter pathways if specified
        if pathway_names:
            all_pathways = all_pathways.intersection(set(pathway_names))
        
        pathway_names_list = sorted(list(all_pathways))[:max_pathways]
        heatmap_matrix = []
        
        # Build heatmap matrix
        for pathway_id in pathway_names_list:
            row = []
            for omics_type in omics_types:
                score = 0.0
                if isinstance(omics_data.get(omics_type), dict) and 'pathways' in omics_data[omics_type]:
                    for pathway in omics_data[omics_type]['pathways']:
                        pid = pathway.get('pathway_id', pathway.get('pathway_name', '')) if isinstance(pathway, dict) else getattr(pathway, 'pathway_id', '')
                        if pid == pathway_id:
                            score = pathway.get('enrichment_score', 0.0) if isinstance(pathway, dict) else getattr(pathway, 'enrichment_score', 0.0)
                            break
                row.append(score)
            heatmap_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix,
            x=omics_types,
            y=pathway_names_list,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         'Omics Type: %{x}<br>' +
                         'Enrichment Score: %{z:.3f}<br>' +
                         '<extra></extra>',
            colorbar=dict(title="Enrichment Score")
        ))
        
        fig.update_layout(
            title="Multi-Omics Pathway Enrichment Heatmap",
            xaxis_title="Omics Type",
            yaxis_title="Pathway",
            height=max(600, len(pathway_names_list) * 15),
            width=max(800, len(omics_types) * 150)
        )
        
        return fig
    
    def create_multi_omics_network(
        self,
        network: Dict[str, Any],
        layout: str = "spring"
    ) -> go.Figure:
        """
        Create multi-omics network visualization.
        
        Args:
            network: Network structure with nodes, edges, and node_attributes
            layout: Network layout algorithm ('spring', 'circular', 'kamada_kawai')
            
        Returns:
            Plotly figure with network
        """
        nodes = network.get('nodes', [])
        edges = network.get('edges', [])
        node_attrs = network.get('node_attributes', {})
        
        if not NETWORKX_AVAILABLE:
            self.logger.warning("NetworkX not available, using simplified layout")
            return self._create_simple_network(nodes, edges, node_attrs)
        
        # Create network layout
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Choose layout algorithm
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Extract positions
        x_nodes = [pos[node][0] for node in nodes]
        y_nodes = [pos[node][1] for node in nodes]
        
        # Get node colors and sizes based on omics activity
        node_colors = []
        node_sizes = []
        
        for node in nodes:
            attrs = node_attrs.get(node, {})
            active_omics = attrs.get('active_omics', [])
            node_colors.append(len(active_omics))
            node_sizes.append(10 + attrs.get('omics_count', 0) * 5)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='lightgray'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Number of Active Omics")
            ),
            text=[node[:20] + '...' if len(node) > 20 else node for node in nodes],
            textposition="middle center",
            hovertemplate='%{text}<br>Omics Count: %{marker.color}<br><extra></extra>',
            name='Pathways'
        ))
        
        fig.update_layout(
            title="Multi-Omics Pathway Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_multi_omics_sankey(
        self,
        mapping: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """
        Create Sankey diagram for multi-omics data flow.
        
        Args:
            mapping: Cross-omics pathway mapping dictionary
            
        Returns:
            Plotly figure with Sankey diagram
        """
        # Prepare Sankey data
        omics_types = set()
        pathways = []
        
        for pathway_id, pathway_data in mapping.items():
            pathways.append(pathway_id)
            omics_types.update(pathway_data.get('active_omics', []))
        
        omics_types = sorted(list(omics_types))
        node_labels = omics_types + pathways
        
        # Create links
        sources = []
        targets = []
        values = []
        
        for i, pathway_id in enumerate(pathways):
            pathway_data = mapping[pathway_id]
            active_omics = pathway_data.get('active_omics', [])
            coverage_score = pathway_data.get('coverage_score', 0.0)
            
            for omics_type in active_omics:
                sources.append(omics_types.index(omics_type))
                targets.append(len(omics_types) + i)
                values.append(coverage_score * 10)  # Scale for visibility
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color="lightblue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(0,0,255,0.3)"
            )
        )])
        
        fig.update_layout(
            title="Multi-Omics Data Flow (Sankey Diagram)",
            font_size=10,
            height=800
        )
        
        return fig
    
    def _create_simple_network(
        self,
        nodes: List[str],
        edges: List[tuple],
        node_attrs: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create a simple network layout without NetworkX."""
        n_nodes = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        
        x_nodes = np.cos(angles)
        y_nodes = np.sin(angles)
        
        node_colors = []
        node_sizes = []
        
        for node in nodes:
            attrs = node_attrs.get(node, {})
            node_colors.append(len(attrs.get('active_omics', [])))
            node_sizes.append(10 + attrs.get('omics_count', 0) * 5)
        
        fig = go.Figure()
        
        # Add edges (simplified)
        for edge in edges[:50]:  # Limit edges for performance
            if edge[0] in nodes and edge[1] in nodes:
                i0, i1 = nodes.index(edge[0]), nodes.index(edge[1])
                fig.add_trace(go.Scatter(
                    x=[x_nodes[i0], x_nodes[i1], None],
                    y=[y_nodes[i0], y_nodes[i1], None],
                    mode='lines',
                    line=dict(width=1, color='lightgray'),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Number of Active Omics")
            ),
            text=[node[:20] + '...' if len(node) > 20 else node for node in nodes],
            textposition="middle center",
            hovertemplate='%{text}<br>Omics Count: %{marker.color}<br><extra></extra>',
            name='Pathways'
        ))
        
        fig.update_layout(
            title="Multi-Omics Pathway Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            showlegend=False
        )
        
        return fig



