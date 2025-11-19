"""
Workflow visualization for PathwayLens.

Provides visualization of workflow structure, execution status, and dependencies.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    logger.warning("NetworkX or Matplotlib not available. Workflow visualization disabled.")


class NodeStatus(str, Enum):
    """Status of workflow nodes."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowNode:
    """Represents a node in the workflow graph."""
    node_id: str
    label: str
    node_type: str
    status: NodeStatus
    dependencies: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WorkflowVisualizer:
    """Visualize workflow structure and execution."""
    
    def __init__(self):
        """Initialize workflow visualizer."""
        self.logger = logger.bind(module="workflow_visualizer")
        if not HAS_VISUALIZATION:
            self.logger.warning("Visualization libraries not available")
    
    def visualize_workflow(
        self,
        workflow_spec: Dict[str, Any],
        execution_status: Optional[Dict[str, Any]] = None,
        output_file: Optional[Path] = None,
        format: str = "png"
    ) -> Optional[Dict[str, Any]]:
        """
        Visualize workflow structure.
        
        Args:
            workflow_spec: Workflow specification
            execution_status: Optional execution status information
            output_file: Optional output file path
            format: Output format ('png', 'svg', 'json')
            
        Returns:
            Visualization data or None if visualization unavailable
        """
        if not HAS_VISUALIZATION:
            self.logger.error("Visualization libraries not available")
            return None
        
        try:
            # Parse workflow into nodes and edges
            nodes, edges = self._parse_workflow(workflow_spec, execution_status)
            
            # Build graph
            graph = self._build_graph(nodes, edges)
            
            # Generate visualization
            if format == "json":
                return self._export_json(graph, nodes, edges)
            else:
                return self._export_image(graph, nodes, edges, output_file, format)
                
        except Exception as e:
            self.logger.error(f"Workflow visualization failed: {e}")
            return None
    
    def _parse_workflow(
        self,
        workflow_spec: Dict[str, Any],
        execution_status: Optional[Dict[str, Any]]
    ) -> tuple[List[WorkflowNode], List[tuple[str, str]]]:
        """Parse workflow specification into nodes and edges."""
        nodes = []
        edges = []
        
        steps = workflow_spec.get("steps", [])
        status_map = {}
        
        if execution_status:
            status_map = execution_status.get("step_status", {})
        
        for step in steps:
            step_id = step.get("id", step.get("step_id", ""))
            step_type = step.get("type", "unknown")
            step_name = step.get("name", step_id)
            depends_on = step.get("depends_on", [])
            
            # Determine status
            status = NodeStatus.PENDING
            if step_id in status_map:
                status_str = status_map[step_id].get("status", "pending")
                try:
                    status = NodeStatus(status_str)
                except ValueError:
                    status = NodeStatus.PENDING
            
            node = WorkflowNode(
                node_id=step_id,
                label=step_name,
                node_type=step_type,
                status=status,
                dependencies=depends_on,
                metadata=step.get("params", {})
            )
            nodes.append(node)
            
            # Add edges for dependencies
            for dep in depends_on:
                edges.append((dep, step_id))
        
        return nodes, edges
    
    def _build_graph(
        self,
        nodes: List[WorkflowNode],
        edges: List[tuple[str, str]]
    ) -> nx.DiGraph:
        """Build NetworkX graph from nodes and edges."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            G.add_node(
                node.node_id,
                label=node.label,
                node_type=node.node_type,
                status=node.status.value,
                metadata=node.metadata
            )
        
        # Add edges
        for source, target in edges:
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target)
        
        return G
    
    def _export_json(
        self,
        graph: nx.DiGraph,
        nodes: List[WorkflowNode],
        edges: List[tuple[str, str]]
    ) -> Dict[str, Any]:
        """Export workflow as JSON."""
        node_data = []
        for node in nodes:
            node_data.append({
                "id": node.node_id,
                "label": node.label,
                "type": node.node_type,
                "status": node.status.value,
                "dependencies": node.dependencies,
                "metadata": node.metadata
            })
        
        edge_data = [{"source": src, "target": tgt} for src, tgt in edges]
        
        return {
            "nodes": node_data,
            "edges": edge_data,
            "graph_properties": {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(graph)
            }
        }
    
    def _export_image(
        self,
        graph: nx.DiGraph,
        nodes: List[WorkflowNode],
        edges: List[tuple[str, str]],
        output_file: Optional[Path],
        format: str
    ) -> Dict[str, Any]:
        """Export workflow as image."""
        if graph.number_of_nodes() == 0:
            self.logger.warning("Empty workflow graph")
            return {}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate layout
        try:
            pos = nx.spring_layout(graph, k=2, iterations=50)
        except Exception:
            pos = nx.circular_layout(graph)
        
        # Color mapping for status
        status_colors = {
            NodeStatus.PENDING.value: "#CCCCCC",
            NodeStatus.RUNNING.value: "#FFA500",
            NodeStatus.COMPLETED.value: "#00FF00",
            NodeStatus.FAILED.value: "#FF0000",
            NodeStatus.SKIPPED.value: "#888888"
        }
        
        # Draw nodes
        node_colors = []
        for node in nodes:
            status = graph.nodes[node.node_id].get("status", "pending")
            node_colors.append(status_colors.get(status, "#CCCCCC"))
        
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=2000,
            alpha=0.9,
            ax=ax
        )
        
        # Draw labels
        labels = {node.node_id: node.label for node in nodes}
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            edge_color="#888888",
            arrows=True,
            arrowsize=20,
            alpha=0.6,
            ax=ax
        )
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=color, label=status.replace("_", " ").title())
            for status, color in status_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title("Workflow Visualization", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Save or return
        if output_file:
            plt.savefig(output_file, format=format, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Workflow visualization saved to {output_file}")
        else:
            # Return figure data (would need to convert to base64 for web)
            plt.close()
        
        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "format": format,
            "output_file": str(output_file) if output_file else None
        }
    
    def generate_workflow_summary(
        self,
        workflow_spec: Dict[str, Any],
        execution_status: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a text summary of the workflow.
        
        Args:
            workflow_spec: Workflow specification
            execution_status: Optional execution status
            
        Returns:
            Summary dictionary
        """
        steps = workflow_spec.get("steps", [])
        
        summary = {
            "total_steps": len(steps),
            "step_types": {},
            "dependencies": {},
            "execution_status": {}
        }
        
        # Count step types
        for step in steps:
            step_type = step.get("type", "unknown")
            summary["step_types"][step_type] = summary["step_types"].get(step_type, 0) + 1
        
        # Analyze dependencies
        for step in steps:
            step_id = step.get("id", step.get("step_id", ""))
            depends_on = step.get("depends_on", [])
            summary["dependencies"][step_id] = depends_on
        
        # Execution status
        if execution_status:
            status_map = execution_status.get("step_status", {})
            status_counts = {}
            for step_id, status_info in status_map.items():
                status = status_info.get("status", "pending")
                status_counts[status] = status_counts.get(status, 0) + 1
            summary["execution_status"] = status_counts
        
        return summary



