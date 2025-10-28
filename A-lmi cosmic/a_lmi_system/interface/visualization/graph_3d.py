"""
3D Knowledge Graph Visualization
Plotly/Viser-based interactive graph visualization
"""

import logging
import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class Graph3D:
    """
    3D Knowledge Graph visualizer.
    
    Displays Temporal Knowledge Graph (Tier 3) with:
    - Interactive 3D node layout
    - Temporal edge annotations
    - Color-coded entity types
    - Temporal filtering
    """
    
    def __init__(self, tkg_client):
        """
        Initialize 3D graph visualizer.
        
        Args:
            tkg_client: Neo4j TKG client
        """
        self.tkg_client = tkg_client
        
    def visualize_subgraph(self, query_result: List[Dict[str, Any]], title: str = "Knowledge Graph"):
        """
        Visualize subgraph with 3D layout.
        
        Args:
            query_result: Cypher query result from Neo4j
            title: Graph title
            
        Returns:
            Plotly figure
        """
        # Build NetworkX graph from Neo4j results
        G = nx.Graph()
        
        for record in query_result:
            # Add nodes
            if 'entity_id' in record:
                G.add_node(record['entity_id'], **record)
            
            # Add edges
            if 'source' in record and 'target' in record:
                G.add_edge(
                    record['source'],
                    record['target'],
                    relationship=record.get('rel_type', 'related'),
                    valid_from=record.get('valid_from'),
                    valid_to=record.get('valid_to')
                )
        
        # 3D spring layout
        pos = nx.spring_layout(G, dim=3, seed=42)
        
        # Extract node and edge traces
        node_trace = self._create_node_trace(G, pos)
        edge_trace = self._create_edge_trace(G, pos)
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z')
                ),
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40)
            )
        )
        
        return fig
    
    def _create_node_trace(self, G: nx.Graph, pos: Dict) -> go.Scatter3d:
        """Create 3D node trace."""
        nodes = list(G.nodes())
        
        x_vals = [pos[node][0] for node in nodes]
        y_vals = [pos[node][1] for node in nodes]
        z_vals = [pos[node][2] for node in nodes]
        
        node_info = [f"ID: {node}<br>Type: {G.nodes[node].get('entity_type', 'unknown')}" 
                      for node in nodes]
        
        return go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers+text',
            marker=dict(
                size=10,
                color='rgb(125, 0, 0)',
                line=dict(width=1, color='rgb(50, 50, 50)')
            ),
            text=nodes,
            textposition="middle center",
            hovertext=node_info,
            name='Entities'
        )
    
    def _create_edge_trace(self, G: nx.Graph, pos: Dict) -> go.Scatter3d:
        """Create 3D edge trace."""
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        return go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(width=2, color='rgb(125, 125, 125)'),
            hoverinfo='none',
            name='Relationships'
        )
    
    def show(self, fig: go.Figure):
        """Display graph in browser."""
        fig.show()

