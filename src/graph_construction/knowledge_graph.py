"""
Knowledge Graph Construction - Build company relationship graphs.
"""

import logging
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class FinancialKnowledgeGraph:
    """
    Build and manage a financial knowledge graph with companies as nodes
    and relationships (supplier, customer, competitor) as edges.
    """
    
    EDGE_TYPES = ['supplier', 'customer', 'competitor']
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize knowledge graph.
        
        Args:
            config: Configuration dictionary with graph settings
        """
        self.config = config.get('graph', {})
        self.graph = nx.DiGraph()  # Directed graph for relationship direction
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}  # Node metadata
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        
    def add_node(
        self,
        company_id: str,
        company_name: str,
        embedding: np.ndarray = None,
        **kwargs
    ) -> None:
        """
        Add a company node to the graph.
        
        Args:
            company_id: Unique company identifier
            company_name: Human-readable company name
            embedding: Node embedding vector (from text encoder)
            **kwargs: Additional metadata
        """
        self.graph.add_node(company_id)
        
        if embedding is not None:
            self.node_embeddings[company_id] = embedding
        
        # Store metadata
        self.metadata[company_id] = {
            'name': company_name,
            'embedding_dim': embedding.shape[0] if embedding is not None else 0,
            **kwargs
        }
        
        logger.debug(f"Added node: {company_id} ({company_name})")
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0,
        **kwargs
    ) -> None:
        """
        Add a relationship edge between companies.
        
        Args:
            source: Source company ID
            target: Target company ID
            relation_type: Type of relationship ('supplier', 'customer', 'competitor')
            weight: Edge weight (strength of relationship)
            **kwargs: Additional edge attributes
            
        Raises:
            ValueError: If relation_type is invalid
        """
        if relation_type not in self.EDGE_TYPES:
            raise ValueError(f"relation_type must be one of {self.EDGE_TYPES}")
        
        if source not in self.graph:
            logger.warning(f"Source node {source} not in graph, adding it")
            self.add_node(source, source)
        
        if target not in self.graph:
            logger.warning(f"Target node {target} not in graph, adding it")
            self.add_node(target, target)
        
        self.graph.add_edge(
            source,
            target,
            relation_type=relation_type,
            weight=weight,
            **kwargs
        )
        
        self.edge_weights[(source, target)] = weight
        logger.debug(f"Added edge: {source} --[{relation_type}]--> {target}")
    
    def add_edges_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Add multiple edges from DataFrame.
        
        Expected columns: 'source', 'target', 'relation_type', 'weight'
        
        Args:
            df: DataFrame with edge information
        """
        for _, row in df.iterrows():
            self.add_edge(
                source=row['source'],
                target=row['target'],
                relation_type=row['relation_type'],
                weight=row.get('weight', 1.0)
            )
        
        logger.info(f"Added {len(df)} edges from DataFrame")
    
    def get_neighbors(
        self,
        company_id: str,
        relation_type: str = None,
        direction: str = 'both'
    ) -> Set[str]:
        """
        Get neighboring companies.
        
        Args:
            company_id: Company ID
            relation_type: Filter by relationship type (or None for all)
            direction: 'in', 'out', or 'both'
            
        Returns:
            Set of neighboring company IDs
        """
        neighbors = set()
        
        if direction in ['out', 'both']:
            for target, attrs in self.graph[company_id].items():
                if relation_type is None or attrs.get('relation_type') == relation_type:
                    neighbors.add(target)
        
        if direction in ['in', 'both']:
            for source in self.graph.predecessors(company_id):
                attrs = self.graph[source][company_id]
                if relation_type is None or attrs.get('relation_type') == relation_type:
                    neighbors.add(source)
        
        return neighbors
    
    def get_n_hop_neighbors(
        self,
        company_id: str,
        n_hops: int = 2,
        relation_filter: List[str] = None
    ) -> Dict[int, Set[str]]:
        """
        Get N-hop neighbors (multi-hop propagation).
        
        Args:
            company_id: Starting company ID
            n_hops: Number of hops
            relation_filter: Filter by relationship types
            
        Returns:
            Dictionary mapping hop level to set of company IDs
        """
        hops = {0: {company_id}}
        current_level = {company_id}
        
        for hop in range(1, n_hops + 1):
            next_level = set()
            for company in current_level:
                neighbors = self.get_neighbors(
                    company,
                    relation_type=None,  # Will filter after
                    direction='both'
                )
                next_level.update(neighbors)
            
            # Remove already visited
            next_level -= set().union(*hops.values())
            hops[hop] = next_level
            current_level = next_level
        
        logger.info(f"Found {sum(len(v) for v in hops.values())} nodes within {n_hops} hops of {company_id}")
        return hops
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with graph metrics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
        }
        
        # Edge type distribution
        edge_type_count = {}
        for _, _, attrs in self.graph.edges(data=True):
            relation_type = attrs.get('relation_type', 'unknown')
            edge_type_count[relation_type] = edge_type_count.get(relation_type, 0) + 1
        
        stats['edge_types'] = edge_type_count
        
        return stats
    
    def visualize_subgraph(
        self,
        company_id: str,
        n_hops: int = 1,
        save_path: str = None
    ) -> nx.DiGraph:
        """
        Get subgraph around a company for visualization.
        
        Args:
            company_id: Center company
            n_hops: Number of hops
            save_path: Optional path to save as GraphML
            
        Returns:
            Subgraph as DiGraph
        """
        neighbors_dict = self.get_n_hop_neighbors(company_id, n_hops)
        nodes_to_include = {company_id}
        for hop_nodes in neighbors_dict.values():
            nodes_to_include.update(hop_nodes)
        
        subgraph = self.graph.subgraph(nodes_to_include).copy()
        
        if save_path:
            nx.write_graphml(subgraph, save_path)
            logger.info(f"Saved subgraph to {save_path}")
        
        return subgraph
    
    def to_tensor_format(self, device: str = 'cpu'):
        """
        Convert graph to tensor format for GNN processing.
        
        Returns:
            Dictionary with node features and edge indices
        """
        # Node features: stack embeddings
        node_ids = sorted(self.graph.nodes())
        
        # Create node feature matrix
        node_features = []
        for node_id in node_ids:
            if node_id in self.node_embeddings:
                node_features.append(self.node_embeddings[node_id])
            else:
                # Use zero embedding if not available
                default_dim = next(iter(self.node_embeddings.values())).shape[0]
                node_features.append(np.zeros(default_dim))
        
        node_features = np.array(node_features)
        
        # Create edge indices pytorch format: [[source], [target]]
        edges = list(self.graph.edges())
        if edges:
            node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
            edge_indices = np.array([
                [node_id_to_idx[src] for src, _ in edges],
                [node_id_to_idx[tgt] for _, tgt in edges]
            ], dtype=np.int64)
        else:
            edge_indices = np.zeros((2, 0), dtype=np.int64)
        
        logger.info(f"Tensor format: nodes={node_features.shape}, edges={edge_indices.shape}")
        
        return {
            'node_features': node_features,
            'edge_indices': edge_indices,
            'node_ids': node_ids,
        }


class GraphDataBuilder:
    """
    Helper class to build knowledge graphs from various data sources.
    """
    
    @staticmethod
    def build_from_sec_filings(config: Dict[str, Any]) -> FinancialKnowledgeGraph:
        """
        Build graph from SEC filings (10-K, 10-Q).
        Placeholder for actual SEC parsing logic.
        
        Args:
            config: Configuration
            
        Returns:
            FinancialKnowledgeGraph instance
        """
        logger.warning("SEC filing parsing not implemented. Use SEC EDGAR or Vala-Fi API.")
        return FinancialKnowledgeGraph(config)
    
    @staticmethod
    def build_from_manual_data(
        config: Dict[str, Any],
        edges_df: pd.DataFrame
    ) -> FinancialKnowledgeGraph:
        """
        Build graph from manual relationship data.
        
        Args:
            config: Configuration
            edges_df: DataFrame with columns: source, target, relation_type, weight
            
        Returns:
            FinancialKnowledgeGraph instance
        """
        graph = FinancialKnowledgeGraph(config)
        
        # Add all unique nodes
        all_companies = set(edges_df['source']) | set(edges_df['target'])
        for company_id in all_companies:
            graph.add_node(company_id, company_name=company_id)
        
        # Add edges
        graph.add_edges_from_dataframe(edges_df)
        
        return graph
