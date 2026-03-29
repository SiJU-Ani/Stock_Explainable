"""
Graph Neural Network Module - Propagate sentiment and events through knowledge graph.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GraphConvolutionLayer(nn.Module):
    """Single Graph Convolution layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize GC layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
        """
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: x_out = activation(A @ x @ W + b)
        
        Args:
            x: Node features (num_nodes, in_features)
            adj_matrix: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Output features (num_nodes, out_features)
        """
        out = torch.mm(adj_matrix, x)
        out = torch.mm(out, self.weight)
        if self.bias is not None:
            out += self.bias
        return out


class GCNModel(nn.Module):
    """Graph Convolutional Network for knowledge graph propagation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension for propagated features
            num_layers: Number of GC layers
            dropout: Dropout rate
            activation: Activation function ('relu', 'elu', 'gelu')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu
        
        # Build layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(GraphConvolutionLayer(input_dim, hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(GraphConvolutionLayer(hidden_dim, output_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GCN.
        
        Args:
            x: Node features (num_nodes, input_dim)
            adj_matrix: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Propagated node features (num_nodes, output_dim)
        """
        out = x
        
        for i, (layer, dropout) in enumerate(zip(self.layers[:-1], self.dropouts)):
            out = layer(out, adj_matrix)
            out = self.activation(out)
            out = dropout(out)
        
        # Output layer (no activation)
        out = self.layers[-1](out, adj_matrix)
        return out


class GATLayer(nn.Module):
    """Graph Attention layer for adaptive weighting of neighbors."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.3
    ):
        """
        Initialize GAT layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # Linear transformation
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * num_heads))
        # Attention weights
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Node features (num_nodes, in_features)
            adj_matrix: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Attended features (num_nodes, out_features * num_heads)
        """
        # Linear transformation
        h = torch.mm(x, self.W)  # (num_nodes, out_features * num_heads)
        
        # Attention computation (simplified single-head for demo)
        # In practice, use proper attention mechanism
        out = h[:, :self.out_features]
        
        # Apply adjacency masking
        out = out * (adj_matrix.sum(dim=1, keepdim=True).clamp(min=1))
        
        return out


class EventPropagationGNN(nn.Module):
    """
    GNN for propagating news events across company network.
    Captures second-order market impacts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize event propagation GNN.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        gnn_config = config.get('gnn', {})
        self.input_dim = gnn_config.get('input_dim', 768)
        self.hidden_dim = gnn_config.get('hidden_dim', 256)
        self.output_dim = gnn_config.get('output_dim', 128)
        self.num_layers = gnn_config.get('num_layers', 3)
        self.dropout = gnn_config.get('dropout', 0.3)
        self.activation = gnn_config.get('activation', 'relu')
        
        # Build GCN
        self.gcn = GCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            activation=self.activation
        )
        
        logger.info(f"EventPropagationGNN initialized: {self.input_dim} -> {self.output_dim}")
    
    def preprocess_adjacency(
        self,
        edge_indices: np.ndarray,
        num_nodes: int,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Convert edge indices to adjacency matrix with symmetric normalization.
        
        Args:
            edge_indices: Edge indices (2, num_edges)
            num_nodes: Number of nodes
            normalize: Whether to normalize
            
        Returns:
            Adjacency matrix as tensor
        """
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes)
        
        if edge_indices.shape[1] > 0:
            sources, targets = edge_indices[0], edge_indices[1]
            adj[sources, targets] = 1.0
            # Make symmetric
            adj = adj + adj.t()
            # Remove duplicates
            adj[adj > 1] = 1.0
        
        if normalize:
            # Symmetric normalization: D^{-1/2} A D^{-1/2}
            degree = adj.sum(dim=1)
            degree_inv_sqrt = torch.zeros_like(degree)
            degree_inv_sqrt[degree > 0] = 1.0 / torch.sqrt(degree[degree > 0])
            degree_mat = torch.diag(degree_inv_sqrt)
            adj = torch.mm(torch.mm(degree_mat, adj), degree_mat)
        
        # Add self-loops
        adj = adj + torch.eye(num_nodes)
        
        return adj
    
    def propagate(
        self,
        node_features: torch.Tensor,
        edge_indices: np.ndarray,
        num_hops: int = 2
    ) -> Dict[int, torch.Tensor]:
        """
        Propagate features through the graph.
        
        Args:
            node_features: Initial node features (num_nodes, input_dim)
            edge_indices: Edge indices (2, num_edges)
            num_hops: Number of propagation hops
            
        Returns:
            Dictionary mapping hop level to propagated features
        """
        num_nodes = node_features.shape[0]
        adj_matrix = self.preprocess_adjacency(edge_indices, num_nodes)
        
        # Ensure tensors are on same device
        device = next(self.gcn.parameters()).device
        node_features = node_features.to(device)
        adj_matrix = adj_matrix.to(device)
        
        # Forward pass
        propagated_features = self.gcn(node_features, adj_matrix)  # (num_nodes, output_dim)
        
        logger.info(f"Propagated features shape: {propagated_features.shape}")
        
        return {
            'final': propagated_features,
            'hops': num_hops
        }
    
    def get_node_impact(
        self,
        node_id: int,
        propagated_features: torch.Tensor,
        impact_type: str = 'norm'
    ) -> float:
        """
        Get impact score for a node (how much it changed).
        
        Args:
            node_id: Node index
            propagated_features: Propagated features
            impact_type: 'norm' (L2), 'mean', or 'max'
            
        Returns:
            Impact score
        """
        if impact_type == 'norm':
            impact = torch.norm(propagated_features[node_id]).item()
        elif impact_type == 'mean':
            impact = propagated_features[node_id].mean().item()
        elif impact_type == 'max':
            impact = propagated_features[node_id].max().item()
        else:
            impact = torch.norm(propagated_features[node_id]).item()
        
        return impact


class GNNTrainer:
    """Trainer for GNN model."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: GNN model
            config: Configuration
        """
        self.model = model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Optimizer
        lr = config.get('training', {}).get('learning_rate', 1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train_step(
        self,
        node_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_indices: np.ndarray
    ) -> float:
        """
        Single training step.
        
        Args:
            node_features: Input node features
            target_features: Target node features
            edge_indices: Edge indices
            
        Returns:
            Loss value
        """
        node_features = node_features.to(self.device)
        target_features = target_features.to(self.device)
        
        # Forward
        output = self.model.gcn(node_features, edge_indices)
        
        # Loss (MSE)
        loss = F.mse_loss(output, target_features)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
