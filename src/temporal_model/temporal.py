"""
Temporal Model Module - LSTM/Transformer for stock price direction prediction.
"""

import logging
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM model for temporal prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.3,
        bidirectional: bool = False,
        output_size: int = 1
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Feature dimension
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            output_size: Output dimension (1 for binary classification)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Dense layers for direction prediction
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"LSTM model: input={input_size}, hidden={hidden_size}, layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            
        Returns:
            Tuple of (output logits, (h_n, c_n))
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_output_size)
        
        # Dense layers
        mlp_out = self.fc1(last_output)
        mlp_out = F.relu(mlp_out)
        mlp_out = self.dropout(mlp_out)
        logits = self.fc2(mlp_out)
        output = self.sigmoid(logits)
        
        return output, (h_n, c_n)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism."""
    
    def __init__(self, hidden_dim: int):
        """Initialize temporal attention."""
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention.
        
        Args:
            x: Input (batch_size, seq_length, hidden_dim)
            
        Returns:
            Attended output
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        output = torch.matmul(weights, V)
        return output


class TransformerBlock(nn.Module):
    """Single Transformer block for temporal processing."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (seq_length, batch_size, d_model)
            
        Returns:
            Output same shape as input
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TemporalTransformer(nn.Module):
    """Transformer model for temporal prediction."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        output_size: int = 1
    ):
        """
        Initialize Temporal Transformer.
        
        Args:
            input_size: Input feature dimension
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            output_size: Output dimension
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding (simplified)
        self.pos_encoding = self._get_positional_encoding(d_model, max_seq_length=512)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_proj = nn.Linear(d_model, output_size)
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"Transformer model: input={input_size}, d_model={d_model}, layers={num_layers}")
    
    def _get_positional_encoding(
        self,
        d_model: int,
        max_seq_length: int = 512
    ) -> torch.Tensor:
        """Generate positional encoding."""
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch_size, seq_length, input_size)
            
        Returns:
            Output (batch_size, 1) with sigmoid activation
        """
        # Input projection
        x = self.input_proj(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        seq_length = x.size(1)
        device = x.device
        pos_enc = self.pos_encoding[:seq_length].unsqueeze(0).to(device)  # (1, seq_length, d_model)
        x = x + pos_enc
        
        # Transpose for transformer (expects seq_length first)
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Use last timestep
        x = x[-1]  # (batch_size, d_model)
        
        # Output projection
        output = self.output_proj(x)  # (batch_size, 1)
        output = self.sigmoid(output)
        
        return output


class HybridTemporalModel(nn.Module):
    """
    Hybrid model combining features from multiple sources:
    - Historical price data
    - Technical indicators
    - Sentiment scores (from Judge model)
    - Risk embeddings (from GNN)
    """
    
    def __init__(
        self,
        historical_dim: int,
        sentiment_dim: int,
        gnn_dim: int,
        model_type: str = 'lstm',
        config: Dict[str, Any] = None
    ):
        """
        Initialize hybrid temporal model.
        
        Args:
            historical_dim: Dimension of historical/technical features
            sentiment_dim: Dimension of sentiment scores
            gnn_dim: Dimension of GNN embeddings
            model_type: 'lstm' or 'transformer'
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.model_type = model_type
        
        # Feature fusion layer
        total_input_dim = historical_dim + sentiment_dim + gnn_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.fused_dim = 256
        
        # Build temporal model
        if model_type == 'lstm':
            lstm_config = config.get('temporal_model', {}).get('lstm', {})
            self.temporal = LSTMModel(
                input_size=self.fused_dim,
                hidden_size=lstm_config.get('hidden_size', 128),
                num_layers=lstm_config.get('num_layers', 2),
                dropout=lstm_config.get('dropout', 0.3),
                bidirectional=lstm_config.get('bidirectional', False)
            )
        else:  # transformer
            self.temporal = TemporalTransformer(
                input_size=self.fused_dim,
                d_model=256,
                num_heads=8,
                num_layers=3,
                d_ff=512,
                dropout=0.2
            )
        
        logger.info(f"Hybrid model created: {model_type} with feature fusion")
    
    def forward(
        self,
        historical: torch.Tensor,
        sentiment: torch.Tensor,
        gnn_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            historical: Historical/technical features (batch_size, seq_length, historical_dim)
            sentiment: Sentiment scores (batch_size, seq_length, sentiment_dim)
            gnn_embeddings: GNN embeddings (batch_size, seq_length, gnn_dim)
            
        Returns:
            Direction prediction (batch_size, 1)
        """
        # Fuse features along feature dimension
        fused = torch.cat([historical, sentiment, gnn_embeddings], dim=-1)  # (..., total_dim)
        
        # Apply fusion network on each timestep
        batch_size, seq_length = fused.shape[0], fused.shape[1]
        fused_flat = fused.view(-1, fused.shape[-1])
        fused_proj = self.feature_fusion(fused_flat)
        fused_proj = fused_proj.view(batch_size, seq_length, self.fused_dim)
        
        # Temporal model
        output = self.temporal(fused_proj)  # (batch_size, 1)
        
        return output
