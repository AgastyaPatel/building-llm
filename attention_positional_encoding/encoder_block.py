"""
Transformer Encoder Block

This module implements a single transformer encoder block consisting of:
1. Multi-head self-attention
2. Position-wise feed-forward network
3. Residual connections and layer normalization around each sub-layer
"""

import torch
import torch.nn as nn
from multi_head import MultiHeadSelfAttention
from feed_forward import FeedForward
from layer_norm import SublayerConnection


class TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder block.
    
    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed forward hidden dimension
        dropout: dropout rate
        trace_shapes: whether to trace tensor shapes for debugging
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        trace_shapes: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.trace_shapes = trace_shapes
        
        # Multi-head self-attention (encoder mode - no causal masking)
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            trace_shapes=trace_shapes,
            causal=False  # Encoder uses bidirectional attention
        )
        
        # Position-wise feed forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Residual connections with layer normalization
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the encoder block.
        
        Args:
            x: input tensor (batch_size, seq_len, d_model)
            mask: attention mask (batch_size, seq_len) - True for positions to mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        if self.trace_shapes:
            print(f"EncoderBlock input: {tuple(x.shape)}")
            
        # Self-attention with residual connection
        x = self.sublayer1(x, lambda x: self.self_attention(x))
        
        if self.trace_shapes:
            print(f"After self-attention: {tuple(x.shape)}")
            
        # Feed forward with residual connection
        x = self.sublayer2(x, lambda x: self.feed_forward(x))
        
        if self.trace_shapes:
            print(f"EncoderBlock output: {tuple(x.shape)}")
            
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N transformer encoder blocks.
    
    Args:
        n_layers: number of encoder blocks
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed forward hidden dimension
        dropout: dropout rate
        trace_shapes: whether to trace tensor shapes for debugging
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        trace_shapes: bool = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                trace_shapes=trace_shapes
            )
            for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through all encoder blocks.
        
        Args:
            x: input tensor (batch_size, seq_len, d_model)
            mask: attention mask (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
