"""
Layer Normalization and Residual Connection Components

This module provides layer normalization and residual connection wrappers
for transformer components. Supports both pre-norm and post-norm configurations.
"""

import torch
import torch.nn as nn
from typing import Callable


class LayerNorm(nn.Module):
    """
    Layer normalization with optional bias.
    
    Args:
        d_model: dimension to normalize over
        eps: small value to avoid division by zero
        bias: whether to include bias parameter
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            normalized: (batch_size, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        normalized = normalized * self.weight
        if self.bias is not None:
            normalized = normalized + self.bias
        return normalized


class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization.
    Supports both pre-norm and post-norm configurations.
    
    Args:
        d_model: model dimension
        dropout: dropout rate
        pre_norm: whether to apply layer norm before the sublayer (True) or after (False)
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, pre_norm: bool = True):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        
    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Apply residual connection to any sublayer with the same size.
        
        Args:
            x: input tensor (batch_size, seq_len, d_model)
            sublayer: function that takes x and returns tensor of same shape
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        if self.pre_norm:
            # Pre-norm: LayerNorm -> Sublayer -> Dropout -> Residual
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            # Post-norm: Sublayer -> Dropout -> Residual -> LayerNorm
            return self.norm(x + self.dropout(sublayer(x)))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    This is a convenience wrapper that combines residual connection and layer norm.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))
