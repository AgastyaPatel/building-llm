"""
Position-wise Feed Forward Network

This implements the feed forward network used in transformer blocks.
It consists of two linear transformations with a GELU activation in between.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    
    Args:
        d_model: input/output dimension
        d_ff: hidden dimension (typically 4 * d_model)
        dropout: dropout rate
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # First linear transformation + GELU activation
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.linear2(x)
        return x
