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
