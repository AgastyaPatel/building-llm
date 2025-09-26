"""
Single Head self attention for better understanding
"""

import torch, torch.nn as nn, torch.nn.functional as F
from attention_mask import casual_mask
import math

class SingleHeadSelfAttention(nn.Module):
    """
    Single-Head attention
    args:
        d_model: embedding dimension
        d_k: dimension of the key, value and query
        dropout: dropout rate
        trace_shapes: whether to trace the shapes of the tensors
    """
    def __init__(self, d_model: int, d_k: int, dropout: float = 0.0, trace_shapes: bool = False):
        super().__init__()
        self.d_k = d_k  # Store d_k as instance variable for use in forward
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # (B, T, d_model)
        B, T, C = x.shape
        q = self.q(x)    # (B, T, d_k)
        k = self.k(x)    # (B, T, d_k)
        v = self.v(x)    # (B, T, d_k)
        if self.trace_shapes:
            print(f"q: {tuple(q.shape)}")
            print(f"k: {tuple(k.shape)}")
            print(f"v: {tuple(v.shape)}")

        scale = 1.0 / math.sqrt(self.d_k)
        attention = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, T, T)
        mask = casual_mask(T, device=x.device)
        attention = attention.masked_fill(mask.squeeze(1), float("-inf"))
        w = F.softmax(attention, dim=-1)
        w = self.dropout(w)
        out = torch.matmul(w, v)   # (B, T, d_k)
        if self.trace_shapes:
            print(f"Weights: {tuple(w.shape)} Out {out.shape}")
        return out, w