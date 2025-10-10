import torch, torch.nn as nn, torch.nn.functional as F
import math
from attention_mask import casual_mask 

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer
    This represents one multi head self attention block.

    Dimensions:
    x: (B, T, d_model)
    qkv: (B, T, 3 * d_model)   # making 3 copies of input
    view-> q,k,v: (B, T, n_head, d_head)
    swap-> (B, n_head, T, d_head)
    scores: (B, n_head, T, T) = q@ K^T / sqrt(d_head)
    weights: (B, n_head, T, T) = softmax(scores
    ctx: (B, n_head, T, d_head) = weights @ v
    merge-> (B, T, n_head, d_head) -> (B, T, d_model)

    Args:
        d_model (int): embedding dimension
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, trace_shapes: bool = False, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes
        self.causal = causal  # Whether to apply causal masking

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.trace_shapes:
            print(f"q: {tuple(q.shape)}")
            print(f"k: {tuple(k.shape)}")
            print(f"v: {tuple(v.shape)}")

        scale = 1.0 / math.sqrt(self.d_head)
        attention = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask only if specified (for decoder-style attention)
        if self.causal:
            mask = casual_mask(T, device=x.device)
            attention = attention.masked_fill(mask, float("-inf"))
            
        w = F.softmax(attention, dim=-1)
        w = self.dropout(w)
        ctx = torch.matmul(w, v)
        if self.trace_shapes:
            print(f"ctx: {tuple(ctx.shape)}")
        ctx = ctx.transpose(1, 2)
        # we need to make sure that the context is contiguous
        # because the view operation is not guaranteed to be contiguous
        # if it is not contiguous, the view operation will throw an error
        # contiguous means that the memory is in a contiguous block
        ctx = ctx.contiguous().view(B, T, C)
        out = self.proj(ctx)
        return out