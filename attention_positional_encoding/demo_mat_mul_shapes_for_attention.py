"""
Walkthrough of multi-head attention with explicit matrix math and shapes
Generates a text log at ./out/mha_shapes.txt
"""
import os, math, torch
from multi_head import MultiHeadSelfAttention

OUT_TXT = os.path.join(os.path.dirname(__file__), "out", "mha_shapes.txt")

def log(s):
    print(s)
    with open(OUT_TXT, "a") as f:
        f.write(s + "\n")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, "w").close()

    # batch size, sequence length, embedding dimension, number of heads
    B, T, d_model, n_heads = 1, 4, 12, 3
    d_head = d_model // n_heads   # dimension of each head

    x = torch.randn(B, T, d_model)
    attn = MultiHeadSelfAttention(d_model, n_heads, trace_shapes=True)

    log(f"Input x:  {tuple(x.shape)} = (B, T, d_model)")
    qkv = attn.qkv(x)
    log(f"linear qkv: {tuple(qkv.shape)} = (B, T, 3 * d_model)")

    qkv = qkv.view(B, T, 3, n_heads, d_head)
    q, k, v = qkv.unbind(dim=2)
    log(f"q,k,v split: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    log(f"transposed heads: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

    # Computing Attention score Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) * V
    scale = 1.0 / math.sqrt(d_head)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    log(f"scores: {tuple(scores.shape)} = (B, n_heads, T, T)")

    weights = torch.softmax(scores, dim=-1)
    log(f"softmax(weights): {tuple(weights.shape)} = (B, n_heads, T, T)")

    ctx = torch.matmul(weights, v)
    log(f"context @v: {tuple(ctx.shape)} = (B, n_heads, T, d_head)")

    out = ctx.transpose(1, 2).contiguous().view(B, T, d_model)
    log(f"merged heads: {tuple(out.shape)} = (B, T, d_model)")

    out = attn.proj(out)
    log(f"final projected out: {tuple(out.shape)} = (B, T, d_model)")

    log("\nlegend:")
    log(" B=batch size")
    log(" T = sequence length")
    log(" d_model = embedding dimension")
    log(" n_heads = number of heads")
    log(" d_head = dimension of each head = d_model // n_heads")
    log(" d_model = n_heads * d_head")
    log(" q = query")
    log(" k = key")
    log(" v = value")
    log(" scores = qk^T")
    log(" qkv(x) is a single Linear producing [Q|K|V]; we reshape then split into q,k,v")