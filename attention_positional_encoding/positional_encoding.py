"""
Positional encoding (absolute learned + sinusoidal)

Positional encoding is a way to add information about the position of the tokens in the sequence.
we basically add a random position vector to the embedding of the tokens.
"""
import math
import torch, torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    """
    d_model: embedding dimension
    max_len: maximum length of the sequence
    """
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        pos_emb = self.emb(pos)
        return x + pos_emb.unsqueeze(0)

class SinusoidalPositionalEncoding(nn.Module):
    """
    sinusoidal positional encoding from the paper "Attention is all you need"
    """
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000)/ d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # this is a buffer which becomes a parameter of the model without gradients

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        return x + self.pe[:T, :].unsqueeze(0)