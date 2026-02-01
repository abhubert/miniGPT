import numpy as np
import torch
import torch.nn as nn 
import math

class CausalSelfAttention(nn.Module): 
    def __init__(self, 
                 n_heads: int, 
                 d_head: int, 
                 d_model: int, 
                 bias: bool, 
                 max_seq_len: int, 
                 dropout: float = 0.1) -> None: 
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_models = d_model

        self.W_qkv = nn.Linear(self.d_models, 3 * self.d_models, bias)

        self.W_out = nn.Linear(self.d_models, self.d_models, bias)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        qkv: torch.Tensor = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2))/math.sqrt(self.d_head)

        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_models)

        out = self.resid_dropout(self.W_out(out))

        return out


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

    pass

