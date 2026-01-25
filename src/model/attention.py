import numpy as np
import torch
import torch.nn as nn 
import math

class CausalSelfAttention(nn.Module): 
    def __init__(self, config) -> None: 
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_models = config.d_models

        self.W_qkv = nn.Linear(self.d_models, 3 * self.d_models, config.bias)

        self.W_out = nn.Linear(self.d_models, self.d_models, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

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


class Attention:
    pass


