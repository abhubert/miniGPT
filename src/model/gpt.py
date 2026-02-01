import torch
import torch.nn as nn 
import torch.nn.functional as fn
from .layers import *

class GPT(nn.Module):
    def __init__(
                self, 
                vocab_size: int,
                d_model: int,
                d_head: int,
                n_heads: int,
                d_feedforward: int,
                max_seq_len: int,
                n_layers:int,
                dropout:float=0.1,
                bias:bool=True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                d_head=d_head,
                n_heads=n_heads,
                d_feedforward=d_feedforward,
                max_seq_len=max_seq_len,
                dropout=dropout,
                bias=bias
            ) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.token_embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence Length {T} exceeds max {self.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        token_embedding = self.token_embeddings(idx)
        position_embedding = self.position_embeddings(pos)
        x = self.dropout(token_embedding + position_embedding)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:

            loss = fn.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):

            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = fn.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx