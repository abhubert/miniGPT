from dataclasses import dataclass

@dataclass 
class GPTConfig: 
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_feedforward: int = 3072
    dropout: float = 0.1
    bias: bool = False

    @property
    def d_head(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads