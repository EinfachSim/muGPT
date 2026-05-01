import torch
import math

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    emb_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12 
    seq_len: int = 1024 
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self):
        assert self.emb_dim % self.num_heads == 0, \
            f"emb_dim {self.emb_dim} must be divisible by num_heads {self.num_heads}"

class CausalAttentionBlock(torch.nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, seq_len: int, bias=False, dropout=0.15):
        super().__init__()
        self.c_attention_head = torch.nn.Linear(emb_dim, 3*emb_dim, bias = bias)

        self.attn_proj = torch.nn.Linear(emb_dim, emb_dim)

        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self.softmax = torch.nn.Softmax(dim=-1)
        self.attention_dropout = torch.nn.Dropout(dropout)

        
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
    def forward(self, x):

        batch_size, seq_len, d = x.size()

        #split output of attention into query, key and value heads
        q,k,v = self.c_attention_head(x).split(self.emb_dim, dim=2)

        #spread across multiple heads
        k = k.view(batch_size, seq_len, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2) # [batch_size, num_heads, seq_len, emb_dim // num_heads]
        q = q.view(batch_size, seq_len, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2) # [batch_size, num_heads, seq_len, emb_dim // num_heads]
        v = v.view(batch_size, seq_len, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2) # [batch_size, num_heads, seq_len, emb_dim // num_heads]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # [batch_size, num_heads, seq_len, seq_len]
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        att = self.softmax(att)
        att = self.attention_dropout(att)

        #computing outputs from values
        out = att @ v # [batch_size, num_heads, seq_len, emb_dim // num_heads]

        #putting it all back together
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_dim) # [batch_size, seq_len, emb_dim]
        out = self.attn_proj(out)
        return out
    
class FeedForwardBlock(torch.nn.Module):
    def __init__(self, in_dim: int, dropout=0.15):
        super().__init__()

        self.l1 = torch.nn.Linear(in_dim, 4*in_dim)
        self.gelu = torch.nn.GELU()
        self.l2 = torch.nn.Linear(4*in_dim, in_dim)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        #x of shape [batch_size, seq_len, in_dim]

        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return self.dropout(x)

class TransformerBlock(torch.nn.Module):
    def __init__(self, emb_dim:int, num_heads:int, seq_len:int, bias=False, dropout=0.15):
        super().__init__()

        self.attn_block = CausalAttentionBlock(emb_dim, num_heads, seq_len, bias, dropout)

        self.ff_block = FeedForwardBlock(emb_dim, dropout)

        self.layernorm1 = torch.nn.LayerNorm(emb_dim)
        self.layernorm2 = torch.nn.LayerNorm(emb_dim)

    def forward(self, x):

        x = x + self.attn_block(self.layernorm1(x))
        x = x + self.ff_block(self.layernorm2(x))

        return x

class DecoderOnlyTransformer(torch.nn.Module):
    def __init__(self, model_config):

        self.cfg = model_config
        
        super().__init__()

        self.token_emb = torch.nn.Embedding(self.cfg.vocab_size, self.cfg.emb_dim)

        self.pos_emb = torch.nn.Embedding(self.cfg.seq_len, self.cfg.emb_dim)

        self.blocks = torch.nn.ModuleList([TransformerBlock(self.cfg.emb_dim, self.cfg.num_heads, self.cfg.seq_len, self.cfg.bias, self.cfg.dropout) for _ in range(self.cfg.num_layers)])

        self.ln_f = torch.nn.LayerNorm(self.cfg.emb_dim)
        self.head = torch.nn.Linear(self.cfg.emb_dim, self.cfg.vocab_size, bias=False)

        self.head.weight = self.token_emb.weight #weight tying
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device)
        x = self.token_emb(x) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        #final layer norm and head
        x = self.ln_f(x)
        x = self.head(x)

        return x