import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """Self-attention mechanism with a causal mask for autoregressive generation."""
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        
        assert cfg.embed_dim % cfg.num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.proj_query = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.add_bias_qkv)
        self.proj_key = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.add_bias_qkv)
        self.proj_value = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.add_bias_qkv)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False) # Renamed proj_o
        
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(cfg.max_len, cfg.max_len)).reshape(1, 1, cfg.max_len, cfg.max_len)
        )

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        queries = self.proj_query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        keys = self.proj_key(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        values = self.proj_value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn_logits = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.embed_dim)
        attn_logits = attn_logits.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        z = torch.matmul(attn_weights, values)
        z = z.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(z)

class FeedForward(nn.Module): # Renamed from MLP for standard terminology
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module): # Renamed from GPTBlock
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ffn = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.resid_dropout1 = nn.Dropout(cfg.resid_dropout)
        self.resid_dropout2 = nn.Dropout(cfg.resid_dropout)

    def forward(self, x):
        x = x + self.resid_dropout1(self.attn(self.norm1(x)))
        x = x + self.resid_dropout2(self.ffn(self.norm2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.embed_dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        
        x = self.token_emb(input_ids)
        x = x + self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = self.drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        return self.head(x)