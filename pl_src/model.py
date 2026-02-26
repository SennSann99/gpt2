# src/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.num_head = cfg.num_head
        assert self.embed_dim % self.num_head == 0

        self.head_dim = self.embed_dim // self.num_head

        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.add_bias_qkv)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.add_bias_qkv)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.add_bias_qkv)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)

        # bool causal mask
        mask = torch.tril(torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, cfg.max_len, cfg.max_len)
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        # Project and reshape
        q = self.proj_q(x).view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        k = self.proj_k(x).view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        v = self.proj_v(x).view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        # shapes: (B, H, T, D)

        # Scaled dot-product attention using einsum
        # (B,H,T,D) x (B,H,T,D) -> (B,H,T,T)
        attn_logits = torch.einsum("bhtd,bhsd->bhts", q, k)
        attn_logits = attn_logits / math.sqrt(self.head_dim)

        mask = self.causal_mask[:, :, :T, :T]
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (B,H,T,T) x (B,H,T,D) -> (B,H,T,D)
        attn_output = torch.einsum("bhts,bhsd->bhtd", attn_weights, v)

        # Merge heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        out = self.out_proj(attn_output)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.resid_dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTLightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.embed_dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )

        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        B, T = input_ids.shape

        tok = self.token_emb(input_ids)
        pos = self.pos_emb(torch.arange(T, device=input_ids.device)).unsqueeze(0)

        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)  # (B,T,V)
        return logits

    def _lm_loss(self, logits, targets):
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.view(B * T, V),
            targets.view(B * T),
            ignore_index=-100,
        )

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)

        loss = self._lm_loss(logits, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss = self._lm_loss(logits, labels)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )