import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from dataclasses import asdict

from gpt2.config import ModelConfig

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al., 2021.
    Precomputes cos/sin tables; applied to Q and K inside attention (not token embeddings).
    """
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        # θ_i = 1 / 10000^(2i / head_dim), shape: (head_dim // 2,)
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # shape: (max_seq_len,)
        positions = torch.arange(max_seq_len).float()
        # freqs[m, i] = m * θ_i, shape: (max_seq_len, head_dim // 2)
        freqs = torch.outer(positions, theta)
        # Duplicate so each pair of dimensions (2i, 2i+1) shares the same angle
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (max_seq_len, head_dim)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for the first `seqlen` positions."""
        return self.cos_cache[:seqlen], self.sin_cache[:seqlen]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Swap and negate the two halves of the last dimension for RoPE rotation.
    Example:
        Since x: (bsz, n_head, seqlen, head_dim), Let's say head_dim = 8,
        Step 1: x1 = x[..., :4] → [x0, x1, x2, x3]  (shape: 4)
        Step 2: x2 = x[..., 4:] → [x4, x5, x6, x7]  (shape: 4)
        Step 3: torch.cat([-x2, x1], dim=-1) → [-x4, -x5, -x6, -x7, x0, x1, x2, x3]  (shape: 8, same as original)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # Concatenates the given sequence of tensors
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply RoPE rotation to Q or K.
    Args:
        x:   (bsz, n_head, seqlen, head_dim)
        cos: (seqlen, head_dim)
        sin: (seqlen, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    """
    For position i in the first half (say i=0):
        result[0] = x0 * cos(mθ₀) + (-x4) * sin(mθ₀)
          = x0·cos − x4·sin    ✓ this is the rotation formula!
    For its partner at position i + d/2 (i=4):
        result[4] = x4 * cos(mθ₀) + x0 * sin(mθ₀)
          = x4·cos + x0·sin    ✓ the other half of the rotation!
    """
    return x * cos + _rotate_half(x) * sin  # x*cos produces [x1·cos, x2·cos], _rotate_half(x)*sin produces [-x2·sin, x1·sin]
    
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, cfg.block_size)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, channels = x.shape
        qkv = self.qkv(x) # So, the shape of the qkv tensor becomes (bsz, seqlen, 3 * channels)
        q, k, v = qkv.split(channels, dim=-1) # The .split(channels, dim=-1) function tells PyTorch to take that last dimension (which is currently 3 * channels wide) and chop it into equal chunks of size channels.

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V)
        cos, sin = self.rope(seqlen)
        q = apply_rotary_emb(q, cos, sin) 
        k = apply_rotary_emb(k, cos, sin) 

        att = torch.einsum("bhid,bhjd->bhij", q, k)
        att = att * (self.head_dim**-0.5)
        mask = self.causal_mask[:, :, :seqlen, :seqlen]
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = torch.einsum("bhij,bhjd->bhid", att, v)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, channels)
        return self.resid_dropout(self.proj(y))

"""
1. The Raw Attention Scores (att)
Right before this line, your code calculates the raw attention scores (often called logits) by taking the dot product of the Queries ($Q$) and Keys ($K$).
At this stage, att contains scores representing how much every token in your sequence "wants" to look at every other token.


2. The Inverted Mask (~mask)
If you look up in your __init__ function, self.causal_mask is created using torch.tril() (triangle-lower). 
This creates a matrix of booleans where the diagonal and everything below it is True, and everything above it is False.

The tilde (~) is PyTorch's logical NOT operator. By doing ~mask, you flip the boolean values. Now, the upper right triangle of the matrix becomes True. 
These True values spatially correspond to the "future" tokens in the sequence.


3. Masking with Negative Infinity (float("-inf"))
The .masked_fill() function looks at your raw att tensor and says: 
"Wherever the flipped mask is True (the future tokens), overwrite the existing score with negative infinity ($-\infty$)."
"""

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seqlen = idx.shape
        if seqlen > self.cfg.block_size:
            raise ValueError(f"Sequence length {seqlen} > block_size {self.cfg.block_size}")

        x = self.token_embedding(idx)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )


def lr_for_step(step: int, cfg) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    return cfg.learning_rate


class GPTLightning(pl.LightningModule):
    def __init__(self, model_cfg: ModelConfig, train_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = GPTModel(model_cfg)
        self.save_hyperparameters(
            {
                "model_cfg": asdict(model_cfg),
                "train_cfg": asdict(train_cfg),
            }
        )

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.model(idx, targets)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        x, y = batch
        _, loss = self.model(x, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        del batch_idx
        x, y = batch
        _, loss = self.model(x, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = build_optimizer(self.model, self.train_cfg)

        def _lr_lambda(step: int) -> float:
            return lr_for_step(step, self.train_cfg) / self.train_cfg.learning_rate

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }
