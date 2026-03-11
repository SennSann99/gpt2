# Compact GPT-2 (PyTorch Lightning)

A compact, research-minded GPT-2 implementation built for an AI graduate student workflow: clear internals, reproducible experiments, and a strong bias toward scalable training via PyTorch Lightning.

## Why this repo exists

- Learn the core mechanics of GPT-2 by reading real, minimal code.
- Iterate quickly with clean configs and an easy CLI.
- Keep scaling paths open (multi-GPU / distributed) without rewriting the training loop.

## What’s inside

- GPT-2 style model with causal attention implemented using `einsum`
- Clean dataclass configs for model + training
- PyTorch Lightning `Trainer` and `LightningModule`
- Weight tying
- AdamW param grouping + warmup
- Deterministic seeding + simple CSV logging

## Parallelism mindset

This project leans on PyTorch Lightning so you can scale from a laptop to multi-GPU setups with minimal code changes. The training loop stays clean while Lightning handles device placement, DDP setup, and mixed precision where available.

## Quick start

```bash
uv run python -m gpt2.train \
  --max-steps 200 \
  --eval-interval 20 \
  --batch-size 2
```

## Smoke run (fast check)

```bash
uv run python -m gpt2.train \
  --limit-rows 40 \
  --block-size 64 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 128 \
  --batch-size 2 \
  --max-steps 2 \
  --eval-interval 1 \
  --eval-batches 1
```

## Project structure

- `main.py`: convenience entrypoint (train + generate)
- `gpt2/config.py`: model/train configs
- `gpt2/data.py`: CSV -> token chunks -> dataloaders
- `gpt2/model.py`: GPT model + LightningModule
- `gpt2/train.py`: Lightning trainer entrypoint (recommended)
- `gpt2/generate.py`: text generation entrypoint

## Notes

- Expects a CSV with a `PaperText` column by default.
- Default dataset path: `data/Papers.csv`.
- Checkpoint is written to `checkpoints/last.ckpt`.
- Logs are written to `logs/` via `CSVLogger`.
