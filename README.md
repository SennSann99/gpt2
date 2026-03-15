# Compact GPT-2 (PyTorch Lightning)

A minimal, research-oriented GPT-2 implementation designed for graduate-level AI work — clear internals, reproducible experiments, and clean scaling paths via PyTorch Lightning.

---

## Motivation

- Study the core mechanics of GPT-2 through readable, minimal code.
- Iterate quickly with clean dataclass configs and a straightforward CLI.
- Keep multi-GPU / distributed training paths open without rewriting the training loop.

---

## Features

| Component | Details |
|---|---|
| Model | GPT-2 style transformer with causal self-attention via `einsum` |
| Config | Typed dataclasses for model and training hyperparameters |
| Training | PyTorch Lightning `Trainer` + `LightningModule` |
| Optimizer | AdamW with parameter group weight decay + linear warmup |
| Tokenization | `tiktoken` (GPT-2 encoding) |
| Logging | CSV logger to `logs/`; TensorBoard-compatible |
| Reproducibility | Deterministic seeding |

---

## Project Structure

```
gpt2/
├── config.py      # Model and training config dataclasses
├── data.py        # CSV → token chunks → DataLoaders
├── model.py       # GPT model + LightningModule
├── train.py       # Lightning Trainer entrypoint (recommended)
└── generate.py    # Text generation entrypoint
main.py            # Convenience entrypoint (train + generate)
```

---

## Requirements

- Python ≥ 3.12
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- PyTorch ≥ 2.5, Lightning ≥ 2.6

Install dependencies:

```bash
uv sync
```

---

## Quick Start

```bash
uv run python -m gpt2.train \
  --max-steps 200 \
  --eval-interval 20 \
  --batch-size 2
```

### Smoke Test (fast sanity check)

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

---

## Data Format

- Expects a CSV file with a `PaperText` column (e.g., academic abstracts or full papers).
- Default dataset path: `data/Papers.csv`

---

## Outputs

| Artifact | Path |
|---|---|
| Latest checkpoint | `checkpoints/last.ckpt` |
| Training logs | `logs/` |

---

## Parallelism

Training is built on PyTorch Lightning, so scaling from a single CPU/GPU to multi-GPU DDP requires no changes to the training loop. Lightning handles device placement, DDP setup, and mixed precision automatically.
