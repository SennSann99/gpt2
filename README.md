# Compact GPT-2 (PyTorch Lightning)

A minimal GPT-2 style training project with:
- clean dataclass configs
- PyTorch Lightning trainer/data module
- causal attention written with `einsum`
- weight tying
- AdamW param grouping + warmup
- simple CLI training entrypoint

## Quick start

```bash
uv run python pl_main.py \
  --max-steps 200 \
  --eval-interval 20 \
  --batch-size 2
```

## Smoke run (fast check)

```bash
uv run python pl_main.py \
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

- `pl_main.py`: thin entrypoint
- `gpt2/config.py`: model/train configs
- `gpt2/data.py`: CSV -> token chunks -> dataloaders
- `gpt2/model.py`: GPT model + LightningModule
- `gpt2/train.py`: Lightning trainer entrypoint

## Notes

- Expects a CSV with a `PaperText` column by default.
- Default dataset path: `data/Papers.csv`.
- Checkpoint is written to `checkpoints/gpt2.ckpt`.
