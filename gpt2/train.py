import argparse
from pathlib import Path

import lightning.pytorch as pl
import tiktoken
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from gpt2.config import ModelConfig, TrainConfig
from gpt2.data import GPTDataModule
from gpt2.model import GPTLightning


def parse_args() -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(description="Train a compact GPT-2 model (Lightning)")

    parser.add_argument("--data-path", default="data/Papers.csv")
    parser.add_argument("--text-column", default="PaperText")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--val-rows", type=int, default=20)

    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-path", default="checkpoints/gpt2.ckpt")

    args = parser.parse_args()

    model_cfg = ModelConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    train_cfg = TrainConfig(
        data_path=args.data_path,
        text_column=args.text_column,
        limit_rows=args.limit_rows,
        val_rows=args.val_rows,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        checkpoint_path=args.checkpoint_path,
    )
    return model_cfg, train_cfg


def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    pl.seed_everything(train_cfg.seed, workers=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    datamodule = GPTDataModule(train_cfg, model_cfg, tokenizer)
    module = GPTLightning(model_cfg, train_cfg)

    ckpt_path = Path(train_cfg.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_path.parent),
        filename=ckpt_path.stem,
        monitor="val_loss",
        mode="min",
        save_top_k=0,
        save_last=True,
    )
    logger = CSVLogger(save_dir="logs", name="gpt2")

    use_amp = train_cfg.amp and torch.cuda.is_available()
    precision = "16-mixed" if use_amp else "32-true"

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=train_cfg.max_steps,
        val_check_interval=train_cfg.eval_interval,
        limit_val_batches=train_cfg.eval_batches,
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step")],
        gradient_clip_val=train_cfg.grad_clip,
        precision=precision,
        log_every_n_steps=1,
    )

    trainer.fit(module, datamodule=datamodule)


def main() -> None:
    model_cfg, train_cfg = parse_args()
    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
