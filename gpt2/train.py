import argparse
import os
import signal
import sys
import warnings
from pathlib import Path

import lightning.pytorch as pl
import torch
from datasets import load_dataset
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from gpt2.chat import format_chat_message
from gpt2.config import ModelConfig, TrainConfig
from gpt2.model import GPTLightning

# 1. Initialize the tokenizer globally so the tokenize_function can use it
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_conversation(
    messages: list[dict[str, str]], max_length: int
) -> dict[str, list[int]]:
    """Tokenize a complete conversation and train only on assistant messages."""
    input_ids: list[int] = []
    labels: list[int] = []

    for message in messages:
        role = message["role"]
        segment = format_chat_message(role, message["content"], tokenizer.eos_token)
        segment_ids = tokenizer.encode(segment, add_special_tokens=False)
        input_ids.extend(segment_ids)
        labels.extend(segment_ids if role == "assistant" else [-100] * len(segment_ids))

    supervised_positions = [i for i, label in enumerate(labels) if label != -100]
    if not supervised_positions:
        raise ValueError("Conversation does not contain an assistant message")

    # End the window at the newest assistant response. This preserves recent
    # history without producing an all-ignored target for a trailing long user turn.
    window_end = supervised_positions[-1] + 1
    window_start = max(0, window_end - max_length)
    input_ids = input_ids[window_start:window_end]
    labels = labels[window_start:window_end]
    attention_mask = [1] * len(input_ids)

    padding_length = max_length - len(input_ids)
    input_ids.extend([tokenizer.pad_token_id] * padding_length)
    attention_mask.extend([0] * padding_length)
    labels.extend([-100] * padding_length)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_function(examples, max_length: int):
    tokenized = [
        tokenize_conversation(messages, max_length) for messages in examples["messages"]
    ]
    return {
        key: [conversation[key] for conversation in tokenized]
        for key in ("input_ids", "attention_mask", "labels")
    }


def get_next_version(root_dir: str, prefix: str = "version_") -> int:
    root_path = Path(root_dir)
    if not root_path.exists():
        return 0
    existing_versions = []
    for d in root_path.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            try:
                v = int(d.name[len(prefix) :])
                existing_versions.append(v)
            except ValueError:
                continue
    return max(existing_versions) + 1 if existing_versions else 0


def parse_args() -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(
        description="Train a compact GPT-2 model (Lightning)"
    )

    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--val-rows", type=int, default=20)

    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps. Use -1 (default) to train until Ctrl+C.",
    )
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
    parser.add_argument("--checkpoint-path", default="checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")

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
        wandb=args.wandb,
    )
    return model_cfg, train_cfg


# グローバルで保存フラグを管理（重複阻止）
_SAVING_FINAL = False
warnings.filterwarnings("ignore", message=".*weights_only.*")


# Updated train function to accept the DataLoaders
def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> None:
    main_pid = os.getpid()
    pl.seed_everything(train_cfg.seed, workers=True)

    module = GPTLightning(model_cfg, train_cfg)

    base_ckpt_path = Path(train_cfg.checkpoint_path).resolve()
    version = get_next_version(str(base_ckpt_path))
    version_dir = base_ckpt_path / f"version_{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(version_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        enable_version_counter=False,
    )

    loggers = [CSVLogger(save_dir="logs", name="gpt2", version=version)]
    if train_cfg.wandb:
        loggers.append(WandbLogger(project="gpt2-training", name="train_test"))

    use_amp = train_cfg.amp and torch.cuda.is_available()
    precision = "16-mixed" if use_amp else "32-true"

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=-1,
        max_steps=train_cfg.max_steps,
        val_check_interval=train_cfg.eval_interval,
        limit_val_batches=train_cfg.eval_batches,
        logger=loggers,
        callbacks=[
            checkpoint_cb,
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=1),
        ],
        gradient_clip_val=train_cfg.grad_clip,
        precision=precision,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    def handle_signal(sig, frame):
        global _SAVING_FINAL
        if _SAVING_FINAL:
            return
        _SAVING_FINAL = True

        if os.getpid() != main_pid:
            os._exit(0)

        interrupted_path = version_dir / "interrupted.ckpt"
        print(f"\n[INFO] 学習を中断して保存しています: {interrupted_path.name}")
        sys.stdout.flush()

        try:
            trainer.save_checkpoint(str(interrupted_path), weights_only=True)
            print(f"[INFO] 保存が完了しました: {interrupted_path.name}")
        except Exception as e:
            print(f"[ERROR] 保存失敗: {e}")

        sys.stdout.flush()
        os._exit(0)

    signal.signal(signal.SIGINT, handle_signal)

    try:
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # ModelCheckpoint may not run when training stops before a validation event.
        # Always leave the completed version directory with a usable checkpoint.
        last_path = version_dir / "last.ckpt"
        if not last_path.is_file():
            trainer.save_checkpoint(str(last_path), weights_only=True)
            print(f"[INFO] Saved final checkpoint: {last_path}")

    except KeyboardInterrupt:
        if not _SAVING_FINAL:
            handle_signal(None, None)
    # Pass the DataLoaders directly into the fit method


def main() -> None:
    # 1. Parse arguments
    model_cfg, train_cfg = parse_args()

    # 2. Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk")
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    if train_cfg.limit_rows > 0:
        train_dataset = train_dataset.select(
            range(min(train_cfg.limit_rows, len(train_dataset)))
        )
    if train_cfg.val_rows > 0:
        val_dataset = val_dataset.select(range(min(train_cfg.val_rows, len(val_dataset))))

    # 3. Tokenize the datasets
    print("Tokenizing data...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"max_length": model_cfg.block_size},
        remove_columns=train_dataset.column_names,
    )
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"max_length": model_cfg.block_size},
        remove_columns=val_dataset.column_names,
    )

    # 4. Convert to PyTorch tensors
    tokenized_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    tokenized_val.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # 5. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        tokenized_train,
        shuffle=True,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        tokenized_val,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )

    # 6. Start training!
    if train_cfg.max_steps == -1:
        print("Starting unlimited training. Press Ctrl+C to save and stop.")
    else:
        print(f"Starting training for {train_cfg.max_steps} steps.")
    train(model_cfg, train_cfg, train_loader, val_loader)


if __name__ == "__main__":
    main()
