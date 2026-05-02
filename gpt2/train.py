import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from gpt2.config import ModelConfig, TrainConfig
from gpt2.model import GPTLightning

# 1. Initialize the tokenizer globally so the tokenize_function can use it
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Combine the input and output into a single conversational string
    texts = [
        f"User: {inp}\nModel: {out}{tokenizer.eos_token}" 
        for inp, out in zip(examples['input'], examples['output'])
    ]
    
    # Tokenize the combined texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=128,       
        padding="max_length"  
    )
    
    # Create the labels for GPT-2 causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def parse_args() -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(
        description="Train a compact GPT-2 model (Lightning)"
    )

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

    parser.add_argument("--batch-size", type=int, default=8) # Updated default to 8 based on previous steps
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
    parser.add_argument("--checkpoint-path", default="checkpoints/last.ckpt")
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


# Updated train function to accept the DataLoaders
def train(model_cfg: ModelConfig, train_cfg: TrainConfig, train_loader: DataLoader, val_loader: DataLoader) -> None:
    pl.seed_everything(train_cfg.seed, workers=True)

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

    loggers = [CSVLogger(save_dir="logs", name="gpt2")]
    if train_cfg.wandb:
        loggers.append(WandbLogger(project="gpt2-training", name="train_test"))

    use_amp = train_cfg.amp and torch.cuda.is_available()
    precision = "16-mixed" if use_amp else "32-true"

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=train_cfg.max_steps,
        val_check_interval=train_cfg.eval_interval,
        limit_val_batches=train_cfg.eval_batches,
        logger=loggers,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="step")],
        #gradient_clip_val=train_cfg.grad_clip,
        gradient_clip_val=0.0, # Fused=Trueの場合、勾配クリッピングを無効化
        precision=precision,
        log_every_n_steps=1,
    )

    # Pass the DataLoaders directly into the fit method
    trainer.fit(
        module, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )


def main() -> None:
    # 1. Parse arguments
    model_cfg, train_cfg = parse_args()

    # 2. Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("arman-bd/guppylm-60k-generic")
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # 3. Tokenize the datasets
    print("Tokenizing data...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=val_dataset.column_names
    )

    # 4. Convert to PyTorch tensors
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 5. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(tokenized_train, shuffle=True, batch_size=train_cfg.batch_size)
    val_loader = DataLoader(tokenized_val, batch_size=train_cfg.batch_size)

    # 6. Start training!
    print("Starting training pipeline...")
    train(model_cfg, train_cfg, train_loader, val_loader)

if __name__ == "__main__":
    main()