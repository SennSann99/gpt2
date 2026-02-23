import os
import torch
import pandas as pd
import tiktoken
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.model import GPTModel
from src.dataset import TextChunkDataset
from src.trainer import Trainer

VAL_SET_SIZE = 20


def load_data(train_cfg: TrainingConfig) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(train_cfg.data_path)
    paper_text = df["PaperText"]
    return paper_text.iloc[:-VAL_SET_SIZE], paper_text.iloc[-VAL_SET_SIZE:]


def build_dataloaders(
    train_texts: pd.Series,
    val_texts: pd.Series,
    tokenizer: tiktoken.Encoding,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    chunk_size = model_cfg.max_len
    num_workers = os.cpu_count() or 0

    train_dataset = TextChunkDataset(train_texts, tokenizer, chunk_size, chunk_size)
    val_dataset = TextChunkDataset(val_texts, tokenizer, chunk_size, chunk_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def main() -> None:
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_texts, val_texts = load_data(train_cfg)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader = build_dataloaders(train_texts, val_texts, tokenizer, model_cfg, train_cfg)

    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, optimizer, device, train_cfg)
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()