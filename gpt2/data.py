import os

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from gpt2.config import ModelConfig, TrainConfig


class TokenChunkDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int, stride: int | None = None):
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride or block_size
        self.n_chunks = max(0, (len(token_ids) - 1 - block_size) // self.stride + 1)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.token_ids[start : start + self.block_size]
        y = self.token_ids[start + 1 : start + self.block_size + 1]
        return x, y


def _flatten_texts_to_tokens(texts, tokenizer) -> torch.Tensor:
    eot = tokenizer.eot_token
    ids: list[int] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        ids.extend(tokenizer.encode_ordinary(text))
        ids.append(eot)
    if len(ids) < 2:
        raise ValueError("No usable text rows were found for tokenization.")
    return torch.tensor(ids, dtype=torch.long)


def build_dataloaders(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    tokenizer,
) -> tuple[DataLoader, DataLoader]:
    df = pd.read_csv(train_cfg.data_path)
    if train_cfg.text_column not in df.columns:
        available = ", ".join(df.columns)
        raise KeyError(f"Missing text column '{train_cfg.text_column}'. Columns: {available}")

    texts = df[train_cfg.text_column]
    if train_cfg.limit_rows > 0:
        texts = texts.iloc[: train_cfg.limit_rows]

    if len(texts) <= train_cfg.val_rows:
        raise ValueError(
            f"Need more than val_rows={train_cfg.val_rows} rows, got {len(texts)} rows."
        )

    train_texts = texts.iloc[:-train_cfg.val_rows]
    val_texts = texts.iloc[-train_cfg.val_rows :]

    train_tokens = _flatten_texts_to_tokens(train_texts, tokenizer)
    val_tokens = _flatten_texts_to_tokens(val_texts, tokenizer)

    train_ds = TokenChunkDataset(train_tokens, block_size=model_cfg.block_size)
    val_ds = TokenChunkDataset(val_tokens, block_size=model_cfg.block_size)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Dataset produced zero chunks. Lower block_size or provide more tokens."
        )

    workers = train_cfg.num_workers if train_cfg.num_workers > 0 else min(4, os.cpu_count() or 0)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, train_cfg: TrainConfig, model_cfg: ModelConfig, tokenizer) -> None:
        super().__init__()
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.tokenizer = tokenizer
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        self._train_loader, self._val_loader = build_dataloaders(
            self.train_cfg,
            self.model_cfg,
            self.tokenizer,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_loader is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        if self._val_loader is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._val_loader
