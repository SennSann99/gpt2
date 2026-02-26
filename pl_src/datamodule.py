# src/datamodule.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pl_src.dataset import TextChunkDataset
from pl_src.config import TrainingConfig

class GPTDataModule(pl.LightningDataModule):
    def __init__(self, train_cfg: TrainingConfig, tokenizer, model_cfg):
        super().__init__()
        self.cfg = train_cfg
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg

    def setup(self, stage=None):
        df = pd.read_csv(self.cfg.data_path)
        texts = df["PaperText"]
        val_size = 20
        train_texts = texts.iloc[:-val_size]
        val_texts = texts.iloc[-val_size:]

        # create datasets
        chunk_size = self.model_cfg.max_len
        stride = chunk_size  # or less if you want overlapping windows
        train_ds = TextChunkDataset(train_texts, self.tokenizer, chunk_size, stride)
        val_ds = TextChunkDataset(val_texts, self.tokenizer, chunk_size, stride)

        self.train_ds = train_ds
        self.val_ds = val_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 0,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 0,
            drop_last=False,
        )