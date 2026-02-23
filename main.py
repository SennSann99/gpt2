import os
import torch
import pandas as pd
import tiktoken
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainingConfig
from src.model import GPTModel
from src.dataset import TextChunkDataset
from src.trainer import Trainer

def main():
    # 1. Setup Configuration & Device
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    df = pd.read_csv(train_cfg.data_path)
    paper_text = df['PaperText']
    
    train_texts = paper_text.iloc[:-20]
    val_texts = paper_text.iloc[-20:]

    tokenizer = tiktoken.get_encoding('gpt2')

    train_dataset = TextChunkDataset(train_texts, tokenizer, model_cfg.max_len, model_cfg.max_len)
    val_dataset = TextChunkDataset(val_texts, tokenizer, model_cfg.max_len, model_cfg.max_len)

    num_workers = os.cpu_count() or 0
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    # 3. Model & Optimizer Initialization
    model = GPTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg.learning_rate, 
        weight_decay=train_cfg.weight_decay
    )

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, optimizer, device, train_cfg)
    train_losses, val_losses = trainer.train()
    
    print("Training complete!")

if __name__ == '__main__':
    main()