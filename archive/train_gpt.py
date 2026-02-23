import pandas as pd
import numpy as np

import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import tiktoken

from dataclasses import dataclass




class MyMultiheadAttention(nn.Module):
    """
    torch.nn.MultiheadAttentionと類似のモジュールのスクラッチ実装.
    デフォルトでcausal maskが適用されるなど,
    オフィシャルのtorch.nn.MultiheadAttentionを正確に再現したわけではない.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        # 各ヘッドの次元
        self.head_dim = cfg.embed_dim // cfg.num_heads
        assert cfg.embed_dim % cfg.num_heads == 0, 'ヘッド数は埋め込み次元の約数'
        # projection layers
        self.proj_query = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.add_bias_qkv)
        self.proj_key = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.add_bias_qkv)
        self.proj_value = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.add_bias_qkv)
        self.proj_o = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        # attention weightのDropout
        self.attn_do = nn.Dropout(cfg.attn_dropout)
        # mask定数テンソルをバッファ
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(cfg.max_len, cfg.max_len)).reshape(1, 1, cfg.max_len, cfg.max_len)
        )

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        queries = self.proj_query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        keys = self.proj_key(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        values = self.proj_value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # attention score (logits)
        attn_logits = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.embed_dim)
        # causal maskの適用
        attn_logits = attn_logits.masked_fill(self.causal_mask[:,:,:seq_len,:seq_len] == 0, float("-inf"))

        # attention weights
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_do(attn_weights)
        # weighted sum of values
        z = torch.matmul(attn_weights, values)
        z = z.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        o = self.proj_o(z)

        return o

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Block = Attention + FFN
class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn = MyMultiheadAttention(cfg)
        self.ffn = MLP(cfg)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.resid_do1 = nn.Dropout(cfg.resid_dropout)
        self.resid_do2 = nn.Dropout(cfg.resid_dropout)

    def forward(self, x):
        x = x + self.resid_do1(self.attn(self.norm1(x)))
        x = x + self.resid_do2(self.ffn(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.embed_dim)
        self.emb_do = nn.Dropout(cfg.embed_dropout)

        self.blocks = nn.Sequential(
            *[GPTBlock(cfg) for _ in range(cfg.num_layers)]
        )

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, idx):
        batch_size, seq_len = idx.size(0), idx.size(1)
        # Stem
        x = self.emb(idx)
        x = x + self.pos_emb(torch.arange(seq_len, device=idx.device))
        x = self.emb_do(x)
        # Body
        x = self.blocks(x)
        # Head
        x = self.norm(x)
        logits = self.head(x)

        return logits

class NIPSDataset(Dataset):
    def __init__(self, texts_df, tokenizer, max_len, stride):
        self.chunks = []
        for s in texts_df:
            text = tokenizer.encode(s)
            for i in range(0, len(text)-max_len, stride):
                self.chunks.append(torch.tensor(text[i:i+max_len+1]))

    def __len__(self,):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx][:-1], self.chunks[idx][1:]



def calc_cross_entropy(xs, ys, model, device):
    xs = xs.to(device)
    ys = ys.to(device)
    preds = model(xs)
    loss = F.cross_entropy(preds.flatten(0,1), ys.flatten())
    return loss


# 検証データで評価するための関数
def model_eval(model, val_loader, device, num_iters):
    model.eval()
    with torch.no_grad():
        val_loss = []
        for i, (xs, ys) in enumerate(val_loader):
            if i < min(len(val_loader), num_iters):
                loss = calc_cross_entropy(xs, ys, model, device)
                val_loss.append(loss.item())
            else:
                break
    model.train()
    return np.mean(val_loss)

def train(model, train_loader, val_loader, optimizer, num_epoches, device, num_iters):
    train_losses = []
    val_losses = []

    model.train()

    for epoch in range(num_epoches):
        print('epoch', epoch)
        losses = []
        for xs, ys in train_loader:
            optimizer.zero_grad()
            loss = calc_cross_entropy(xs, ys, model, device)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)
        train_losses.append(train_loss)
        val_loss = model_eval(model, val_loader, device, num_iters)
        val_losses.append(val_loss)
        print('train loss', train_loss, ':', 'val loss', val_loss)

    return train_losses, val_losses



if __name__ == '__main__':

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    # configurationのインスタンス
    # GPT2 (124M)
    GPT2_CONFIG = GPTConfig(num_layers=12, num_heads=12, embed_dim=768) # 今回はこれを使う


    data = pd.read_csv('./data/Papers.csv')
    paper_text = data['PaperText']

    text_lens = []

    for i in range(len(paper_text)):
        text_lens.append(len(paper_text.iloc[i]))

    tokenizer = tiktoken.get_encoding('gpt2')

    first_text = paper_text.iloc[0]
    first_text_ids = tokenizer.encode(first_text)

    first_text_decoded = tokenizer.decode(first_text_ids)

    dataset = NIPSDataset(paper_text, tokenizer, 64, 64)
    data_iter = iter(dataset)

    x, y = next(data_iter)

    BATCH_SIZE = 2
    # GPTのconfigを小型の入力に改変
    MAX_LEN = 256
    GPT2_CONFIG.max_len = MAX_LEN

    text_df_train = paper_text.iloc[:-20]
    text_df_val = paper_text.iloc[-20:]

    train_dataset = NIPSDataset(text_df_train, tokenizer, MAX_LEN, MAX_LEN)
    val_dataset = NIPSDataset(text_df_val, tokenizer, MAX_LEN, MAX_LEN)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count()
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=os.cpu_count()
    )

    OPTIMIZER_CONFIG = OptimizerConfig()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # サーバでは'cuda:0'など

    model = GPTModel(GPT2_CONFIG)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = OPTIMIZER_CONFIG.learning_rate,
        weight_decay = OPTIMIZER_CONFIG.weight_decay
    )

    num_epoches = 8
    num_iters = 5

    train_losses, val_losses = train(model, train_dataloader, val_dataloader, optimizer, num_epoches, device, num_iters)