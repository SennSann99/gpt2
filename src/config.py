from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    max_len: int = 256  # Reduced for training size
    num_heads: int = 12
    num_layers: int = 12
    embed_dim: int = 768
    add_bias_qkv: bool = False
    attn_dropout: float = 0.1
    embed_dropout: float = 0.1
    resid_dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    num_epochs: int = 8
    val_iters: int = 5
    data_path: str = './data/Papers.csv'