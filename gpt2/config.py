from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False


@dataclass(slots=True)
class TrainConfig:
    data_path: str = "data/Papers.csv"
    text_column: str = "PaperText"
    limit_rows: int = 0
    val_rows: int = 20
    batch_size: int = 2
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    seed: int = 1337
    num_workers: int = 0
    amp: bool = True
    checkpoint_path: str = "checkpoints/gpt2.ckpt"
