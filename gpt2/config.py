from dataclasses import dataclass

"""
1. The @dataclass Decorator
When you put @dataclass above a class, Python automatically writes all the tedious background code for you. 
You don't have to write an __init__ method, a representation method (__repr__), or equality checks. 
Python looks at your variables and handles the rest behind the scenes.

2. The slots=True Optimization
This is a specific performance feature (introduced in Python 3.10). 
Normally, Python classes store their attributes in a dynamic dictionary (__dict__), which takes up extra memory. 
Adding slots=True tells Python: "I am only going to use exactly these variables, lock it down." 
* The Benefit: It prevents the creation of the underlying dictionary, saving memory and making accessing these variables slightly faster.
"""

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
    checkpoint_path: str = "checkpoints/last.ckpt"
