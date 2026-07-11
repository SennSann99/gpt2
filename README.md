<div align="center">

# Compact GPT-2 ⚡

**A compact GPT-2 + Mixture of Experts implementation for learning, research, and experimentation with PyTorch Lightning**

[日本語](README_jp.md) · [Documentation](docs/index.html) · [Problem Report](docs/problem-report.html)

</div>

---

## About this project

This project provides a readable GPT-2 implementation for studying the model's internals and running the complete training, evaluation, and text-generation workflow. PyTorch Lightning keeps experiment configuration, logging, checkpointing, and CPU/GPU execution straightforward.

### Highlights

| Area | Implementation |
|---|---|
| Model | GPT-2-style causal Transformer |
| Position encoding | Rotary Position Embedding (RoPE) |
| Feed-forward network | SwiGLU + Mixture of Experts (MoE) |
| MoE routing | Top-k routing, capacity control, auxiliary loss, and Router Z-Loss |
| Training | PyTorch Lightning `Trainer` / `LightningModule` |
| Dataset | Hugging Face Dataset `arman-bd/guppylm-60k-generic` |
| Logging | CSV + optional Weights & Biases |
| Checkpoints | `best.ckpt` / `last.ckpt` / `interrupted.ckpt` |
| UI | Interactive Streamlit chat interface |

---

## Contents

- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Running the project](#running-the-project)
- [Generating from a trained model](#generating-from-a-trained-model)
- [Launching the chat UI](#launching-the-chat-ui)
- [Configuring WandB](#configuring-wandb)
- [Checkpoints and logs](#checkpoints-and-logs)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick start

On a Linux server with an NVIDIA GPU, the following commands build the Docker image, train the model, load the resulting checkpoint, and generate text:

```bash
git clone https://github.com/SennSann99/gpt2.git
cd gpt2
git switch sen/moe
chmod +x run.sh frontend/run_app.sh open_docs.sh
./run.sh gpu
```

To enable WandB, complete the [WandB configuration](#configuring-wandb) first, then run:

```bash
./run.sh gpu wandb
```

> [!NOTE]
> The first run can take some time because Docker dependencies, the GPT-2 tokenizer, and the training dataset must be downloaded.

---

## Requirements

### Docker workflow (recommended)

- Git
- Docker
- For NVIDIA GPU execution:
  - NVIDIA Driver
  - NVIDIA Container Toolkit

Confirm that Docker can access the GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Local workflow

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- PyTorch 2.5 or newer
- PyTorch Lightning 2.6 or newer

Install the locked dependencies:

```bash
uv sync --frozen
```

> [!IMPORTANT]
> An internet connection is required on the first run to download the dataset and tokenizer from Hugging Face.

---

## Running the project

### 1. Run with `run.sh` (recommended)

`run.sh` automates the Docker build and container startup. Its options can be combined in any order.

| Option | Description |
|---|---|
| `gpu` | Enable NVIDIA GPU support |
| `wandb` | Send training metrics to Weights & Biases |
| `tmux` | Run inside a tmux session named `gpt2` |

```bash
# Train on CPU, then generate text
./run.sh

# Train on GPU, then generate text
./run.sh gpu

# GPU + WandB
./run.sh gpu wandb

# GPU + WandB + tmux
./run.sh gpu wandb tmux
```

Training options can be passed through the same script:

```bash
./run.sh gpu \
  --max-steps 200 \
  --eval-interval 20 \
  --batch-size 2
```

The following host directories persist after the container exits:

```text
checkpoints/    Trained model checkpoints
logs/           CSV training logs
data/           Local data workspace
```

### 2. Run a short smoke test

Use a small model and a few training steps to verify the training pipeline:

```bash
./run.sh gpu \
  --block-size 64 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 128 \
  --batch-size 2 \
  --max-steps 2 \
  --eval-interval 1 \
  --eval-batches 1
```

### 3. Train without Docker

```bash
uv run python -m gpt2.train \
  --max-steps 200 \
  --eval-interval 20 \
  --batch-size 2
```

### 4. Use Docker directly

```bash
docker build -t gpt2 .

docker run --rm --init -it --gpus all \
  -v "$PWD/logs:/workspace/logs" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  -v "$PWD/data:/workspace/data" \
  gpt2
```

---

## Generating from a trained model

Pass the checkpoint directory to automatically select the newest valid checkpoint:

```bash
uv run python -m gpt2.generate \
  --checkpoint-path checkpoints \
  --prompt "The future of artificial intelligence"
```

To use a specific checkpoint, pass its complete path:

```bash
uv run python -m gpt2.generate \
  --checkpoint-path checkpoints/version_0/best.ckpt \
  --prompt "The future of artificial intelligence" \
  --max-new-tokens 128
```

Automatic selection searches version directories from newest to oldest using this priority:

1. `best.ckpt`
2. `last.ckpt`
3. `interrupted.ckpt`

If the newest version directory is empty, the loader automatically falls back to an older valid checkpoint.

---

## Launching the chat UI

After training, launch the Streamlit chat interface in Docker:

```bash
frontend/run_app.sh checkpoints
```

Open the following URL in a browser:

```text
http://localhost:8501
```

To load a specific checkpoint:

```bash
frontend/run_app.sh checkpoints/version_0/best.ckpt
```

> [!NOTE]
> The current `frontend/run_app.sh` uses `--gpus all`, so it requires an NVIDIA GPU and NVIDIA Container Toolkit.

---

## Configuring WandB

1. Create an account at [Weights & Biases](https://wandb.ai/).
2. Copy your API key from the WandB settings page.
3. Create a `.env` file in the project root:

```dotenv
WANDB_ENTITY=your_entity_name
WANDB_API_KEY=your_api_key
```

4. Run with the `wandb` option:

```bash
./run.sh gpu wandb
```

> [!WARNING]
> Never commit `.env`. It is already excluded by this repository's `.gitignore`.

---

## Checkpoints and logs

Each training run creates a new `version_N` directory:

```text
checkpoints/
├── version_0/
│   ├── best.ckpt
│   └── last.ckpt
└── version_1/
    ├── best.ckpt
    └── last.ckpt

logs/
└── gpt2/
    ├── version_0/
    └── version_1/
```

| File | Purpose |
|---|---|
| `best.ckpt` | Model with the lowest observed `val_loss` |
| `last.ckpt` | Most recent model at the end of training |
| `interrupted.ckpt` | Model saved when training is interrupted with `Ctrl+C` |

---

## Project structure

```text
.
├── gpt2/
│   ├── chat.py          # Chat formatting for training and inference
│   ├── checkpoint.py    # Checkpoint discovery
│   ├── config.py        # Model and training configuration
│   ├── generate.py      # Text generation
│   ├── manager.py       # MoE auxiliary-loss management
│   ├── model.py         # GPT-2, RoPE, SwiGLU, and MoE
│   └── train.py         # Data processing and training
├── frontend/
│   ├── app.py           # Streamlit application
│   ├── Dockerfile
│   └── run_app.sh       # UI startup script
├── docs/                # Local documentation
├── checkpoints/         # Trained models (ignored by Git)
├── logs/                # Training logs (ignored by Git)
├── Dockerfile
├── main.py              # Train, then generate text
├── open_docs.sh
├── pyproject.toml
└── run.sh
```

---

## Configuration

Common training settings can be changed from the command line:

```bash
uv run python -m gpt2.train --help
```

| Argument | Default | Description |
|---|---:|---|
| `--block-size` | `256` | Context length |
| `--n-layer` | `12` | Number of Transformer layers |
| `--n-head` | `12` | Number of attention heads |
| `--n-embd` | `768` | Embedding dimension |
| `--batch-size` | `8` | Batch size |
| `--max-steps` | `100` | Maximum training steps |
| `--eval-interval` | `100` | Validation interval |
| `--learning-rate` | `3e-4` | Learning rate |
| `--checkpoint-path` | `checkpoints` | Checkpoint output directory |
| `--no-amp` | Disabled | Disable mixed-precision training |

Advanced MoE settings are defined in `ModelConfig` inside `gpt2/config.py`.

| Setting | Default | Description |
|---|---:|---|
| `n_exp` | `8` | Number of experts; `1` uses a standard MLP |
| `top_k` | `2` | Experts selected for each token |
| `stride` | `2` | Interval between MoE layers |
| `train_capacity` | `1.25` | Expert capacity factor during training |
| `eval_capacity` | `2.0` | Expert capacity factor during evaluation |

---

## Local documentation

```bash
./open_docs.sh
```

Open [http://localhost:8080](http://localhost:8080) after the server starts. Press `Ctrl+C` in the terminal to stop it.

---

## Troubleshooting

### No checkpoint was found

List saved checkpoint files:

```bash
find checkpoints -maxdepth 2 -type f -name '*.ckpt' -print
```

If a file exists, pass its path explicitly:

```bash
uv run python -m gpt2.generate \
  --checkpoint-path checkpoints/version_0/best.ckpt
```

### Docker cannot access the GPU

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, verify the NVIDIA Driver and NVIDIA Container Toolkit installation.

### WandB is not enabled

Check the `.env` configuration:

```bash
grep -E '^(WANDB_ENTITY|WANDB_API_KEY)=' .env
```

Do not paste your API key into screenshots, logs, or GitHub issues.

### Git rejects a push as non-fast-forward

Rebase your local work onto the remote branch before pushing:

```bash
git fetch origin
git rebase origin/sen/moe
git push --set-upstream origin sen/moe
```

If conflicts occur, resolve them and continue:

```bash
git add -u
git rebase --continue
```

---

## News

| Date | Update |
|---|---|
| 2026-03-15 | Replaced positional encoding with Rotary Position Embedding (RoPE) |
| 2026-04-16 | Added web-based monitoring with Weights & Biases |
| 2026-07-11 | Integrated MoE, chat formatting, and robust checkpoint discovery |

---

<div align="center">

**Read the code. Run the experiment. Understand the model.**

</div>
