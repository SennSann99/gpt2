import argparse

import tiktoken
import torch

from gpt2.chat import format_inference_prompt, strip_generated_response
from gpt2.checkpoint import resolve_checkpoint_path
from gpt2.config import ModelConfig, TrainConfig
from gpt2.model import GPTLightning


def parse_args() -> tuple[ModelConfig, TrainConfig, str, int]:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained GPT-2 model"
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to checkpoint (e.g. checkpoints/version_0/best.ckpt). If None, tries to find the latest version's best.ckpt",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--prompt", default="One day, ")
    parser.add_argument("--max-new-tokens", type=int, default=128)

    args, _ = parser.parse_known_args()

    model_cfg = ModelConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    train_cfg = TrainConfig(
        limit_rows=0,
        val_rows=0,
        batch_size=1,
        max_steps=0,
        eval_interval=0,
        eval_batches=0,
        learning_rate=0.0,
        weight_decay=0.0,
        beta1=0.0,
        beta2=0.0,
        grad_clip=0.0,
        warmup_steps=0,
        seed=1337,
        num_workers=0,
        amp=False,
        checkpoint_path=args.checkpoint_path,
    )
    return model_cfg, train_cfg, args.prompt, args.max_new_tokens


@torch.no_grad()
def generate(
    model_cfg: ModelConfig, train_cfg: TrainConfig, prompt: str, max_new_tokens: int
) -> str:
    ckpt_path = resolve_checkpoint_path(train_cfg.checkpoint_path)
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    module = GPTLightning.load_from_checkpoint(
        str(ckpt_path),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.model.to(device)
    module.model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = format_inference_prompt(messages)
    encoded = tokenizer.encode(formatted_prompt)[-model_cfg.block_size :]
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    out = module.model.generate(idx, max_new_tokens=max_new_tokens)
    generated_tokens = out[0][len(encoded) :].tolist()
    return strip_generated_response(tokenizer.decode(generated_tokens))


def main() -> None:
    model_cfg, train_cfg, prompt, max_new_tokens = parse_args()
    text = generate(model_cfg, train_cfg, prompt, max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
