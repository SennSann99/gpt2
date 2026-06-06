"""GPT-2 Streamlit チャットフロントエンド."""

import os
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（gpt2 パッケージのインポート解決用）
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402
import transformers

from gpt2.config import ModelConfig, TrainConfig  # noqa: E402
from gpt2.model import GPTLightning  # noqa: E402

transformers.utils.logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# モデルのロード（キャッシュ付き）
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[GPTLightning, torch.device]:
    """チェックポイントからモデルをロードし、キャッシュする."""
    ckpt_path_str = train_cfg.checkpoint_path
    if ckpt_path_str is None or Path(ckpt_path_str).is_dir():
        base_dir = Path(ckpt_path_str or "checkpoints")
        if base_dir.exists():
            versions = []
            for d in base_dir.iterdir():
                if d.is_dir() and d.name.startswith("version_"):
                    try:
                        versions.append(int(d.name[len("version_") :]))
                    except ValueError:
                        continue
            if versions:
                latest_version = max(versions)
                best_path = base_dir / f"version_{latest_version}" / "best.ckpt"
                if best_path.exists():
                    ckpt_path_str = str(best_path)
                else:
                    last_path = base_dir / f"version_{latest_version}" / "last.ckpt"
                    if last_path.exists():
                        ckpt_path_str = str(last_path)

    if ckpt_path_str is None or not Path(ckpt_path_str).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path_str}. "
            "checkpoints/version_N/best.ckpt が存在するか確認してください。"
        )

    ckpt_path = Path(ckpt_path_str)
    module = GPTLightning.load_from_checkpoint(
        str(ckpt_path),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.model.to(device)
    module.model.eval()
    return module, device


# ---------------------------------------------------------------------------
# テキスト生成
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_text(
    module: GPTLightning,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """プロンプトからテキストを生成する."""
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    out = module.model.generate(idx, max_new_tokens=max_new_tokens)
    # プロンプト部分を除外し、生成部分のみ返す
    generated_tokens = out[0][len(encoded) :].tolist()
    return tokenizer.decode(generated_tokens)


#---------------------------------------------------------------------------
# オープンソースモデルのロードとテキスト生成
#---------------------------------------------------------------------------
@st.cache_resource
def load_hf_model(model_name: str) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer, torch.device]:
    """Hugging Faceからモデルをロードし、キャッシュする."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device

@torch.no_grad()
def generate_text_hf(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """HFモデルを用いてテキストを生成する."""
    # Instructモデル用にチャットテンプレートを適用（可能であれば）
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = prompt

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # プロンプト部分を除外し、生成部分のみ返す
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


#---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="GPT-2 Chat", page_icon="🤖", layout="centered")
    st.title("GPT-2 Chat")

    # --- サイドバー ---
    with st.sidebar:
        st.header("Settings")
        model_type = st.radio("Model Source", ["Custom GPT-2 (Checkpoint)", "Open Source (Hugging Face)"])
        hf_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        if model_type == "Open Source (Hugging Face)":
            hf_model_name = st.text_input("Hugging Face Model ID", value="Qwen/Qwen2.5-0.5B-Instruct")

        max_new_tokens = st.slider(
            "Max New Tokens", min_value=16, max_value=512, value=128, step=16
        )


    # --- モデルのロード ---
    if model_type == "Custom GPT-2 (Checkpoint)":
        model_cfg = ModelConfig()
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", "checkpoints")
        train_cfg = TrainConfig(checkpoint_path=checkpoint_path)

        try:
            module, device = load_model(model_cfg, train_cfg)
        except (FileNotFoundError, OSError) as e:
            st.error(f"モデルのロードに失敗しました: {e}")
            st.stop()
    else:
        try:
            with st.spinner("Hugging Faceモデルをロード中..."):
                hf_model, hf_tokenizer, device = load_hf_model(hf_model_name)
        except Exception as e:
            st.error(f"HFモデルのロードに失敗しました: {e}")
            st.stop()

    # --- チャット履歴 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- ユーザー入力 ---
    if prompt := st.chat_input("メッセージを入力..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("生成中..."):
                if model_type == "Custom GPT-2 (Checkpoint)":
                    response = generate_text(module, device, prompt, max_new_tokens)
                else:
                    response = generate_text_hf(hf_model, hf_tokenizer, device, prompt, max_new_tokens)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
