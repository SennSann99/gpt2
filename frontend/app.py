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

from gpt2.chat import format_inference_prompt, strip_generated_response  # noqa: E402
from gpt2.checkpoint import resolve_checkpoint_path  # noqa: E402
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
    ckpt_path = resolve_checkpoint_path(train_cfg.checkpoint_path)
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
    formatted_prompt = format_inference_prompt(prompt)
    encoded = tokenizer.encode(formatted_prompt)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    out = module.model.generate(idx, max_new_tokens=max_new_tokens)
    generated_tokens = out[0][len(encoded) :].tolist()
    return strip_generated_response(tokenizer.decode(generated_tokens))

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="GPT-2 Chat", page_icon="🤖", layout="centered")
    st.title("GPT-2 Chat")

    # --- サイドバー ---
    with st.sidebar:
        st.header("Settings")
        max_new_tokens = st.slider(
            "Max New Tokens", min_value=16, max_value=512, value=128, step=16
        )

    # --- モデルのロード ---
    model_cfg = ModelConfig()
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "checkpoints")
    train_cfg = TrainConfig(checkpoint_path=checkpoint_path)

    try:
        module, device = load_model(model_cfg, train_cfg)
    except (FileNotFoundError, OSError) as e:
        st.error(f"モデルのロードに失敗しました: {e}")
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
                response = generate_text(module, device, prompt, max_new_tokens)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
