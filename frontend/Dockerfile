# Streamlit フロントエンド用 Dockerfile (CPU)
#FROM python:3.12-slim
# Streamlit フロントエンド用 Dockerfile (GPU)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# システム依存パッケージ
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv インストール
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# 依存関係メタデータをコピー（Docker キャッシュ活用）
COPY pyproject.toml uv.lock .python-version ./

# 依存関係インストール
RUN uv sync --frozen --no-install-project

# ソースコードをコピー
COPY . .

# プロジェクトをインストール
RUN uv sync --frozen

# Streamlit のデフォルトポート
EXPOSE 8501

# Streamlit 起動
CMD ["uv", "run", "streamlit", "run", "frontend/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--browser.gatherUsageStats", "false"]
