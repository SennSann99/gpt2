# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3 \
    python3-pip \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Copy dependency metadata first (better Docker caching)
COPY pyproject.toml uv.lock README.md .python-version ./

# Install Python dependencies
RUN uv sync --frozen --no-install-project

# Now copy the rest of the project
COPY . .

# Install the local project after source files are present
RUN uv sync --frozen

# Default command
CMD ["uv", "run", "main.py"]
