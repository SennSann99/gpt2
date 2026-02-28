# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
# "Do not ask me any questions; just use the default settings for everything."
ENV DEBIAN_FRONTEND=noninteractive
# Sets the "Home Base" inside the container.
WORKDIR /workspace
# After this line: 
# 1. Docker creates a folder called /workspace if it doesn't already exist.
# 2. Every following command (like COPY . . or RUN uv sync) happens inside that folder.
# 3. When you eventually docker run the container and jump inside, you will automatically start in /workspace.

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
# curl -Ls: This tells the container to reach out to the internet (curl) and download the installation script. 
# The -L follows redirects if the URL moves, and -s (silent) keeps the logs from being cluttered with progress bars.
# | sh: The "pipe" (|) takes the script that was just downloaded and immediately hands it to the shell (sh) to execute it.
RUN curl -Ls https://astral.sh/uv/install.sh | sh
# When you type uv in your terminal, the computer doesn't instinctively know where that program lives. 
# It has a list of "folders to check"—this list is the PATH variable. It checks the first folder, then the second, and so on.
# If it's not in any of them, you get the dreaded command not found error.
# /root/.local/bin: This is the modern default location where many Python tools (like pipx or uv) install themselves when run as the root user.
# /root/.cargo/bin: This is the traditional location for tools written in Rust (which uv is).
#                   Since uv is built with Rust, it often places its executable here.
# :${PATH}: This is the most important part. It says: "Take the new folders I just listed, and then attach the old, original list of folders to the end."
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Copy dependency metadata first (better Docker caching)
# In your GPT-2 project, you have two types of files:
# The Heavy Stuff (Dependencies): Your uv.lock and pyproject.toml. These rarely change. 
#     You only touch them when you need a new library (like torch or numpy).
# The Frequent Stuff (Your Code): 
#     Your main.py or model.py. You change these every few minutes as you fix bugs or tune your GPT-2 parameters.

# What happens: You change that same print statement in main.py.

# Docker's reaction: 
# 1.  "Did uv.lock change? No. I'll use the cached Step 1."
# 2.  "Since Step 1 is the same, I can skip Step 2 entirely! I'll use the 5GB of libraries I already downloaded yesterday."
# 3.  "Now, Step 3 (COPY . .)—yes, main.py changed here. I'll re-run this tiny copy."
#Result: Your build finishes in 2 seconds instead of 10 minutes.
COPY pyproject.toml uv.lock .python-version ./

# Install Python dependencies
# Without --frozen: 
# uv will look at your pyproject.toml, see if there are newer versions available on the internet, 
# and potentially update your lock file during the build. This is slow and dangerous.
# With --frozen: 
# It tells uv: "If the pyproject.toml and uv.lock don't match perfectly, fail the build. Do not try to fix them. Use exactly what is in the lock file."
# --no-install-project: 
# This is the clever part. It installs all the heavy libraries (like torch, transformers, etc.) 
# but doesn't try to install your actual GPT-2 code yet (because you haven't copied it in!).
RUN uv sync --frozen --no-install-project

# Now copy the rest of the project
COPY . .

# Install the local project after source files are present
# This second run "links" your actual project code into the environment. Since the libraries are already there, this step takes less than a second.
RUN uv sync --frozen

# Default command
CMD ["uv", "run", "main.py"]
