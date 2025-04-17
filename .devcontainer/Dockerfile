FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /workspaces

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Install CPU-only PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 