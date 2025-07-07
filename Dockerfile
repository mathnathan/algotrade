FROM python:3.13

# Set up a non-root user for VS Code
RUN useradd --create-home --shell /bin/bash --uid 1000 vscode

# Create the HuggingFace cache directory with proper permissions
RUN mkdir -p /workspace/.cache/huggingface && \
chown -R vscode:vscode /workspace/.cache && \
chmod -R 755 /workspace/.cache

WORKDIR /workspace

# Install dependencies if you have a requirements.txt
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv

USER vscode

# Set HuggingFace environment variables as defaults
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface