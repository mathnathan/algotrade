FROM python:3.13

# Set up a non-root user for VS Code
RUN useradd --create-home --shell /bin/bash --uid 1000 vscode

# Create the HuggingFace cache directory with proper permissions
RUN mkdir -p /app/.cache/huggingface && \
chown -R vscode:vscode /app/.cache && \
chmod -R 755 /app/.cache

WORKDIR /workspace

# Install dependencies if you have a requirements.txt
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv

USER vscode

# Set HuggingFace environment variables as defaults
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface