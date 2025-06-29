FROM python:3.13

# Set up a non-root user for VS Code
RUN useradd --create-home --shell /bin/bash --uid 1000 vscode

WORKDIR /workspace

# Install dependencies if you have a requirements.txt
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv

USER vscode
