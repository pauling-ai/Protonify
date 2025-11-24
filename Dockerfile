# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System dependencies for RDKit, matplotlib, Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libexpat1 \
        libxcb1 \
        libfontconfig1 \
        libfreetype6 \
        libjpeg62-turbo \
        libpng16-16 \
        zlib1g \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY protonify/ protonify/

# Install from pyproject.toml using uv
RUN uv pip install --system .

# Set model path
ENV PROTONIFY_MODEL_PATH=/app/protonify/models/t_dwar_v_novartis_a_b.pt
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "protonify"]
CMD ["--help"]

# Build:  docker build -t protonify .
# Run:    docker run --rm protonify --smiles "CCO" --ph 7.4 --template smart
