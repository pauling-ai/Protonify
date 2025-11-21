# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System dependencies for RDKit and scientific stack
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install "rdkit>=2022.3.5"

WORKDIR /app

COPY pyproject.toml ./
COPY protonify/ protonify/

RUN pip install --no-cache-dir .[dev]

# Install pip build and build the package
RUN pip install build && python -m build

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

HEALTHCHECK --interval=1m --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "from rdkit import Chem; print(Chem.MolToSmiles(Chem.MolFromSmiles('CCO')))" || exit 1

ENTRYPOINT ["python", "-m", "protonify"]

# Documentation
# Build:   docker build -t protonify .
# Run:     docker run --rm -it protonify --template smart --ph 7.4 --smiles "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C" --model-path models/t_dwar_v_novartis_a_b.pt
# You can override the entrypoint or pass any CLI args as needed.
