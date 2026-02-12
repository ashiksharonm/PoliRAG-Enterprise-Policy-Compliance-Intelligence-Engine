# PoliRAG — Production Dockerfile
# Multi-stage build: slim base keeps the final image small (~1.2 GB with pytorch).
# We install system deps first (rarely changes), then pip deps (changes occasionally),
# then copy source (changes often). This maximises layer cache reuse.

# ---------------------------------------------------------------------------
# Stage 1 — Builder (install Python deps into a venv we can copy later)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed at build time only
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install spacy && \
    python -m spacy download en_core_web_sm 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="PoliRAG Team"
LABEL description="PoliRAG — Enterprise Policy & Compliance Intelligence Engine"

# libmagic is needed at runtime by python-magic
RUN apt-get update && \
    apt-get install -y --no-install-recommends libmagic1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local
# Copy spacy model data (installed globally during build)
COPY --from=builder /usr/local/lib/python3.11/site-packages/en_core_web_sm /usr/local/lib/python3.11/site-packages/en_core_web_sm

WORKDIR /app

# Copy application source
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY indexes/ indexes/
COPY pyproject.toml .
COPY requirements.txt .
COPY README.md .

# Create required directories (will be overridden by volume mounts in compose)
RUN mkdir -p data/raw data/staged data/manifests data/eval \
             indexes/faiss indexes/metadata \
             logs

# Non-root user for security
RUN groupadd -r polirag && useradd -r -g polirag polirag && \
    chown -R polirag:polirag /app
USER polirag

# Healthcheck — matches the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

# Default entrypoint: start the API server
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
