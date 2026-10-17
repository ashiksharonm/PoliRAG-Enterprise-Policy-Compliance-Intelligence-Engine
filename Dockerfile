# syntax=docker/dockerfile:1
# Policy change impact service. CPU only. The legacy pipeline image lives in docker/Dockerfile.
FROM python:3.14-slim AS build
WORKDIR /w
COPY requirements-impact.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements-impact.txt
ARG PRELOAD_MODELS=1
ENV HF_HOME=/opt/hf
RUN if [ "$PRELOAD_MODELS" = "1" ]; then /opt/venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; else mkdir -p /opt/hf; fi

FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/opt/hf PATH=/opt/venv/bin:$PATH
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/hf /opt/hf
COPY src/__init__.py src/__init__.py
COPY src/impact src/impact
COPY src/eval/__init__.py src/eval/__init__.py
COPY src/eval/retrieval_metrics.py src/eval/retrieval_metrics.py
COPY data/raw data/raw
COPY data/versions data/versions
COPY data/eval/retrieval_gold.json data/eval/retrieval_gold.json
RUN chown -R app:app /opt/hf
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"
CMD ["uvicorn", "src.impact.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
