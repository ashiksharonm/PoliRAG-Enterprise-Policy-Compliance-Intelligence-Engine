# PoliRAG - Enterprise Policy & Compliance Intelligence Engine

> **A production-grade RAG file system for enterprise compliance, policy reasoning, and audit-safe AI.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Overview

PoliRAG is an **enterprise-grade Retrieval-Augmented Generation (RAG) system** designed for compliance, legal, and audit teams. It treats documents as first-class data assets with versioning, access control, evaluation, and full observability.

### Key Features

- ✅ **No Hallucinated Answers** - Citation-required responses with confidence thresholds
- ✅ **Full Traceability** - Every answer linked to source documents
- ✅ **Strict Access Control** - Role-Based Access Control (RBAC) with multi-tenancy
- ✅ **Evaluation-Driven** - Automated Recall@K, MRR, and hallucination rate tracking
- ✅ **Production-Ready** - Comprehensive logging, metrics, and observability
- ✅ **Document Versioning** - Content-hash based deduplication and version tracking
- ✅ **Hybrid Search** - BM25 + Semantic search with cross-encoder reranking
- ✅ **PII Protection** - Automatic PII detection and redaction

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                    │
│  ├── /ingest   ├── /query   ├── /health   ├── /metrics │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│  Guardrails Layer                                       │
│  ├── RBAC   ├── PII Redaction   ├── Rate Limiting      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│  RAG Pipeline                                           │
│  ├── Ingestion → Chunking → Embedding → Vector Store   │
│  └── Retrieval (Hybrid) → Rerank → Generation          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│  Observability & Evaluation                             │
│  ├── Recall@K   ├── MRR   ├── Hallucination Rate       │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
rag-compliance-system/
├── README.md                   # This file
├── pyproject.toml             # Poetry dependencies
├── requirements.txt           # Pip dependencies
├── .env.example              # Environment variables template
├── docker/                   # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/                     # Data storage
│   ├── raw/                 # Original uploaded documents
│   ├── staged/              # Normalized + chunked text
│   └── manifests/           # Ingestion logs and hashes
├── indexes/                 # Vector indexes
│   ├── faiss/              # FAISS vector indexes
│   └── metadata/           # Document + chunk metadata
├── src/                    # Source code
│   ├── config.py          # Configuration management
│   ├── models.py          # Shared data models
│   ├── app/               # FastAPI routes
│   ├── ingestion/         # Document loaders & versioning
│   ├── chunking/          # Chunking strategies
│   ├── embeddings/        # Embedding generation & caching
│   ├── vectorstore/       # FAISS adapter
│   ├── retrieval/         # Hybrid search + reranking
│   ├── generation/        # LLM prompt templates
│   ├── guardrails/        # RBAC, PII, safety checks
│   ├── eval/              # Evaluation framework
│   └── observability/     # Logging & metrics
├── tests/                 # Test suite
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_eval.py
└── scripts/               # CLI tools
    ├── ingest.py         # Document ingestion CLI
    ├── build_index.py    # Index building CLI
    └── serve.py          # API server CLI
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API Key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-compliance-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and configuration
```

4. **Download spaCy model for PII detection**
```bash
python -m spacy download en_core_web_sm
```

### Usage

#### 1. Ingest Documents

```bash
python scripts/ingest.py --path ./documents --tenant-id company-a --role legal
```

#### 2. Build Vector Index

```bash
python scripts/build_index.py
```

#### 3. Start API Server

```bash
python scripts/serve.py
```

#### 4. Query the System

```bash
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What are the data retention policies?",
    "tenant_id": "company-a",
    "user_role": "legal"
  }'
```

## 📚 Supported Document Formats

- **PDF** - Including tables and structured content
- **DOCX** - Microsoft Word documents
- **Markdown** - .md files
- **JSON** - Structured data
- **Email** - .eml files
- **Text** - Plain text files

## 🔒 Security & Guardrails

### Role-Based Access Control (RBAC)

Supported roles:
- `admin` - Full access
- `legal` - Legal documents
- `audit` - Audit reports
- `compliance` - Compliance policies
- `read_only` - Read-only access

### PII Redaction

Automatically detects and redacts:
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- Personal names (via NER)

### Multi-Tenancy

Complete tenant isolation at the metadata level ensures data separation.

## 📊 Evaluation Metrics

### Recall@K
Measures retrieval quality - % of relevant documents in top K results.

### Mean Reciprocal Rank (MRR)
Measures ranking quality - average of reciprocal ranks of first relevant result.

### Hallucination Rate
% of answers not supported by retrieved context.

### Confidence Score
LLM-generated confidence in the answer based on context quality.

## 🔧 Configuration

All configuration is managed via environment variables. See `.env.example` for full list.

Key configurations:

```bash
# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=77

# Retrieval
RETRIEVAL_TOP_K=20
RETRIEVAL_RERANK_TOP_K=5
RETRIEVAL_BM25_WEIGHT=0.3
RETRIEVAL_SEMANTIC_WEIGHT=0.7

# Generation
GENERATION_CONFIDENCE_THRESHOLD=0.65
GENERATION_REQUIRE_CITATION=true
```

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v --cov=src
```

Run specific test modules:

```bash
pytest tests/test_ingestion.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_eval.py -v
```

## 📈 Monitoring

### Metrics Endpoint

Prometheus metrics available at:
```
http://localhost:9090/metrics
```

### Key Metrics

- `polirag_requests_total` - Total API requests
- `polirag_retrieval_duration_seconds` - Retrieval latency
- `polirag_generation_confidence` - Answer confidence distribution
- `polirag_eval_recall_at_k` - Current Recall@K score
- `polirag_eval_hallucination_rate` - Current hallucination rate
- `polirag_pii_detections_total` - PII detection count

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t polirag:latest -f docker/Dockerfile .
```

### Run with Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up
```

## 🛠️ Development

### Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Run checks:

```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## 📖 API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 🎯 Production Deployment

### Checklist

- [ ] Set `ENVIRONMENT=production` in .env
- [ ] Configure proper log aggregation
- [ ] Set up Prometheus metrics scraping
- [ ] Enable rate limiting
- [ ] Configure RBAC policies
- [ ] Set up backup for indexes
- [ ] Configure SSL/TLS
- [ ] Set proper resource limits

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a production system template. Contributions should focus on:
- Bug fixes
- Performance improvements
- Additional document format support
- Enhanced evaluation metrics

## 📧 Support

For issues and questions, please open an issue on the repository.

---

**Built for production. Designed for compliance. Tested for reliability.**
