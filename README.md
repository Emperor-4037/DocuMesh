# AI Writing Assistant Platform

A production-ready, modular **microservices AI text-processing platform** with a RAG assistant — built with FastAPI, Celery, Qdrant, PostgreSQL, Redis, Prometheus, and Grafana.

---

## Architecture Overview

```
[Client]
   │
   ▼
[API Gateway :8000]  ← JWT auth, Trace ID, Prometheus metrics, routing
   │
   ├─► paraphrase-service   :8000
   ├─► grammar-service       :8000
   ├─► simplify-service      :8000
   ├─► tone-service          :8000
   ├─► summarize-service     :8000
   └─► rag-service           :8000
            │
            ├─► qdrant (vector DB)
            ├─► redis  (cache / Celery broker)
            ├─► postgres (metadata / job tracking)
            └─► rag-worker (Celery background ingestion)

[Prometheus :9090] ← scrapes /metrics from all services
[Grafana    :3000] ← dashboards over Prometheus
```

---

## Repository Structure

```
.
├── gateway/                        # API Gateway FastAPI app
│   ├── app/main.py                 # Routes, auth, metrics middleware
│   └── requirements.txt
├── services/
│   ├── paraphrase-service/app/
│   ├── grammar-service/app/
│   ├── simplify-service/app/
│   ├── tone-service/app/
│   ├── summarize-service/app/
│   └── rag-service/app/
│       ├── main.py                 # FastAPI app + upload endpoint
│       ├── pipeline.py             # Retrieve → Rerank → Generate
│       ├── ingestion.py            # Extract → Chunk → Embed → Upsert
│       └── worker.py               # Celery task definitions
├── shared/                         # Reusable Python modules
│   ├── auth.py                     # JWT verification
│   ├── config.py                   # Pydantic settings
│   ├── db.py                       # SQLAlchemy async engine
│   ├── logging.py                  # JSON structured logging
│   ├── metrics.py                  # Prometheus counters/histograms
│   ├── qdrant_client.py            # Qdrant async client
│   ├── redis_client.py             # Redis async client
│   ├── schemas.py                  # All Pydantic request/response models
│   └── tracing.py                  # OpenTelemetry OTLP exporter setup
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.gateway
│   │   └── Dockerfile.service      # Reusable for all NLP services
│   ├── k8s/
│   │   ├── gateway-deployment.yaml
│   │   ├── nlp-service-deployment.yaml
│   │   ├── rag-deployment.yaml
│   │   └── secrets.yaml
│   └── prometheus/prometheus.yml
├── tests/
│   └── test_gateway.py
├── .github/workflows/ci-cd.yml
└── docker-compose.yml
```

---

## Quick Start (Local with Docker Compose)

**Prerequisites:** Docker Desktop installed and running.

```bash
# 1. Clone the repo
git clone https://github.com/yourorg/ai-writing-assistant.git
cd ai-writing-assistant

# 2. (Optional) Copy and edit the env file
cp .env.example .env

# 3. Start everything
docker-compose up --build
```

| Service        | URL                        |
|----------------|----------------------------|
| API Gateway    | http://localhost:8000      |
| API Docs       | http://localhost:8000/docs |
| Prometheus     | http://localhost:9090      |
| Grafana        | http://localhost:3000      |
| Qdrant UI      | http://localhost:6333      |

---

## Authentication

All `/api/*` routes require a Bearer JWT token.

**Generate a test token:**

```python
import jwt
token = jwt.encode({"sub": "testuser"}, "super-secret-key-change-in-production", algorithm="HS256")
print(token)
```

Use it in requests:
```bash
curl -X POST http://localhost:8000/api/paraphrase \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "tone": "formal"}'
```

---

## API Endpoints

| Method | Path               | Description                         |
|--------|--------------------|-------------------------------------|
| GET    | /health            | Gateway health check                |
| POST   | /api/paraphrase    | Rewrite text in a given tone        |
| POST   | /api/grammar       | Grammar correction with diffs       |
| POST   | /api/simplify      | Simplify text to a reading level    |
| POST   | /api/tone          | Apply a target tone/style           |
| POST   | /api/summarize     | Summarize text                      |
| POST   | /api/rag/query     | Query the RAG assistant             |
| POST   | /api/rag/ingest    | Upload a document for RAG indexing  |
| GET    | /api/rag/ingest/status/{task_id} | Poll ingestion job status |

---

## Running Tests

```bash
pip install -r gateway/requirements.txt pytest httpx
PYTHONPATH=. pytest tests/ -v
```

---

## Kubernetes Deployment

```bash
# Apply secrets first (fill in real values!)
kubectl apply -f infra/k8s/secrets.yaml

# Deploy all services
kubectl apply -f infra/k8s/gateway-deployment.yaml
kubectl apply -f infra/k8s/nlp-service-deployment.yaml
kubectl apply -f infra/k8s/rag-deployment.yaml
```

---

## Next Steps / Roadmap

- [ ] Swap mock LLM responses with **OpenAI / Anthropic / Ollama** integration
- [ ] Add HuggingFace model pipelines for `paraphrase`, `grammar`, `summarize`
- [ ] Add PostgreSQL Alembic migrations for job tracking
- [ ] Add Grafana dashboards JSON for auto-provisioning
- [ ] Add HPA (Horizontal Pod Autoscaler) for inference services
- [ ] Add rate limiting middleware (e.g., slowapi)
- [ ] Add audit logging to PostgreSQL on every request
