# Book Recommendation System
![CI](https://github.com/simon-bouchard/book-recommendation-platform/actions/workflows/ci.yml/badge.svg)

A full-stack, production-grade book recommendation platform with personalized recommendations, semantic and full-text search, item similarity, and an AI-powered chatbot. Built to explore the end-to-end challenges of deploying ML systems: data cleaning, model training, serving infrastructure, observability, and automation.

---

## Architecture

The system is composed of several independent layers:

- **Frontend** — React + TypeScript SPA (search, recommendations, ratings, chatbot)
- **Backend** — FastAPI application handling auth, routing, caching, and business logic
- **Model servers** — 5 independent microservices, each owning a specific set of ML artifacts and endpoints
- **Support services** — MySQL (primary store), Redis (sessions, rate limiting, cache), Meilisearch (full-text search)
- **Enrichment pipeline** — Kafka + Spark were used for a one-time async enrichment job (LLM-driven metadata enrichment written back to MySQL); not running continuously in production
- **Observability** — Prometheus metrics, Grafana dashboards, Jaeger distributed tracing via OpenTelemetry

Each model server runs in its own Docker container with read-only artifact mounts. Hot-reload is implemented via a shared version pointer file: the training pipeline writes a new version, signals the servers, and each server reloads its artifacts with zero downtime.

---

## Model Servers

| Server | Port | Responsibilities |
|---|---|---|
| Embedder | 8001 | Attention-pooled subject embeddings (PyTorch) |
| Similarity | 8002 | Subject HNSW index, ALS HNSW index, hybrid similarity |
| ALS | 8003 | ALS user/item factors, warm-user detection |
| Metadata | 8004 | Book metadata lookup, Bayesian popularity scores |
| Semantic | 8005 | FAISS vector index for semantic search |

---

## Recommendation Engine

The system supports two user states and three explicit modes:

**Warm users** (have prior ratings)
- ALS (Alternating Least Squares) collaborative filtering retrieves candidate books based on behavioral patterns.

**Cold users** (no ratings)
- Attention-pooled subject embeddings compute similarity between the user's preferred subjects and all books.
- A Bayesian popularity prior blends embedding similarity with global popularity (adjustable weight).

**Item similarity** offers three strategies:
- *Behavioral (ALS)*: strong signal for same author/series, sparse for niche books.
- *Subject*: noisier but better at surfacing hidden gems and underrepresented titles.
- *Hybrid*: weighted combination of both, adjustable at query time.

In `auto` mode the system detects whether the user has ALS factors and routes accordingly, falling back to subject-based or popularity-based recommendations when needed.

---

## Inference Pipeline

### Recommendations

```
Request
  └─ Fetch user profile from DB (subjects, if mode=auto or subject)
       └─ Check warm status (ALS server, mode=auto only)
            └─ Model server call → ranked candidates
                 └─ asyncio.gather (parallel)
                      ├─ DB query: filter already-read books
                      └─ Metadata server: enrich candidates with title/author/year/cover
                           └─ Filter, slice to k, return
```

The final step runs the DB filter and metadata enrichment concurrently since both only require the candidate ID list. This reduces the combined cost from ~18ms sequential to ~max(8ms, 10ms).

### Similarity

```
Request → Similarity model server → Metadata server → Return
```

No DB round-trip — similarity is 2–3x faster than the recommendation path as a result.

---

## ML Training Pipeline

Training is automated and runs daily via a systemd timer. The pipeline:

1. **Run training scripts** — ALS factors, subject embeddings, similarity indices, metadata aggregates, Bayesian scores
2. **Quality gate** — evaluates the new artifacts against baseline thresholds; blocks promotion on regression
3. **Artifact promotion** — if the gate passes, the new version is written to a versioned directory and the active version pointer is updated
4. **Worker reload** — signals all 5 model servers to reload from the new pointer; each reloads independently with no downtime
5. **Notifications** — sends an email report on success or failure

Old artifact versions are retained for rollback and automatically retired after a configurable number of versions.

---

## Chatbot & Agents

The chatbot is built with LangGraph and routes requests across specialized agents:

- **Router** — classifies the intent of each message
- **Recommendation agent** — multi-stage pipeline: query understanding → candidate retrieval → ranking → response generation
- **Docs agent** — answers questions about the platform itself
- **Web agent** — handles general book/author questions via web search
- **Response agent** — handles messages that require no tool use (direct answers, greetings, clarifications)

Responses are streamed to the client via SSE. Conversation history is stored in Redis per session. Per-user rate limiting is also enforced via Redis, independently of history.

An offline evaluation framework runs scenarios against each agent using an LLM judge with pass/fail criteria. Evaluations are run manually due to API cost.

---

## Observability

**Metrics** (Prometheus + Grafana)
- Request counters and latency histograms per path (recommendations, similarity, search, chat)
- Exposed at `/metrics`, scraped by Prometheus, visualized in Grafana

**Distributed tracing** (OpenTelemetry → Jaeger)
- Auto-instrumented for FastAPI, httpx (model server calls), and SQLAlchemy
- Manual spans added at service and pipeline boundaries
- Health and metrics endpoints excluded from trace noise
- Trace ID propagated through all model server calls for end-to-end request visibility

---

## CI/CD

GitHub Actions runs on every push and pull request to `master`:

1. **Backend** — `ruff check` (lint), `ruff format --check`, `pytest tests/unit/`
2. **Frontend** — ESLint, TypeScript type check, Vite build
3. **Deploy** (master push only, after both pass) — SSH trigger runs `cd.sh` on the production server: `git pull` → `systemctl restart` → health check loop

---

## Research & Experiments

Several architectures were explored before settling on the current design:

- Residual MLPs over dot-product predictions
- Two-tower and three-tower architectures
- Clustering and regression methods on user embeddings
- Gated-fusion mechanisms
- Alternative attention pooling strategies (scalar, per-dimension, transformer-based self-attention)

These experiments informed the tradeoffs between accuracy, latency, and serving complexity. The final stack favors simple serving (dot products and matrix lookups at inference time) with complexity pushed to training.

---

## Data & Processing

The Book-Crossing dataset is noisy and incomplete — inconsistent ISBNs, duplicate editions, missing metadata, and no subject information. Significant preprocessing was required before the data was usable for modeling.

**1. ID Normalization & Book Merging**
ISBNs were normalized and mapped to Open Library work IDs. Different editions of the same book were merged under a single `work_id`, reducing duplication and ensuring consistent interaction counts. Each book is assigned a stable integer `item_idx` for modeling.

**2. Subject Enrichment & Reduction**
Subjects were pulled from Open Library metadata. Raw extraction yielded ~130,000 unique subject strings; after cleaning, deduplication, and frequency filtering, this was reduced to ~1,000 meaningful subjects. A subject vocabulary (`subject_idx → subject`) is maintained for indexing.

**3. User Data Cleaning**
Ages: extreme or implausible values removed or bucketed. Locations: parsed into country and normalized. Favorite subjects: top-k subjects derived from each user's rated books and stored separately for cold-start embeddings.

**4. Rating Data Cleaning**
Out-of-range ratings discarded. Duplicates dropped. Users and books with too few interactions filtered out to stabilize training.

**5. Subject & Metadata Normalization**
Subjects stored as indexed lists with padding/truncation to fixed length. Generic categories (e.g., "Fiction", "General") excluded from `main_subject`. Authors, years, and page counts cleaned into canonical forms.

**6. Aggregate Features**
Book-level and user-level aggregates (rating count, mean, standard deviation) precomputed during export to ensure consistency across training and inference.

Result: a normalized schema with clean IDs, consistent metadata, and a manageable subject vocabulary that feeds both collaborative and content-based models.

---

## Tech Stack

**ML / Data**
- Implicit (ALS collaborative filtering)
- PyTorch (attention-pooled subject embeddings)
- FAISS + HNSW (similarity indices)
- Sentence-Transformers (semantic embeddings)
- Pandas, NumPy, SciPy

**Backend**
- FastAPI, Uvicorn, Gunicorn
- SQLAlchemy + aiomysql (async MySQL)
- Redis (sessions, rate limiting, cache)
- Meilisearch (full-text search)
- Kafka + Spark (enrichment pipeline, one-time use)
- LangChain, LangGraph, OpenAI SDK (chatbot agents)

**Frontend**
- React 19, TypeScript, Tailwind CSS 4, Vite
- Radix UI (headless components)

**Observability**
- OpenTelemetry (tracing instrumentation)
- Jaeger (trace storage and UI)
- Prometheus (metrics)
- Grafana (dashboards)

**Infrastructure**
- Docker, Docker Compose
- Nginx (reverse proxy)
- Systemd (service management)
- GitHub Actions (CI/CD)

**Code Quality**
- Ruff (lint + format)
- Pyright (type checking)
- Pytest + pytest-asyncio

---

## Local Setup

The model servers and supporting services are orchestrated with Docker Compose.

```bash
git clone https://github.com/simon-bouchard/book-recommendation-platform
cd book-recommendation-platform

# Copy and fill in environment variables
cp deploy/deploy.env.example deploy/deploy.env

# Start model servers and support services
docker compose -f docker/compose/docker-compose.yml up -d

# Set up Python environment
conda env create -f environment.yml
conda activate bookrec-api

# Run the backend
uvicorn main:app --reload
```

The frontend is built separately:

```bash
cd frontend
npm ci
npm run build
```

Grafana is available at `/grafana`, Jaeger at port `16686`, Prometheus at port `9090`.
