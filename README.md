# Kairos CDSS

Kairos is an open-source Clinical Decision Support System (CDSS) built to interpret unstructured clinical queries and synthesize evidence-grounded answers. It leverages a Retrieval-Augmented Generation (RAG) architecture powered by Qdrant, MinIO, MedCPT embeddings, and domain-tuned medical language models.

For detailed architecture blueprints and AI coding guidelines, see:
- [Kairos CDSS Architecture Design.md](file:///c:/DevDojo/kairos/Kairos%20CDSS%20Architecture%20Design.md) - Deep-dive technical blueprint.
- [AGENTS.md](file:///c:/DevDojo/kairos/AGENTS.md) - AI agent context, guidelines, and repository conventions.

---

## Technical Architecture Overview

Kairos consists of five primary subsystems:
1. **Ingestion & Normalization Engine (`etl/`):** Fetches PubMed abstracts and PMC JATS XML full-text articles, parses XML structure with `lxml`, applies paragraph-level sliding window chunking, and computes MedCPT embeddings.
2. **Dual-Retrieval Core:** Combines MedCPT dense vector search (Qdrant) and BM25 sparse keyword search via Reciprocal Rank Fusion ($k=60$) and MedCPT Cross-Encoder reranking.
3. **Evidence Appraisal Engine (GRADE):** Evaluates study designs, risk of bias, and evidence certainty to assign EvidenceGrade ratings (A through D, or U).
4. **FastAPI Backend (`src/app/`):** High-performance async API service exposing CDSS search, evidence generation, and interoperability endpoints.
5. **Streamlit Point-of-Care Harness (`ui/`):** Interactive UI for querying literature, testing RAG response generation, and inspecting retrieved source provenance.

---

## Developer Setup & Usage Guide

### Prerequisites
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (Python package installer and runner)
- Docker & Docker Compose

### 1. Environment Setup

Copy `.env.example` to `.env` and fill in required variables:
```bash
cp .env.example .env
```

Sync all project dependencies (including ETL and UI groups):
```bash
uv sync --group etl --group ui
```

---

### 2. Data ETL & Local Vector Ingestion (Dev Environment)

To ensure high performance and zero downtime in production, the heavy Extract, Transform, Load (ETL) pipeline runs *only* in developer or staging environments.

1. **Start Local Infrastructure (Qdrant + MinIO):**
   ```bash
   docker compose up qdrant minio -d
   ```

2. **Run Automated Ingestion Pipeline:**
   ```bash
   uv run python etl/run_etl.py
   ```
   This script parses XML articles, implements sentence-boundary sliding-window chunking, generates MedCPT embeddings, and upserts them to Qdrant (`pubmed_articles` collection).

3. **Interactive Notebook (Optional):**
   ```bash
   uv run --group etl jupyter notebook data/pubmed_qdrant_etl.ipynb
   ```

---

### 3. Running the Services

#### A. Streamlit CDSS UI Test Harness
Test RAG generation, vector similarity search, and source provenance interactively:
```bash
uv run streamlit run ui/app.py --server.port 8501 --server.headless true
```
Open `http://localhost:8501` in your browser. Ensure `GROQ_API_KEY` is configured in `.env`.

#### B. FastAPI Core Backend Service
Start the core REST API server:
```bash
uv run uvicorn src.app.main:app --reload --port 8000
```
API docs will be accessible at `http://localhost:8000/docs`.

---

### 4. Production Migration (Init Container Strategy)

To avoid running ETL scripts on production servers, Kairos uses an **Init Container (Raw Storage Volume Sync)** approach.

1. **Create & Export Qdrant Snapshot (Staging/Dev):**
   ```bash
   docker compose stop qdrant
   tar -czf qdrant_snapshot.tar.gz -C <path_to_qdrant_storage_volume> .
   ```
   Upload `qdrant_snapshot.tar.gz` to the `qdrant-snapshots` bucket in MinIO and record its SHA256 checksum.

2. **Deploy via Production Docker Compose:**
   Set `QDRANT_SNAPSHOT_SHA256` in your production environment and launch:
   ```bash
   docker compose -f docker-compose.prod.yml up -d
   ```

**How it works:**
- The `qdrant-init` container runs before the main `qdrant` service.
- If `/qdrant/storage` is empty, it downloads `qdrant_snapshot.tar.gz` from MinIO, verifies the SHA256 hash, and extracts the volume contents.
- The main `qdrant` container starts with all vector index data pre-populated on disk.
