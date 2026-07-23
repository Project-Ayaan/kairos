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

**Required environment variables:**
- `QDRANT_URL`: Qdrant vector database endpoint (default: `http://localhost:6333`)
- `COLLECTION_NAME`: Shared knowledge collection for PubMed + policy documents (default: `kairos_knowledge`)
- `GROQ_API_KEY`: Groq API key for LLM synthesis
- `A2A_API_KEY`: Optional Bearer token for A2A endpoint (open by default if unset)

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

2. **Ingest PubMed Literature:**
   ```bash
   uv run python etl/run_etl.py
   ```
   Fetches PubMed articles via NCBI E-utilities API, applies sentence-boundary sliding-window chunking (256 words, 50-word overlap), generates MedCPT embeddings, and upserts to Qdrant (`kairos_knowledge` collection with `source_type="pubmed"`).

3. **Ingest Local Policy Documents (Optional):**
   ```bash
   uv run python etl/ingest_documents.py /path/to/pdf_or_docx_files
   ```
   Accepts local PDF/DOCX files, extracts text by page/heading, chunks via same sentence-boundary logic, embeds with MedCPT, and upserts to shared `kairos_knowledge` collection with `source_type="policy_document"`.

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

#### C. A2A Agent Service
Expose Kairos as an A2A-compatible agent for orchestrators:
```bash
uv run uvicorn src.app.main:app --reload --port 8689
```
Agent discovery available at `http://localhost:8689/.well-known/agent.json`. Supports both PubMed and policy-document queries via single `/a2a` endpoint. Optional Bearer token authentication via `A2A_API_KEY` environment variable.

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

**Migration Note:** If migrating from earlier deployments with `pubmed_articles` collection, manually reindex or create an alias to `kairos_knowledge` before deploying. The system now defaults to `kairos_knowledge` for unified PubMed + policy-document storage.
