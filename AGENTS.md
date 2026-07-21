# Kairos CDSS - AI Agent Guidelines & Project Context

> **Note for AI Coding Assistants & Agents:**  
> This file outlines the vision, architectural blueprint, domain rules, codebase topology, and coding standards for **Kairos**. Any AI agent working on this repository MUST read and adhere to these guidelines.

---

## 1. Project Mission & Overview

**Kairos** is an open-source, point-of-care **Clinical Decision Support System (CDSS)** engineered to interpret unstructured clinical queries and synthesize evidence-grounded, verifiable medical answers. 

It is designed as an open-source, self-hostable alternative to commercial clinical AI products (such as OpenEvidence), focusing on:
- **Absolute Evidence Grounding:** Minimizing and eliminating clinical hallucination by requiring strict citation provenance for all generated statements.
- **Auditable & Transparent Architecture:** Permissively licensed, open-weight models, and transparent data processing pipelines.
- **HIPAA-Compliant / Self-Hosted Privacy:** Zero data exfiltration to external proprietary black-box APIs when deployed locally.

---

## 2. Core System Architecture

The technical design of Kairos is documented in detail in [`Kairos CDSS Architecture Design.md`](file:///c:/DevDojo/kairos/Kairos%20CDSS%20Architecture%20Design.md). Key subsystems include:

### 2.1 Ingestion & Normalization Engine (PubMed / PMC)
- **Source Data:** PubMed Baseline & Daily Update XMLs, PMC Open Access JATS XML documents.
- **Parsing:** Custom `lxml`-based JATS XML parser resolving `<front>`, `<body>`, structural headings (IMRaD), and inline citation tags (`<xref ref-type="bibr">`).
- **Network Resilience:** Two-pass scraping harness resolving NCBI Proof-of-Work (`cloudpmc-viewer-pow` cookie challenge) using headless browser execution before delegating to `curl` or SFTP.
- **Chunking Strategy:** Paragraph-level semantic sliding-window chunking (256-word target with 50-word overlap at sentence boundaries), preserving parent headers, PMID, PMCID, and provenance metadata.

### 2.2 Dual-Retrieval & Reranking Core
- **Dense Vector Search:** MedCPT Query Encoder (`ncbi/MedCPT-Query-Encoder`) generating 768-dimensional dense vectors against a Qdrant vector database (`pubmed_articles` collection) indexed with HNSW graph traversal.
- **Sparse Lexical Search:** BM25 sparse index ($k_1=1.5, b=0.8$) to ensure exact matches for drug names, ICD-10/CPT codes, and trial identifiers.
- **Reciprocal Rank Fusion (RRF):** Merging dense and sparse candidate lists using rank position scoring:
  $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k=60)$$
- **Late-Interaction Reranking:** MedCPT Cross-Encoder (`ncbi/MedCPT-Cross-Encoder`) re-scoring top $N$ fused candidates for high-precision context injection.

### 2.3 Clinical LLM & Reasoning Layer
- **Supported Models:** MedGemma 1.5 (4B/27B), Apertus-70B-MeditronFO, or BioMistral 7B hosted via `vLLM` (PagedAttention) or cloud API providers (e.g., Groq using `openai/gpt-oss-20b` or Llama/Mixtral variants for testing).
- **Zero-Hallucination Guardrails:** System prompts instructing explicit refusal if retrieved context is insufficient; post-generation factual consistency checks.

### 2.4 Kairos Evidence Grade Engine
- Automated real-time evidence appraisal based on the Cochrane-validated **GRADE framework**:
  - Classifies study designs (Systematic Review, RCT, Cohort, Case Series).
  - Evaluates Risk of Bias, Directness, Consistency, and Precision.
  - Outputs evidence grade ratings (A, B, C, D, or U for Unable to Grade).

### 2.5 Interoperability Layer
- SMART on FHIR integration capabilities.
- Model Context Protocol (MCP) server & Agent-to-Agent (A2A) interfaces for EHR/EHR-adjacent tool interoperability.

---

## 3. Codebase Topology & File Map

```
kairos/
├── .agents/                 # AG Kit agents, skills, and workflows
├── data/                    # Interactive notebooks and dataset scratchpads
│   └── pubmed_qdrant_etl.ipynb
├── docker/                  # Docker initialization & configuration scripts
├── etl/                     # Data Extraction, Transformation & Loading
│   └── run_etl.py           # Main PubMed XML parsing, embedding & Qdrant upsert pipeline
├── src/                     # Core FastAPI Application Source
│   └── app/
│       ├── api/             # API routes & endpoints
│       ├── core/            # Config, DB clients, logging, settings
│       └── main.py          # FastAPI application entrypoint
├── ui/                      # Frontend interfaces
│   ├── app.py               # Streamlit point-of-care UI test harness
│   └── rag.py               # RAG chain integration (Groq API, Qdrant vector search)
├── Dockerfile.dev           # Development container definition
├── Dockerfile.prod          # Production container definition
├── docker-compose.yml       # Local development stack (Qdrant + MinIO)
├── docker-compose.prod.yml  # Production deployment stack (Qdrant init container sync)
├── pyproject.toml           # Project dependencies & tool configurations (uv)
└── README.md                # General developer guide & setup documentation
```

---

## 4. Environment & Package Management

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic Python dependency management.

- **Sync all dependencies (ETL + UI):**
  ```bash
  uv sync --group etl --group ui
  ```
- **Environment variables (`.env`):**
  Ensure `.env` contains:
  - `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_API_KEY`
  - `MINIO_ENDPOINT` / `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`
  - `NCBI_API_KEY` (for NCBI e-utilities rate limit expansion)
  - `GROQ_API_KEY` (for Streamlit UI testing harness)

---

## 5. Developer Workflows & Execution Commands

### 5.1 Local ETL & Vector Database Ingestion
Running the heavy ingestion pipeline is **restricted to Dev/Staging environments**:
```bash
# Start vector store and object storage
docker compose up qdrant minio -d

# Run local PubMed ETL ingestion pipeline
uv run python etl/run_etl.py
```

### 5.2 Streamlit Point-of-Care Testing UI
```bash
uv run streamlit run ui/app.py --server.port 8501 --server.headless true
```

### 5.3 FastAPI Service
```bash
uv run uvicorn src.app.main:app --reload --port 8000
```

---

## 6. Mandatory Rules for AI Agents

When authoring, refactoring, or reviewing code in this codebase:

1. **Clinical Accuracy & Citation Provenance:** Never strip citation metadata (`pmid`, `pmcid`, `doi`, `journal`, `authors`, `year`) from text chunk data structures or RAG response outputs.
2. **Deterministic Fallbacks:** Ensure fallback handling when medical queries cannot be grounded in retrieved context (return clear refusal or low-confidence warning rather than hallucinating).
3. **No ETL in Production Containers:** Never trigger raw dataset scraping or batch vector embeddings inside production runtime API containers. Production uses pre-built Qdrant snapshots via `docker-compose.prod.yml` and the `qdrant-init` container strategy.
4. **Clean Code & Type Hints:** Write concise Python code with strict type annotations (`pydantic` models for schemas, `typing` primitives).
5. **Respect `uv` Package Isolation:** Add dependencies using `pyproject.toml` dependency groups (`etl`, `ui`, `dev`) via `uv add --group <group>`.
