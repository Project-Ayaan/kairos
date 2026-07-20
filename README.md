# Kairos CDSS

Kairos is an open-source Clinical Decision Support System (CDSS) built to interpret unstructured clinical queries and synthesize evidence-grounded answers. It leverages a Retrieval-Augmented Generation (RAG) architecture powered by Qdrant, MinIO, and large language models tailored for the medical domain.

---

## Architecture Overview

The system operates across several core components:
- **FastAPI Backend:** Provides the API for processing clinical queries.
- **Qdrant:** High-performance vector database storing dense embeddings (MedCPT) of medical literature.
- **MinIO:** S3-compatible object storage used primarily for snapshot management and data migration.

---

## Developer Guide: Data ETL & Production Migration

To ensure high performance and zero downtime in production, the heavy Extract, Transform, Load (ETL) pipeline runs *only* in the developer or staging environments. The resulting data is then migrated to production using Docker init containers.

### 1. Running the ETL Pipeline (Dev Environment)

The ETL pipeline pulls abstracts and full-text XML from PubMed/PMC, parses the JATS structure, creates MedCPT embeddings, and upserts them into a local Qdrant instance.

**Step-by-Step Setup:**

1. **Start the Infrastructure:**
   Spin up the required services (Qdrant and MinIO) using Docker Compose:
   ```bash
   docker compose up qdrant minio -d
   ```

2. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in the necessary keys (e.g., `NCBI_API_KEY`, `MINIO_ROOT_USER`).

3. **Install Dependencies:**
   The ETL pipeline and UI require specific dependencies. Sync both package groups using `uv`:
   ```bash
   uv sync --group etl --group ui
   ```

4. **Run the Automated ETL Pipeline:**
   Ingest 1,000 PubMed articles, split them into chunks, generate MedCPT embeddings, and load them to Qdrant:
   ```bash
   uv run python etl/run_etl.py
   ```
   This script parses XML articles, implements sentence-boundary sliding-window chunking, generates embeddings in batches, and upserts them to the `pubmed_articles` collection.

5. **Interactive Notebook (Optional):**
   Alternatively, you can run the pipeline interactively:
   ```bash
   uv run --group etl jupyter notebook data/pubmed_qdrant_etl.ipynb
   ```

### 2. Testing with Streamlit CDSS UI

After populating Qdrant, you can start the Streamlit testing interface to query the database and test the RAG response pipeline using Groq.

1. **Verify environment keys:**
   Ensure `GROQ_API_KEY` is present in your `.env` file. The RAG pipeline defaults to using the `openai/gpt-oss-20b` model.

2. **Launch Streamlit server:**
   Start the Streamlit application in headless mode:
   ```bash
   uv run streamlit run ui/app.py --server.port 8501 --server.headless true
   ```

3. **Interact and test:**
   Open `http://localhost:8501` in your browser. Use the provided quick test query buttons or input your own clinical query. You will see:
   - The synthesized grounded answer with inline citations.
   - The retrieved evidence sources from Qdrant with authors, journal, year, PMID, and cosine similarity score.
   - Configurable parameters in the sidebar (Top-K chunks, Temperature, and custom Groq model ID).

### 3. Exporting Data (Snapshotting)

Once the ETL pipeline has finished and you have verified the data in Qdrant, you must create a snapshot for production.

1. **Stop Qdrant Gracefully:**
   Ensure no active writes are occurring, then stop the Qdrant container:
   ```bash
   docker compose stop qdrant
   ```

2. **Create the Snapshot Tarball:**
   Compress the local Qdrant storage volume into a tarball:
   ```bash
   tar -czf qdrant_snapshot.tar.gz -C <path_to_qdrant_storage_volume> .
   ```

3. **Upload to MinIO:**
   Upload the `qdrant_snapshot.tar.gz` to the `qdrant-snapshots` bucket in your MinIO instance. Ensure you capture the SHA256 checksum of the file.

### 4. Production Migration (Init Container Strategy)

To avoid running ETL scripts on production servers, Kairos uses an **Init Container (Raw Storage Volume Sync)** approach.

When you deploy to production:

1. **Configure Production Variables:**
   In your production environment, set the `QDRANT_SNAPSHOT_SHA256` environment variable to the checksum of your newly uploaded snapshot.

2. **Deploy via Docker Compose:**
   Run the production compose file:
   ```bash
   docker compose -f docker-compose.prod.yml up -d
   ```

**How it works:**
- Before the main `qdrant` service boots, Docker launches the `qdrant-init` container.
- `qdrant-init` checks if the `/qdrant/storage` volume is empty. 
- If empty, it securely downloads the `qdrant_snapshot.tar.gz` from MinIO, verifies the SHA256 hash, and extracts the contents directly into the persistent volume.
- Once `qdrant-init` exits successfully (code `0`), the main `qdrant` service starts up instantly with all the vector data already present on disk. 
- Subsequent restarts will bypass the download step if the directory is already populated.
