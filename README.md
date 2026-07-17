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

3. **Install ETL Dependencies:**
   The ETL pipeline requires specific dependencies isolated from the core application. Use `uv` to install the `etl` group:
   ```bash
   uv sync --group etl
   ```

4. **Run the Notebook:**
   Launch Jupyter to execute the pipeline interactively:
   ```bash
   uv run --group etl jupyter notebook data/pubmed_qdrant_etl.ipynb
   ```
   Execute the cells in sequence. The pipeline supports incremental updates—it will automatically diff against existing PMIDs in Qdrant and only process new articles.

### 2. Exporting Data (Snapshotting)

Once the ETL pipeline has finished and you have verified the data in Qdrant (available at `http://localhost:6333/dashboard`), you must create a snapshot for production.

1. **Stop Qdrant Gracefully:**
   Ensure no active writes are occurring, then stop the Qdrant container:
   ```bash
   docker compose stop qdrant
   ```

2. **Create the Snapshot Tarball:**
   Compress the local Qdrant storage volume into a tarball. (See the helper scripts at the end of the ETL notebook for automated commands).
   ```bash
   tar -czf qdrant_snapshot.tar.gz -C <path_to_qdrant_storage_volume> .
   ```

3. **Upload to MinIO:**
   Upload the `qdrant_snapshot.tar.gz` to the `qdrant-snapshots` bucket in your MinIO instance. Ensure you capture the SHA256 checksum of the file.

### 3. Production Migration (Init Container Strategy)

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
