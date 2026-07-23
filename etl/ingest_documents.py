"""Ingests local PDF/DOCX reference documents (e.g. policy documents) into the
shared Kairos Qdrant knowledge collection, tagged with source_type="policy_document".

Usage:
    uv run python etl/ingest_documents.py /path/to/documents_dir
"""
import os
import sys
import uuid
import argparse
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument

from chunking import chunk_text

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kairos_knowledge")
MEDCPT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def extract_pdf_sections(path):
    """Yields (section_label, text) pairs, one per PDF page."""
    reader = PdfReader(str(path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield f"Page {page_num}", text


def extract_docx_sections(path):
    """Yields (section_label, text) pairs, grouped under the most recent heading."""
    doc = DocxDocument(str(path))
    current_heading = "Document Body"
    buffer = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            return (current_heading, text)
        return None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style is not None and para.style.name and para.style.name.startswith("Heading"):
            result = flush()
            if result:
                yield result
            buffer.clear()
            current_heading = text
        else:
            buffer.append(text)

    result = flush()
    if result:
        yield result


def extract_sections(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return list(extract_pdf_sections(path))
    if suffix == ".docx":
        return list(extract_docx_sections(path))
    raise ValueError(f"Unsupported file type: {suffix}")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF/DOCX documents into Kairos's Qdrant knowledge base.")
    parser.add_argument("directory", help="Directory containing PDF/DOCX files to ingest.")
    args = parser.parse_args()

    source_dir = Path(args.directory)
    if not source_dir.is_dir():
        print(f"Not a directory: {source_dir}", file=sys.stderr)
        sys.exit(1)

    files = [p for p in sorted(source_dir.rglob("*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        print(f"No PDF/DOCX files found under {source_dir}. Exiting.")
        sys.exit(0)

    print(f"Found {len(files)} document(s) to ingest.")

    # 1. Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}", file=sys.stderr)
        sys.exit(1)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    # 2. Load MedCPT Model (same article encoder used for PubMed content)
    print("Loading HuggingFace transformers and MedCPT model...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(MEDCPT_ARTICLE_ENCODER)
        model = AutoModel.from_pretrained(MEDCPT_ARTICLE_ENCODER).to(device)
    except Exception as e:
        print(f"Failed to load transformers/model: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Extract, chunk
    all_chunks = []
    for path in tqdm(files, desc="Extracting documents"):
        try:
            sections = extract_sections(path)
        except Exception as e:
            print(f"\nFailed to extract {path.name}: {e}", file=sys.stderr)
            continue

        for section_label, section_text in sections:
            text_chunks = chunk_text(title=None, body_text=section_text)
            for idx, chunk in enumerate(text_chunks):
                all_chunks.append({
                    "document_name": path.name,
                    "section": section_label,
                    "text": chunk,
                    "chunk_idx": idx,
                })

    print(f"Generated {len(all_chunks)} text chunks from {len(files)} document(s).")
    if not all_chunks:
        print("No chunks generated. Exiting.")
        sys.exit(0)

    # 4. Embed and Upsert
    embed_batch_size = 16
    print("Embedding chunks and preparing Qdrant upserts...")
    for i in tqdm(range(0, len(all_chunks), embed_batch_size), desc="Embedding & Uploading"):
        batch = all_chunks[i : i + embed_batch_size]
        batch_texts = [item["text"] for item in batch]

        try:
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()

            batch_points = []
            for item, emb in zip(batch, embeddings):
                point_id = str(uuid.uuid4())
                batch_points.append(
                    PointStruct(
                        id=point_id,
                        vector=emb,
                        payload={
                            "source_type": "policy_document",
                            "document_name": item["document_name"],
                            "section": item["section"],
                            "title": item["document_name"],
                            "text": item["text"],
                            "chunk_idx": item["chunk_idx"],
                        }
                    )
                )

            client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
        except Exception as e:
            print(f"\nError processing embedding batch {i // embed_batch_size + 1}: {e}", file=sys.stderr)

    try:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"\nIngestion completed successfully! Total points in Qdrant '{COLLECTION_NAME}': {info.points_count}")
    except Exception as e:
        print(f"Ingestion finished but collection validation failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
