import os
import sys
import re
import uuid
import time
import httpx
import xml.etree.ElementTree as ET
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Qdrant Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "pubmed_articles"

# NCBI API Config
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
PUBMED_SEARCH_TERM = os.getenv("PUBMED_SEARCH_TERM", '"clinical"[Title/Abstract] OR "medicine"[Title/Abstract] OR "therapy"[Title/Abstract] OR "diagnosis"[Title/Abstract]')
PUBMED_BATCH_LIMIT = int(os.getenv("PUBMED_BATCH_LIMIT", "1000"))

# Models
MEDCPT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"

# Sentence splitting regex
SENTENCE_END = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

def split_into_sentences(text):
    sentences = SENTENCE_END.split(text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(title, abstract_text, max_words=256, overlap_words=50):
    text = f"{title}. {abstract_text}"
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    
    for sentence in sentences:
        sent_words = sentence.split()
        if not sent_words:
            continue
        
        # If a single sentence is longer than max_words, chunk by words
        if len(sent_words) > max_words:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_word_count = 0
            for i in range(0, len(sent_words), max_words - overlap_words):
                chunk_w = sent_words[i : i + max_words]
                chunks.append(" ".join(chunk_w))
            continue
        
        if current_word_count + len(sent_words) > max_words:
            chunks.append(" ".join(current_chunk_sentences))
            # Create overlap from end of current chunk
            overlap_sentences = []
            overlap_count = 0
            for prev_sent in reversed(current_chunk_sentences):
                prev_sent_words = prev_sent.split()
                if overlap_count + len(prev_sent_words) <= overlap_words:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_count += len(prev_sent_words)
                else:
                    break
            
            current_chunk_sentences = overlap_sentences + [sentence]
            current_word_count = overlap_count + len(sent_words)
        else:
            current_chunk_sentences.append(sentence)
            current_word_count += len(sent_words)
            
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks

def fetch_pmids(term, limit):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "xml",
        "retmax": str(limit),
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
        
    print(f"Querying NCBI esearch for term: {term} (Limit: {limit})...")
    response = httpx.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    
    root = ET.fromstring(response.text)
    pmids = [id_elem.text for id_elem in root.findall(".//Id")]
    return pmids

def fetch_articles(pmids):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
        
    response = httpx.post(url, data=params, timeout=60.0)
    response.raise_for_status()
    
    root = ET.fromstring(response.text)
    articles = []
    
    for article in root.findall(".//PubmedArticle"):
        try:
            # PMID
            pmid_elem = article.find(".//MedlineCitation/PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            if not pmid:
                continue
                
            # Title
            title_elem = article.find(".//ArticleTitle")
            title_text = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
            
            # Abstract
            abstract_parts = []
            abstract_elem = article.find(".//Abstract")
            if abstract_elem is not None:
                for text_elem in abstract_elem.findall(".//AbstractText"):
                    label = text_elem.get("Label")
                    inner_text = "".join(text_elem.itertext()).strip()
                    if inner_text:
                        if label:
                            abstract_parts.append(f"{label}: {inner_text}")
                        else:
                            abstract_parts.append(inner_text)
            abstract_text = " ".join(abstract_parts)
            
            # Journal
            journal_elem = article.find(".//Journal/Title")
            journal_text = journal_elem.text if journal_elem is not None else "Unknown Journal"
            
            # Publication Year
            year_elem = article.find(".//JournalIssue/PubDate/Year")
            if year_elem is not None and year_elem.text:
                pub_year = year_elem.text
            else:
                medline_date_elem = article.find(".//JournalIssue/PubDate/MedlineDate")
                if medline_date_elem is not None and medline_date_elem.text:
                    match = re.search(r"\b(19|20)\d{2}\b", medline_date_elem.text)
                    pub_year = match.group(0) if match else medline_date_elem.text[:4]
                else:
                    pub_year = "Unknown"
            
            # Authors
            author_names = []
            author_list = article.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall(".//Author"):
                    last = author.find("LastName")
                    fore = author.find("ForeName")
                    last_name = last.text if last is not None and last.text else ""
                    fore_name = fore.text if fore is not None and fore.text else ""
                    if last_name or fore_name:
                        author_names.append(f"{fore_name} {last_name}".strip())
            authors_text = ", ".join(author_names[:5])
            if len(author_names) > 5:
                authors_text += " et al."
                
            articles.append({
                "pmid": pmid,
                "title": title_text,
                "abstract": abstract_text,
                "journal": journal_text,
                "year": pub_year,
                "authors": authors_text
            })
        except Exception as e:
            print(f"Error parsing article xml: {e}", file=sys.stderr)
            
    return articles

def main():
    print("Starting Kairos CDSS ETL Pipeline...")
    
    # 1. Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        client = QdrantClient(url=QDRANT_URL)
        # Check connectivity
        client.get_collections()
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Ensure collection exists
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        
    # 2. Fetch PMIDs
    try:
        pmids = fetch_pmids(PUBMED_SEARCH_TERM, PUBMED_BATCH_LIMIT)
        print(f"Total PMIDs returned by search: {len(pmids)}")
    except Exception as e:
        print(f"Failed to fetch PMIDs: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not pmids:
        print("No articles found to ingest. Exiting.")
        sys.exit(0)
        
    # 3. Load MedCPT Model
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
        
    # 4. Fetch, Chunk, and Embed in batches
    batch_size = 300
    all_chunks = []
    
    print("Fetching articles metadata and abstracts from NCBI...")
    for i in tqdm(range(0, len(pmids), batch_size), desc="NCBI Fetching"):
        batch_pmids = pmids[i : i + batch_size]
        try:
            articles = fetch_articles(batch_pmids)
            for art in articles:
                text_chunks = chunk_text(art["title"], art["abstract"])
                for idx, chunk in enumerate(text_chunks):
                    all_chunks.append({
                        "pmid": art["pmid"],
                        "title": art["title"],
                        "text": chunk,
                        "journal": art["journal"],
                        "year": art["year"],
                        "authors": art["authors"],
                        "chunk_idx": idx
                    })
            # Sleep to respect rate limits
            time.sleep(0.35)
        except Exception as e:
            print(f"\nError fetching batch {i//batch_size + 1}: {e}", file=sys.stderr)
            
    print(f"Generated {len(all_chunks)} text chunks from fetched articles.")
    
    # 5. Embed and Upsert
    if not all_chunks:
        print("No chunks generated. Exiting.")
        sys.exit(0)
        
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
                            "pmid": item["pmid"],
                            "title": item["title"],
                            "text": item["text"],
                            "journal": item["journal"],
                            "year": item["year"],
                            "authors": item["authors"],
                            "chunk_idx": item["chunk_idx"]
                        }
                    )
                )
            
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch_points
            )
        except Exception as e:
            print(f"\nError processing embedding batch {i//embed_batch_size + 1}: {e}", file=sys.stderr)
            
    # Verify Collection points
    try:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"\nETL completed successfully! Total points in Qdrant '{COLLECTION_NAME}': {info.points_count}")
    except Exception as e:
        print(f"ETL finished but collection validation failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
