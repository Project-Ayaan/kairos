import os
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from groq import Groq

# Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "pubmed_articles"
MEDCPT_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

# Lazy-loaded encoder model
tokenizer = None
model = None
device = None

def get_encoder():
    global tokenizer, model, device
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_ENCODER)
        model = AutoModel.from_pretrained(MEDCPT_QUERY_ENCODER).to(device)
    return tokenizer, model, device

def embed_query(query_text):
    tok, mod, dev = get_encoder()
    inputs = tok(query_text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(dev)
    with torch.no_grad():
        outputs = mod(**inputs)
    # CLS token embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()
    return embedding

def search_qdrant(vector, top_k=5):
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION_NAME):
        return []
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k
    )
    
    chunks = []
    for hit in results.points:
        payload = hit.payload
        chunks.append({
            "pmid": payload.get("pmid"),
            "title": payload.get("title"),
            "text": payload.get("text"),
            "journal": payload.get("journal"),
            "year": payload.get("year"),
            "authors": payload.get("authors"),
            "chunk_idx": payload.get("chunk_idx"),
            "score": hit.score
        })
    return chunks

def generate_answer(query, chunks, model_name=DEFAULT_GROQ_MODEL, temperature=0.2):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in your .env file or environment.")
        
    client = Groq(api_key=api_key)
    
    if not chunks:
        return "No relevant clinical literature found in the Qdrant database to address this query."
        
    # Format the context for the prompt
    context_str = ""
    for idx, chunk in enumerate(chunks, 1):
        context_str += f"[{idx}] Title: {chunk['title']}\n"
        context_str += f"Authors: {chunk['authors']} | Journal: {chunk['journal']} ({chunk['year']})\n"
        context_str += f"PMID: {chunk['pmid']} | Relevance Score: {chunk['score']:.4f}\n"
        context_str += f"Content: {chunk['text']}\n\n"
        
    system_prompt = (
        "You are Kairos, an advanced Clinical Decision Support System (CDSS) assistant.\n"
        "Your role is to answer the clinician's query using ONLY the provided clinical evidence context from PubMed.\n\n"
        "Strict Guidelines:\n"
        "1. GROUNDEDNESS: Every claim you make MUST be directly supported by the retrieved context. Do NOT extrapolate or introduce external facts.\n"
        "2. INLINE CITATIONS: Reference source articles using inline numbers corresponding to the context index (e.g., [1], [2]).\n"
        "3. REFUSAL POLICY: If the retrieved context is insufficient or does not contain direct evidence to answer the query, clearly state that you cannot answer based on the retrieved literature. Do not attempt to answer using pre-trained clinical assumptions.\n"
        "4. TONE: Professional, precise, and objective clinical language."
    )
    
    user_prompt = (
        f"Retrieved Clinical Evidence Context:\n{context_str}\n"
        f"Clinical Query: {query}\n\n"
        "Grounded Clinical Synthesis:"
    )
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=1024
    )
    
    return response.choices[0].message.content
