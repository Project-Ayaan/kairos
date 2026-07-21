import os
import torch
import anyio
from typing import List, Optional
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from groq import Groq

# Constants
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "pubmed_articles"
MEDCPT_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

class SourceChunk(BaseModel):
    """Pydantic model representing a retrieved PubMed article chunk with metadata."""
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    authors: Optional[str] = None
    chunk_idx: Optional[int] = None
    score: float = 0.0

    @property
    def pubmed_url(self) -> Optional[str]:
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}"
        return None

class KairosRAGPipeline:
    """Core RAG pipeline for retrieval and synthesis of clinical evidence."""
    
    def __init__(self, qdrant_url: str = QDRANT_URL, collection_name: str = COLLECTION_NAME):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self._qdrant_client = None

    @property
    def qdrant_client(self) -> QdrantClient:
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(url=self.qdrant_url)
        return self._qdrant_client

    def _get_encoder(self):
        if self.model is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(MEDCPT_QUERY_ENCODER)
            self.model = AutoModel.from_pretrained(MEDCPT_QUERY_ENCODER).to(self.device)
        return self.tokenizer, self.model, self.device

    def embed_query(self, query_text: str) -> List[float]:
        tok, mod, dev = self._get_encoder()
        inputs = tok(query_text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(dev)
        with torch.no_grad():
            outputs = mod(**inputs)
        # CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()
        return embedding

    def search_qdrant(self, vector: List[float], top_k: int = 5) -> List[SourceChunk]:
        client = self.qdrant_client
        if not client.collection_exists(self.collection_name):
            return []
        
        results = client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k
        )
        
        chunks = []
        for hit in results.points:
            payload = hit.payload
            chunks.append(SourceChunk(
                pmid=payload.get("pmid"),
                pmcid=payload.get("pmcid"),
                doi=payload.get("doi"),
                title=payload.get("title"),
                text=payload.get("text"),
                journal=payload.get("journal"),
                year=str(payload.get("year")) if payload.get("year") is not None else None,
                authors=payload.get("authors"),
                chunk_idx=payload.get("chunk_idx"),
                score=hit.score
            ))
        return chunks

    def generate_answer(
        self, 
        query: str, 
        chunks: List[SourceChunk], 
        groq_api_key: str,
        model_name: str = DEFAULT_GROQ_MODEL, 
        temperature: float = 0.2
    ) -> str:
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set.")
            
        client = Groq(api_key=groq_api_key)
        
        if not chunks:
            return "No relevant clinical literature found in the Qdrant database to address this query."
            
        # Format the context for the prompt
        context_str = ""
        for idx, chunk in enumerate(chunks, 1):
            context_str += f"[{idx}] Title: {chunk.title}\n"
            context_str += f"Authors: {chunk.authors} | Journal: {chunk.journal} ({chunk.year})\n"
            context_str += f"PMID: {chunk.pmid} | Relevance Score: {chunk.score:.4f}\n"
            context_str += f"Content: {chunk.text}\n\n"
            
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

    async def retrieve(self, query: str, top_k: int = 5) -> List[SourceChunk]:
        """Async wrapper for embed_query and search_qdrant to avoid blocking event loop."""
        def _retrieve():
            vector = self.embed_query(query)
            return self.search_qdrant(vector, top_k=top_k)
        
        return await anyio.to_thread.run_sync(_retrieve)

    async def synthesize(
        self, 
        query: str, 
        chunks: List[SourceChunk], 
        groq_api_key: str, 
        model_name: str = DEFAULT_GROQ_MODEL, 
        temperature: float = 0.2
    ) -> str:
        """Async wrapper for generate_answer to avoid blocking event loop."""
        def _synthesize():
            return self.generate_answer(
                query=query,
                chunks=chunks,
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature
            )
        
        return await anyio.to_thread.run_sync(_synthesize)
