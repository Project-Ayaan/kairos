from typing import Optional, List
from pydantic import BaseModel, Field

class SourceMetadata(BaseModel):
    """Structured citation metadata for a retrieved knowledge chunk.

    Covers both PubMed literature (source_type="pubmed") and other reference
    material such as policy documents (source_type="policy_document").
    """
    source_type: str = "pubmed"
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    pubmed_url: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    document_name: Optional[str] = None
    section: Optional[str] = None
    similarity_score: float = 0.0
    chunk_idx: Optional[int] = None

class ClinicalQueryRequest(BaseModel):
    """Payload representing a request to query the CDSS RAG endpoint directly via REST."""
    query: str
    top_k: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    model: str = "openai/gpt-oss-20b"

class KairosResponse(BaseModel):
    """JSON response for direct REST queries to the CDSS."""
    answer: str
    sources: List[SourceMetadata]
    query: str
