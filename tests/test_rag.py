import sys
from unittest.mock import MagicMock

# Mock torch and transformers to avoid loading compiled DLLs in CI/sandboxed envs
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

from unittest.mock import patch

from app.core.rag import KairosRAGPipeline, SourceChunk


class FakeHit:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


def test_search_qdrant_maps_pubmed_payload_including_doi_pmcid():
    pipeline = KairosRAGPipeline()
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_client.query_points.return_value = FakeQueryResult([
        FakeHit(
            payload={
                "source_type": "pubmed",
                "pmid": "12345",
                "pmcid": "PMC98765",
                "doi": "10.1000/example.doi",
                "title": "A Study",
                "text": "Some evidence text.",
                "journal": "Journal of Testing",
                "year": 2024,
                "authors": "A. Author",
                "chunk_idx": 0,
            },
            score=0.87,
        )
    ])

    with patch.object(KairosRAGPipeline, "qdrant_client", new=fake_client):
        chunks = pipeline.search_qdrant(vector=[0.1, 0.2], top_k=1)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_type == "pubmed"
    assert chunk.doi == "10.1000/example.doi"
    assert chunk.pmcid == "PMC98765"
    assert chunk.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/12345"


def test_search_qdrant_maps_policy_document_payload():
    pipeline = KairosRAGPipeline()
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_client.query_points.return_value = FakeQueryResult([
        FakeHit(
            payload={
                "source_type": "policy_document",
                "document_name": "clinical_policy.pdf",
                "section": "Page 3",
                "title": "clinical_policy.pdf",
                "text": "Policy text content.",
                "chunk_idx": 2,
            },
            score=0.75,
        )
    ])

    with patch.object(KairosRAGPipeline, "qdrant_client", new=fake_client):
        chunks = pipeline.search_qdrant(vector=[0.1, 0.2], top_k=1)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_type == "policy_document"
    assert chunk.document_name == "clinical_policy.pdf"
    assert chunk.section == "Page 3"
    # pubmed_url must not populate for non-pubmed sources even if pmid is absent
    assert chunk.pubmed_url is None


def test_generate_answer_refuses_when_no_chunks():
    pipeline = KairosRAGPipeline()
    answer = pipeline.generate_answer(query="test", chunks=[], groq_api_key="fake-key")
    assert "No relevant" in answer


def test_generate_answer_formats_pubmed_and_policy_citations_differently():
    pipeline = KairosRAGPipeline()

    pubmed_chunk = SourceChunk(
        source_type="pubmed",
        pmid="1",
        title="Study Title",
        text="Evidence.",
        journal="Journal",
        year="2024",
        authors="Author",
        score=0.9,
    )
    policy_chunk = SourceChunk(
        source_type="policy_document",
        document_name="policy.pdf",
        section="Section 2",
        text="Policy evidence.",
        score=0.8,
    )

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Synthesized answer."

    captured_prompt = {}

    def fake_create(**kwargs):
        captured_prompt["messages"] = kwargs["messages"]
        return mock_response

    with patch("app.core.rag.Groq") as mock_groq_cls:
        mock_groq_cls.return_value.chat.completions.create.side_effect = fake_create
        result = pipeline.generate_answer(
            query="test query",
            chunks=[pubmed_chunk, policy_chunk],
            groq_api_key="fake-key",
        )

    assert result == "Synthesized answer."
    user_prompt = captured_prompt["messages"][1]["content"]
    assert "PMID: 1" in user_prompt
    assert "Document: policy.pdf" in user_prompt
    assert "Section: Section 2" in user_prompt
