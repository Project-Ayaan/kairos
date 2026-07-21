import sys
from unittest.mock import MagicMock, AsyncMock

# Mock torch and transformers to prevent compiled DLL loading issues
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from app.main import app, pipeline
from app.core.rag import SourceChunk
from app.core.config import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_pipeline_methods():
    """Fixture to mock RAG pipeline methods before each test."""
    pipeline.retrieve = AsyncMock(return_value=[
        SourceChunk(
            pmid="999",
            title="Evidence Title",
            text="Evidence details.",
            journal="Journal of Medicine",
            year="2026",
            authors="Dr. Tester",
            chunk_idx=0,
            score=0.9
        )
    ])
    pipeline.synthesize = AsyncMock(return_value="Mocked CDSS answer.")
    yield

def test_get_agent_card():
    # 1. Test public discovery endpoint
    response = client.get("/.well-known/agent.json")
    assert response.status_code == 200
    card_data = response.json()
    assert card_data["name"] == "Kairos CDSS"
    assert "skills" in card_data
    assert card_data["skills"][0]["id"] == "clinical_query"

def test_post_a2a_unauthenticated_when_open():
    # When settings.a2a_api_key is None, request should proceed without auth
    old_key = settings.a2a_api_key
    settings.a2a_api_key = None
    try:
        rpc_request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "SendMessage",
            "params": {
                "message": {
                    "message_id": "msg_123",
                    "role": "ROLE_USER",
                    "parts": [{"text": "Is clinical decision support useful?"}]
                }
            }
        }
        response = client.post(
            "/a2a", 
            json=rpc_request,
            headers={"a2a-version": "1.0"}
        )
        assert response.status_code == 200
        
        # Verify JSON-RPC response contains completed Task with artifacts
        res_json = response.json()
        assert "result" in res_json
        task = res_json["result"]["task"]
        assert "id" in task
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert len(task["artifacts"]) == 2
        
        # Artifact 1: Answer text
        assert task["artifacts"][0]["parts"][0]["text"] == "Mocked CDSS answer."
        # Artifact 2: Evidence sources JSON structure
        assert "sources" in task["artifacts"][1]["parts"][0]["data"]
    finally:
        settings.a2a_api_key = old_key

def test_post_a2a_auth_middleware():
    # Test middleware behaviour when A2A_API_KEY is set in settings
    old_key = settings.a2a_api_key
    settings.a2a_api_key = "secret_key_123"
    try:
        rpc_request = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "SendMessage",
            "params": {
                "message": {
                    "message_id": "msg_456",
                    "role": "ROLE_USER",
                    "parts": [{"text": "Test query"}]
                }
            }
        }
        
        # 1. Unauthenticated request -> should return 401
        response = client.post(
            "/a2a", 
            json=rpc_request,
            headers={"a2a-version": "1.0"}
        )
        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower() or "missing" in response.json()["detail"].lower()

        # 2. Authenticated request with invalid token -> should return 401
        response = client.post(
            "/a2a", 
            json=rpc_request, 
            headers={"Authorization": "Bearer wrong_token", "a2a-version": "1.0"}
        )
        assert response.status_code == 401

        # 3. Authenticated request with valid token -> should succeed (200)
        response = client.post(
            "/a2a", 
            json=rpc_request, 
            headers={"Authorization": "Bearer secret_key_123", "a2a-version": "1.0"}
        )
        assert response.status_code == 200
        assert "result" in response.json()
    finally:
        settings.a2a_api_key = old_key
