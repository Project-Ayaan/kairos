import sys
from unittest.mock import MagicMock, AsyncMock

# Mock torch and transformers to prevent loading compiled DLLs (which may be blocked by system policies)
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

import pytest
from google.protobuf.struct_pb2 import Value
from google.protobuf.json_format import MessageToDict

from a2a.types import TaskState, Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent
from app.a2a.executor import KairosAgentExecutor
from app.core.rag import SourceChunk

class MockEventQueue:
    """Mock EventQueue to capture enqueued events."""
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event) -> None:
        self.events.append(event)

class MockRequestContext:
    """Mock RequestContext for execution context."""
    def __init__(self, task_id="task_123", context_id="ctx_456", user_input="What is Cdss?"):
        self.task_id = task_id
        self.context_id = context_id
        self.user_input = user_input

    def get_user_input(self, delimiter="\n") -> str:
        return self.user_input

@pytest.mark.asyncio
async def test_kairos_executor_success():
    # 1. Setup mock pipeline returning dummy chunks and answer
    mock_pipeline = MagicMock()
    mock_pipeline.retrieve = AsyncMock(return_value=[
        SourceChunk(
            pmid="12345",
            title="PubMed Article title",
            text="Evidence details",
            journal="Clinical Journal",
            year="2025",
            authors="John Doe",
            chunk_idx=0,
            score=0.95
        )
    ])
    mock_pipeline.synthesize = AsyncMock(return_value="Synthesized answer text.")

    # 2. Instantiate executor
    executor = KairosAgentExecutor(pipeline=mock_pipeline)

    # 3. Create mocks
    mock_ctx = MockRequestContext()
    mock_queue = MockEventQueue()

    # 4. Run execute
    await executor.execute(mock_ctx, mock_queue)

    # 5. Assert pipeline calls
    mock_pipeline.retrieve.assert_called_once_with(query="What is Cdss?", top_k=5)
    mock_pipeline.synthesize.assert_called_once()

    # 6. Verify enqueued events
    events = mock_queue.events
    assert len(events) == 4

    # Event 1: TASK_STATE_WORKING status update
    assert isinstance(events[0], Task)
    assert events[0].id == "task_123"
    assert events[0].status.state == TaskState.TASK_STATE_WORKING

    # Event 2: Answer artifact update
    assert isinstance(events[1], TaskArtifactUpdateEvent)
    assert events[1].task_id == "task_123"
    assert events[1].artifact.name == "Synthesized Clinical Answer"
    assert events[1].artifact.parts[0].text == "Synthesized answer text."

    # Event 3: Sources artifact update
    assert isinstance(events[2], TaskArtifactUpdateEvent)
    assert events[2].task_id == "task_123"
    assert events[2].artifact.name == "Retrieved Evidence Sources"
    # Convert protobuf struct data back to dict to verify
    data_dict = MessageToDict(events[2].artifact.parts[0].data)
    assert "sources" in data_dict
    assert len(data_dict["sources"]) == 1
    assert data_dict["sources"][0]["pmid"] == "12345"
    assert data_dict["sources"][0]["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/12345"
    assert data_dict["sources"][0]["similarity_score"] == 0.95

    # Event 4: TASK_STATE_COMPLETED status update
    assert isinstance(events[3], TaskStatusUpdateEvent)
    assert events[3].task_id == "task_123"
    assert events[3].status.state == TaskState.TASK_STATE_COMPLETED


@pytest.mark.asyncio
async def test_kairos_executor_failure():
    # 1. Setup mock pipeline that raises an exception
    mock_pipeline = MagicMock()
    mock_pipeline.retrieve = AsyncMock(side_effect=Exception("Database connection failure"))

    executor = KairosAgentExecutor(pipeline=mock_pipeline)
    mock_ctx = MockRequestContext()
    mock_queue = MockEventQueue()

    # 2. Execute should propagate exception and emit FAILED event
    with pytest.raises(Exception, match="Database connection failure"):
        await executor.execute(mock_ctx, mock_queue)

    # 3. Verify enqueued events
    events = mock_queue.events
    assert len(events) == 2

    # Event 1: TASK_STATE_WORKING
    assert isinstance(events[0], Task)
    assert events[0].status.state == TaskState.TASK_STATE_WORKING

    # Event 2: TASK_STATE_FAILED
    assert isinstance(events[1], TaskStatusUpdateEvent)
    assert events[1].task_id == "task_123"
    assert events[1].status.state == TaskState.TASK_STATE_FAILED


@pytest.mark.asyncio
async def test_kairos_executor_cancel():
    mock_pipeline = MagicMock()
    executor = KairosAgentExecutor(pipeline=mock_pipeline)
    mock_ctx = MockRequestContext()
    mock_queue = MockEventQueue()

    await executor.cancel(mock_ctx, mock_queue)

    events = mock_queue.events
    assert len(events) == 1
    assert isinstance(events[0], TaskStatusUpdateEvent)
    assert events[0].status.state == TaskState.TASK_STATE_CANCELED
