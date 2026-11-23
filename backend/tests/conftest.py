import os

import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from src import app as app_module
from src import db
from utils.models import HandbookDoc


class FakeRAGService:
    """Stands in for RAGService: no Chroma, OpenAI or Groq calls (and no spend) in tests."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    async def get_rag_response(self, query, history):
        return {
            "content": f"Answer to: {query}",
            "sources": [{"docId": "doc-1", "snippet": "a snippet"}],
            "tool_steps": [
                {"tool_name": "search_handbook", "arguments": {"query": query}, "order": 1}
            ],
        }

    async def get_rag_response_streamed(self, query, history):
        import asyncio

        yield {
            "event": "tool_step",
            "data": {"tool_name": "search_handbook", "arguments": {"query": query}, "order": 1},
        }
        if self.delay:
            await asyncio.sleep(self.delay)
        yield {
            "event": "done",
            "data": {
                "content": f"Answer to: {query}",
                "sources": [{"docId": "doc-1", "snippet": "a snippet"}],
                "tool_steps": [
                    {"tool_name": "search_handbook", "arguments": {"query": query}, "order": 1}
                ],
            },
        }


@pytest.fixture(autouse=True)
def aws(monkeypatch):
    """Every test gets a fresh, in-memory DynamoDB table and a fake RAG service."""
    os.environ.update(
        AWS_ACCESS_KEY_ID="test",
        AWS_SECRET_ACCESS_KEY="test",
        AWS_REGION="us-east-1",
        TABLE_NAME="nexus-test",
    )
    for name in ("DYNAMODB_ENDPOINT", "ORIGIN_SECRET", "AWS_PROFILE", "SSM_PREFIX"):
        os.environ.pop(name, None)
    monkeypatch.setattr(app_module, "rag_service", FakeRAGService())
    monkeypatch.setattr(
        app_module,
        "handbook_docs",
        [HandbookDoc(id="doc-1", title="Holidays", category="Benefits", content="25 days")],
    )
    with mock_aws():
        db.get_table.cache_clear()
        db.create_table_if_missing()
        yield
    db.get_table.cache_clear()


@pytest.fixture
def client() -> TestClient:
    # Not used as a context manager, so the real startup (Chroma, SSM) never runs.
    return TestClient(app_module.app)
