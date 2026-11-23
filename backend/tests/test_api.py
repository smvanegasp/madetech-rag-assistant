import json

from boto3.dynamodb.conditions import Key

from src import app as app_module
from src import db
from tests.conftest import FakeRAGService


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if not lines:
            continue
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events


def test_health(client):
    assert client.get("/api/health").json() == {"status": "healthy", "documents_loaded": 1}


def test_handbook(client):
    docs = client.get("/api/handbook").json()
    assert docs == [
        {"id": "doc-1", "title": "Holidays", "category": "Benefits", "content": "25 days"}
    ]


def test_chat(client):
    body = client.post("/api/chat", json={"query": "holidays?", "history": []}).json()
    assert body["content"] == "Answer to: holidays?"
    assert body["sources"] == [{"docId": "doc-1", "snippet": "a snippet"}]
    assert body["chat_id"] and body["interaction_id"]


def test_stream_sends_tool_steps_then_done(client):
    response = client.post(
        "/api/chat/stream", json={"query": "holidays?", "history": [], "chat_id": "c1"}
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["tool_step", "done"]
    assert events[0][1]["tool_name"] == "search_handbook"
    done = events[1][1]
    assert done["content"] == "Answer to: holidays?"
    assert done["chat_id"] == "c1"
    assert "isError" not in done


def test_stream_heartbeat_while_agent_is_quiet(client, monkeypatch):
    monkeypatch.setattr(app_module, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(app_module, "rag_service", FakeRAGService(delay=0.3))
    response = client.post("/api/chat/stream", json={"query": "q", "history": []})
    assert ": ping\n\n" in response.text
    assert [name for name, _ in parse_sse(response.text)] == ["tool_step", "done"]


def test_chat_is_logged(client):
    client.post("/api/chat/stream", json={"query": "holidays?", "history": [], "chat_id": "c1"})
    items = db.get_table().query(KeyConditionExpression=Key("pk").eq("CHAT#c1"))["Items"]
    assert len(items) == 1
    item = items[0]
    assert item["sk"].startswith("MSG#")
    assert item["user_message"] == "holidays?"
    assert item["llm_response"] == "Answer to: holidays?"
    assert item["ttl"] > 0


def test_chat_not_logged_without_table(client, monkeypatch):
    monkeypatch.delenv("TABLE_NAME")
    assert client.post("/api/chat", json={"query": "q", "history": []}).status_code == 200


def test_origin_secret_required(client, monkeypatch):
    monkeypatch.setenv("ORIGIN_SECRET", "s3cret")
    assert client.get("/api/health").status_code == 403
    assert client.get("/api/health", headers={"X-Origin-Verify": "wrong"}).status_code == 403
    assert client.get("/api/health", headers={"X-Origin-Verify": "s3cret"}).status_code == 200


def test_chat_daily_cap_per_ip(client, monkeypatch):
    monkeypatch.setenv("CHATS_PER_IP_PER_DAY", "2")
    headers = {"CloudFront-Viewer-Address": "203.0.113.7:4444"}
    for _ in range(2):
        r = client.post("/api/chat/stream", json={"query": "q", "history": []}, headers=headers)
        assert parse_sse(r.text)[-1][1]["content"] == "Answer to: q"

    r = client.post("/api/chat/stream", json={"query": "q", "history": []}, headers=headers)
    events = parse_sse(r.text)
    assert [name for name, _ in events] == ["done"]
    assert "limit" in events[0][1]["content"]
    assert "isError" not in events[0][1]  # shown as an answer; the UI doesn't retry it

    # Another visitor is unaffected.
    other = {"CloudFront-Viewer-Address": "198.51.100.1:1"}
    r = client.post("/api/chat", json={"query": "q", "history": []}, headers=other)
    assert r.json()["content"] == "Answer to: q"


def test_chat_global_daily_cap(client, monkeypatch):
    monkeypatch.setenv("CHATS_PER_DAY", "1")
    assert client.post("/api/chat", json={"query": "q", "history": []}).json()["content"] == (
        "Answer to: q"
    )
    assert "limit" in client.post("/api/chat", json={"query": "q", "history": []}).json()["content"]


def test_contact_daily_cap(client, monkeypatch):
    sent = []
    monkeypatch.setattr(app_module, "send_contact_email", lambda **kw: sent.append(kw))
    monkeypatch.setenv("CONTACTS_PER_IP_PER_DAY", "1")
    body = {"contact_type": "feedback", "name": "A", "email": "a@example.com", "message": "hi"}
    assert client.post("/api/contact", json=body).status_code == 204
    assert client.post("/api/contact", json=body).status_code == 429
    assert len(sent) == 1
