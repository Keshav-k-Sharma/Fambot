"""API tests for SSE chat streaming."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fambot_backend.core.chat_orchestrator import StreamEvent

@pytest.mark.api
def test_chat_stream_returns_sse_text_and_done(
    client: TestClient,
    dry_api_env: None,
) -> None:
    events = [
        StreamEvent(payload={"type": "message_start", "chatId": "c1", "turnId": "t1", "sequence": 1, "timestamp": "now"}),
        StreamEvent(payload={"type": "token", "chatId": "c1", "turnId": "t1", "sequence": 2, "timestamp": "now", "text": "Hello"}),
        StreamEvent(
            payload={
                "type": "message_end",
                "chatId": "c1",
                "turnId": "t1",
                "sequence": 3,
                "timestamp": "now",
                "citations": None,
                "new_title": None,
                "state": "completed",
            }
        ),
    ]
    with patch("fambot_backend.api.routers.chats._orchestrator.run_stream", return_value=iter(events)):
        r = client.post("/chat/00000000-0000-0000-0000-0000000000c1/stream", data={"message": "hi"})

    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/event-stream")
    assert r.headers.get("cache-control") == "no-cache"
    body = r.text
    assert "data:" in body
    assert '"type"' in body and "text" in body
    assert "Hello" in body
    assert "done" in body


@pytest.mark.api
def test_chat_stream_error_event(
    client: TestClient,
    dry_api_env: None,
) -> None:
    events = [
        StreamEvent(payload={"type": "message_start", "chatId": "c1", "turnId": "t1", "sequence": 1, "timestamp": "now"}),
        StreamEvent(payload={"type": "error", "chatId": "c1", "turnId": "t1", "sequence": 2, "timestamp": "now", "detail": "broken"}),
    ]
    with patch("fambot_backend.api.routers.chats._orchestrator.run_stream", return_value=iter(events)):
        r = client.post("/chat/00000000-0000-0000-0000-0000000000c1/stream", data={"message": "hi"})
    assert r.status_code == 200
    assert '"type": "error"' in r.text
    assert "broken" in r.text


@pytest.mark.api
def test_chat_message_v1_stream_contract(
    client: TestClient,
    dry_api_env: None,
) -> None:
    events = [
        StreamEvent(payload={"type": "message_start", "chatId": "c5", "turnId": "tx", "sequence": 1, "timestamp": "now"}),
        StreamEvent(payload={"type": "token", "chatId": "c5", "turnId": "tx", "sequence": 2, "timestamp": "now", "text": "A"}),
        StreamEvent(payload={"type": "tool_call", "chatId": "c5", "turnId": "tx", "sequence": 3, "timestamp": "now", "name": "list_my_stored_documents"}),
        StreamEvent(payload={"type": "message_end", "chatId": "c5", "turnId": "tx", "sequence": 4, "timestamp": "now", "state": "completed"}),
    ]
    with patch("fambot_backend.api.routers.chats._orchestrator.run_stream", return_value=iter(events)) as run_stream:
        r = client.post(
            "/v1/chats/00000000-0000-0000-0000-0000000000c5/messages",
            data={"message": "hello"},
            headers={"Accept": "text/event-stream"},
        )
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("text/event-stream")
    assert '"type": "tool_call"' in r.text
    assert '"chatId": "c5"' in r.text
    assert '"turnId": "tx"' in r.text
    # Regression: streaming must pass chat_id as str (same as buffered), not UUID.
    assert run_stream.call_args.kwargs["chat_id"] == "00000000-0000-0000-0000-0000000000c5"
    assert isinstance(run_stream.call_args.kwargs["chat_id"], str)
