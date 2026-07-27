from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile
from fastapi.responses import StreamingResponse

from fambot_backend.core.chat_orchestrator import ChatOrchestrator
from fambot_backend.core.deps import firebase_uid
from fambot_backend.schemas import (
    ChatInteractionResponse,
    ChatMessageResponse,
    ChatResponse,
    MessageResponse,
)
from fambot_backend.services.chat_history import (
    create_chat,
    list_chat_messages,
    list_chats as list_chat_sessions,
)

router = APIRouter(tags=["chats"])
_orchestrator = ChatOrchestrator()


@router.post("/chat/new", response_model=ChatResponse)
def create_chat_session(
    uid: str = Depends(firebase_uid),
) -> ChatResponse:
    chat_id = str(uuid.uuid4())
    payload = create_chat(uid, chat_id=chat_id)
    return ChatResponse(
        id=chat_id,
        title=str(payload.get("title") or "New chat"),
        created_at=_as_dt(payload.get("created_at")),
        last_updated=_as_dt(payload.get("last_updated")),
    )


@router.get("/chats", response_model=list[ChatResponse])
def list_chats(uid: str = Depends(firebase_uid)) -> list[ChatResponse]:
    payloads = list_chat_sessions(uid)
    out: list[ChatResponse] = []
    now = datetime.now(timezone.utc)
    for item in payloads:
        out.append(
            ChatResponse(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or "New chat"),
                created_at=_as_dt(item.get("created_at"), now),
                last_updated=_as_dt(item.get("last_updated"), now),
            )
        )
    return out


@router.post("/v1/chats/{chat_id}/messages", response_model=ChatMessageResponse)
def create_chat_message_v1(
    chat_id: UUID,
    request: Request,
    message: str = Form(...),
    file: UploadFile | None = File(None),
    uid: str = Depends(firebase_uid),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatMessageResponse | StreamingResponse:
    file_name, file_content_type, file_payload = _read_upload(file)
    accept_value = request.headers.get("accept", "").lower()
    idempotency_key = idempotency_key_header or None
    if "text/event-stream" in accept_value:
        return StreamingResponse(
            _new_streaming_sse(
                chat_id=str(chat_id),
                uid=uid,
                message=message,
                file_name=file_name,
                file_content_type=file_content_type,
                file_payload=file_payload,
                idempotency_key=idempotency_key,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    return _orchestrator.run_buffered(
        uid=uid,
        chat_id=str(chat_id),
        user_message=message,
        upload_name=file_name,
        upload_content_type=file_content_type,
        upload_payload=file_payload,
        idempotency_key=idempotency_key,
    )


@router.post("/chat/{chat_id}", response_model=ChatInteractionResponse, deprecated=True)
def chat_interaction(
    chat_id: UUID,
    message: str = Form(...),
    file: UploadFile | None = File(None),
    uid: str = Depends(firebase_uid),
) -> ChatInteractionResponse:
    file_name, file_content_type, file_payload = _read_upload(file)
    response = _orchestrator.run_buffered(
        uid=uid,
        chat_id=str(chat_id),
        user_message=message,
        upload_name=file_name,
        upload_content_type=file_content_type,
        upload_payload=file_payload,
        idempotency_key=None,
    )
    return ChatInteractionResponse(
        role="model",
        content=response.content,
        citations=response.citations,
        new_title=response.new_title,
    )


@router.post("/chat/{chat_id}/stream", deprecated=True)
def chat_interaction_stream(
    chat_id: UUID,
    message: str = Form(...),
    file: UploadFile | None = File(None),
    uid: str = Depends(firebase_uid),
) -> StreamingResponse:
    file_name, file_content_type, file_payload = _read_upload(file)

    return StreamingResponse(
        _legacy_streaming_sse(
            chat_id=str(chat_id),
            uid=uid,
            message=message,
            file_name=file_name,
            file_content_type=file_content_type,
            file_payload=file_payload,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/{chat_id}/history", response_model=list[MessageResponse])
def get_history(chat_id: UUID, uid: str = Depends(firebase_uid)) -> list[MessageResponse]:
    payloads = list_chat_messages(uid, str(chat_id))
    now = datetime.now(timezone.utc)
    out: list[MessageResponse] = []
    for item in payloads:
        citations = item.get("citations")
        out.append(
            MessageResponse(
                role=str(item.get("role") or "unknown"),
                content=str(item.get("content") or ""),
                created_at=_as_dt(item.get("created_at"), now),
                citations=citations if isinstance(citations, list) else None,
                has_file=item.get("has_file") if isinstance(item.get("has_file"), bool) else None,
            )
        )
    return out


def _as_dt(raw: object, fallback: datetime | None = None) -> datetime:
    if isinstance(raw, datetime):
        return raw
    return fallback or datetime.now(timezone.utc)


def _sse_event(obj: object) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


def _read_upload(file: UploadFile | None) -> tuple[str | None, str | None, bytes | None]:
    if file is None:
        return None, None, None
    return (
        file.filename,
        file.content_type or "application/octet-stream",
        file.file.read(),
    )


def _new_streaming_sse(
    *,
    chat_id: str,
    uid: str,
    message: str,
    file_name: str | None,
    file_content_type: str | None,
    file_payload: bytes | None,
    idempotency_key: str | None,
) -> Iterator[bytes]:
    for event in _orchestrator.run_stream(
        uid=uid,
        chat_id=chat_id,
        user_message=message,
        upload_name=file_name,
        upload_content_type=file_content_type,
        upload_payload=file_payload,
        idempotency_key=idempotency_key,
    ):
        yield _sse_event(event.payload)


def _legacy_streaming_sse(
    *,
    chat_id: str,
    uid: str,
    message: str,
    file_name: str | None,
    file_content_type: str | None,
    file_payload: bytes | None,
) -> Iterator[bytes]:
    citations: list[dict[str, object]] | None = None
    new_title: str | None = None
    for event in _orchestrator.run_stream(
        uid=uid,
        chat_id=chat_id,
        user_message=message,
        upload_name=file_name,
        upload_content_type=file_content_type,
        upload_payload=file_payload,
        idempotency_key=None,
    ):
        event_type = str(event.payload.get("type") or "")
        if event_type == "token":
            text = event.payload.get("text")
            if isinstance(text, str):
                yield _sse_event({"type": "text", "text": text})
        elif event_type == "message_end":
            c = event.payload.get("citations")
            t = event.payload.get("new_title")
            citations = c if isinstance(c, list) else citations
            new_title = t if isinstance(t, str) else None
            yield _sse_event({"type": "done", "new_title": new_title, "citations": citations})
        elif event_type == "error":
            detail = event.payload.get("detail")
            yield _sse_event(
                {
                    "type": "error",
                    "detail": detail if isinstance(detail, str) else "Request failed",
                }
            )
