from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

from fastapi import HTTPException

from fambot_backend.services.document_storage import get_user_document_payload, list_user_documents
from fambot_backend.services.firestore_users import get_user_profile
from fambot_backend.services.family_invites import family_peers_for_scoring

CHAT_ASSISTANT_FALLBACK = (
    "I wasn't able to generate a response just now. Please try again in a moment."
)

CHAT_SYSTEM_INSTRUCTION = dedent(
        """
        ## Role
        You are **Fambot**: a helpful, friendly health companion focused on **cardiovascular wellness**
        and everyday lifestyle habits. You sound like a knowledgeable friend who cares—never cold,
        preachy, or like a textbook.

        ## Response length and tone (critical)
        - **Default to short**: Most replies should be roughly **3–8 sentences** or the equivalent
          in chat-style lines. Lead with the directly useful answer; trim repetition.
        - **Stay conversational**: Plain language, natural pacing, and **one optional follow-up
          question** when it genuinely helps (e.g. clarifying a symptom or goal)—not an interrogation.
        - **Avoid walls of text**: No long essays, dense bullet farms, or multi-section reports unless
          the user clearly asks for detail, a plan, or “explain more.” If they want depth, you may go
          longer in structured sections—but still stay readable.
        - **No filler**: Skip throat-clearing (“That’s a great question”), redundant disclaimers
          repeated every sentence, or repeating the same caveat in different words.
        - **Help first**: Answer what they asked; then add only the smallest extra context that
          improves safety or usefulness.

        ## Medical boundaries
        - You are **not** a substitute for a clinician or emergency services. **Do not diagnose**
          from vague or incomplete information.
        - For alarming or unclear cardiac/neuro symptoms, new severe symptoms, medication changes,
          or emergencies: urge appropriate **in-person or emergency care** clearly and briefly.
        - End substantive medical or lifestyle guidance with **one short** line that this is
          educational, not a diagnosis, and not a substitute for professional or emergency care.

        ## Context and tools
        - Each user message includes **USER_PROFILE_AND_RISK** and **CHAT_HISTORY**: treat the profile
          block as your **baseline**; don’t ignore it unless the user overrides it for the turn.
        - When document detail matters, **use function tools** before guessing: **list** stored
          document names, **include** a specific file by exact name, and use **family risk context**
          when shared lifestyle matters.
        - **Never claim** you saw a document or value you did not retrieve via tools or attachment.
        - In this chat path you **do not** have live web search; rely on profile, function tools,
          uploaded attachments, and general evidence-based guidance.

        ## Style details
        - Prefer **short paragraphs** (1–3 sentences each). Use bullets **only** when listing steps,
          options, or when the user asked for a list.
        - Mirror the user’s energy: brief question → brief answer; worried user → warm and steady.
        - If something is uncertain, say so in one line and say what *would* clarify it.
        """
).strip()

_MAX_TOOL_ROUNDS = 8


@dataclass
class _ToolExecResult:
    response_json: str
    file_ref: Any | None = None


def _get_client() -> Any:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY required")
    from google import genai

    return genai.Client(api_key=api_key)


def _upload_bytes(client: Any, *, file_name: str, content_type: str, payload: bytes) -> Any:
    suffix = Path(file_name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        temp_name = tmp.name
    try:
        return client.files.upload(file=temp_name, config={"mime_type": content_type})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini file upload failed: {exc}") from exc
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def _profile_context(uid: str) -> str:
    profile = get_user_profile(uid)
    profile_data = profile.model_dump(mode="json", exclude_none=True)
    lines = [f"- {k}: {v}" for k, v in sorted(profile_data.items()) if v is not None]
    return "\n".join(lines) if lines else "(No profile data)"


def _user_profile_and_risk_block(uid: str) -> str:
    return dedent(
        f"""
        USER_PROFILE_AND_RISK (authoritative app snapshot; includes risk score, vitals, habits when present):
        {_profile_context(uid)}
        """
    ).strip()


def _list_stored_documents_json(uid: str) -> str:
    try:
        items = list_user_documents(uid)
    except Exception as exc:
        return json.dumps({"error": "Could not list documents", "detail": str(exc)[:200]})
    out: list[dict[str, Any]] = []
    for item in items:
        u = item.get("updated_at")
        updated = u.isoformat() if hasattr(u, "isoformat") else str(u) if u else None
        out.append(
            {
                "file_name": str(item.get("file_name") or ""),
                "size_bytes": int(item.get("size_bytes") or 0),
                "content_type": str(item.get("content_type") or ""),
                "updated_at": updated,
            }
        )
    return json.dumps({"stored_documents": out, "count": len(out)})


def _family_lifestyle_risk_json(uid: str) -> str:
    try:
        peers = family_peers_for_scoring(uid)
    except Exception as exc:
        return json.dumps({"error": "Could not load family context", "detail": str(exc)[:200]})
    if not peers:
        return json.dumps(
            {
                "family_members": [],
                "note": "No family group or no other members in the user’s app account.",
            }
        )
    rows: list[dict[str, Any]] = []
    for peer_uid, role in peers:
        prof = get_user_profile(peer_uid)
        rel = str(role) if role is not None else "peer"
        rows.append(
            {
                "relationship_to_me": rel,
                "display_name": prof.display_name,
                "onboarding_complete": prof.onboarding_complete,
                "risk_score": prof.risk_score,
                "risk_class": prof.risk_class,
            }
        )
    return json.dumps({"family_members": rows, "count": len(rows)})


def _include_stored_document_by_name(uid: str, file_name: str) -> _ToolExecResult:
    if not file_name or not str(file_name).strip():
        return _ToolExecResult(response_json=json.dumps({"error": "file_name required"}))
    target = str(file_name).strip()
    items = list_user_documents(uid)
    match: dict[str, Any] | None = None
    for item in items:
        if str(item.get("file_name") or "") == target:
            match = item
            break
    if match is None:
        return _ToolExecResult(
            response_json=json.dumps(
                {
                    "error": "file_not_found",
                    "file_name": target,
                    "available": [str(i.get("file_name") or "") for i in items],
                }
            )
        )
    sp = match.get("storage_path")
    if not isinstance(sp, str) or not sp:
        return _ToolExecResult(response_json=json.dumps({"error": "storage path missing"}))
    try:
        payload = get_user_document_payload(sp)
    except Exception as exc:
        return _ToolExecResult(
            response_json=json.dumps({"error": "read_failed", "detail": str(exc)[:200]})
        )
    client = _get_client()
    file_ref = _upload_bytes(
        client,
        file_name=target,
        content_type=str(match.get("content_type") or "application/octet-stream"),
        payload=payload,
    )
    return _ToolExecResult(
        response_json=json.dumps(
            {
                "ok": True,
                "file_name": target,
                "message": "The file is now attached in the follow-up for this turn.",
            }
        ),
        file_ref=file_ref,
    )


def _tool_dispatch(uid: str, name: str, args: object) -> _ToolExecResult:
    args = args if isinstance(args, dict) else {}
    if name == "list_my_stored_documents":
        return _ToolExecResult(response_json=_list_stored_documents_json(uid))
    if name == "get_family_lifestyle_risk_context":
        return _ToolExecResult(response_json=_family_lifestyle_risk_json(uid))
    if name == "include_stored_document":
        fn = args.get("file_name")
        if not isinstance(fn, str):
            return _ToolExecResult(
                response_json=json.dumps({"error": "file_name (string) required"}),
            )
        return _include_stored_document_by_name(uid, fn)
    return _ToolExecResult(response_json=json.dumps({"error": f"Unknown tool: {name}"}))


def _function_declarations() -> list[Any]:
    from google.genai import types

    return [
        types.FunctionDeclaration(
            name="list_my_stored_documents",
            description=(
                "List the user’s uploaded health document file names, sizes, and last-updated time from "
                "app storage. Call this before referring to a specific file by name."
            ),
        ),
        types.FunctionDeclaration(
            name="get_family_lifestyle_risk_context",
            description=(
                "Retrieves a summary of family group members’ stored app risk scores and relationship "
                "labels (as recorded in Fambot). Use when lifestyle advice may relate to family history "
                "or shared risk in the app."
            ),
        ),
        types.FunctionDeclaration(
            name="include_stored_document",
            description=(
                "Load the contents of a stored user document by its exact `file_name` (from list "
                "results). The tool attaches the file for the rest of the turn so it can be cited directly."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Exact `file_name` for a document returned by list_my_stored_documents.",
                    },
                },
                "required": ["file_name"],
            },
        ),
    ]


def _build_tools_list(uid: str) -> list[Any]:
    from google.genai import types

    # Gemini rejects requests that combine built-in tools such as `file_search`
    # with custom `function_declarations`. Chat relies on functions for stored
    # documents and family context, so keep the request on the function-calling path.
    return [types.Tool(function_declarations=_function_declarations())]


def _citations_from_response(response: Any) -> list[dict[str, Any]] | None:
    cands = getattr(response, "candidates", None) or []
    if not cands:
        return None
    out: list[dict[str, Any]] = []
    for i, c in enumerate(cands):
        for attr, typ in (("grounding_metadata", "grounding_metadata"), ("citation_metadata", "citation_metadata")):
            gm = getattr(c, attr, None)
            if gm is None:
                continue
            try:
                d: Any = gm.model_dump(mode="json", exclude_none=True) if hasattr(gm, "model_dump") else str(gm)
            except Exception:
                d = str(gm)
            out.append({"type": typ, "index": i, "data": d})
    return out or None


def _user_message_text(
    *, uid: str, user_message: str, history: list[dict[str, Any]] | None
) -> str:
    history = history or []
    transcript_lines: list[str] = []
    for row in history[-20:]:
        role = str(row.get("role") or "user")
        content = str(row.get("content") or "").strip()
        if content:
            transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(No history)"
    return dedent(
        f"""
        {_user_profile_and_risk_block(uid)}

        CHAT_HISTORY:
        {transcript}

        USER_MESSAGE:
        {user_message}
        """
    ).strip()


def _part_from_genai_upload(uploaded: Any) -> Any:
    from google.genai import types

    uri = getattr(uploaded, "uri", None) or getattr(uploaded, "name", None)
    if uri is None:
        return uploaded
    mt = getattr(uploaded, "mime_type", None) or "application/octet-stream"
    return types.Part(
        file_data=types.FileData(file_uri=str(uri), mime_type=mt),
    )


def _fr_payload(result: _ToolExecResult) -> dict[str, Any]:
    try:
        j = json.loads(result.response_json)
        if isinstance(j, dict):
            return j
    except Exception:
        pass
    return {"result": result.response_json}


def run_chat_text_and_citations(
    *,
    uid: str,
    user_message: str,
    history: list[dict[str, Any]] | None,
    upload_name: str | None,
    upload_content_type: str | None,
    upload_payload: bytes | None,
) -> tuple[str, str, list[dict[str, Any]] | None]:
    """Run one chat turn (optional File Search + function tools). Public for SSE handler."""
    return _run_chat_tool_loop(
        uid=uid,
        user_message=user_message,
        history=history,
        upload_name=upload_name,
        upload_content_type=upload_content_type,
        upload_payload=upload_payload,
    )


def _run_chat_tool_loop(
    *,
    uid: str,
    user_message: str,
    history: list[dict[str, Any]] | None,
    upload_name: str | None,
    upload_content_type: str | None,
    upload_payload: bytes | None,
) -> tuple[str, str, list[dict[str, Any]] | None]:
    from google.genai import types

    client = _get_client()
    model_name = _model_name()
    tools = _build_tools_list(uid)
    user_text = _user_message_text(uid=uid, user_message=user_message, history=history)

    user_parts: list[Any] = [types.Part(text=user_text)]
    if upload_payload:
        up = _upload_bytes(
            client,
            file_name=upload_name or "attachment.bin",
            content_type=upload_content_type or "application/octet-stream",
            payload=upload_payload,
        )
        user_parts.append(_part_from_genai_upload(up))
    contents: list[types.Content] = [types.Content(role="user", parts=user_parts)]
    config = types.GenerateContentConfig(
        system_instruction=CHAT_SYSTEM_INSTRUCTION,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    last: Any = None
    for _ in range(_MAX_TOOL_ROUNDS):
        last = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        cands = getattr(last, "candidates", None) or []
        fcall: Any = None
        fcall_id: str | None = None
        if cands and cands[0].content and cands[0].content.parts:
            for p in cands[0].content.parts:
                fc = getattr(p, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    fcall = fc
                    fcall_id = getattr(fc, "id", None) or "call"
                    break
        if fcall is None:
            t = (last.text or "").strip() or CHAT_ASSISTANT_FALLBACK
            return model_name, t, _citations_from_response(last)

        if not cands[0].content or not cands[0].content.parts:
            break
        contents.append(
            types.Content(
                role="model",
                parts=list(cands[0].content.parts),
            )
        )

        name = str(fcall.name or "")
        args = fcall.args if isinstance(getattr(fcall, "args", None), dict) else {}
        t_result = _tool_dispatch(uid, name, args)
        fr = types.FunctionResponse(
            id=fcall_id,
            name=name,
            response=_fr_payload(t_result),
        )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(function_response=fr)],
            )
        )
        if t_result.file_ref is not None and name == "include_stored_document":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        _part_from_genai_upload(t_result.file_ref),
                        types.Part(
                            text="Incorporate this file when replying. Summarize or quote only what is relevant; "
                            "do not invent values not in the file."
                        ),
                    ],
                )
            )
    if last is None:
        return model_name, CHAT_ASSISTANT_FALLBACK, None
    out = (last.text or "").strip() or CHAT_ASSISTANT_FALLBACK
    return model_name, out, _citations_from_response(last)


def _model_name() -> str:
    return os.environ.get("GEMINI_REPORT_MODEL", "gemini-2.5-flash")


def _chat_title_model_name() -> str:
    return os.environ.get("GEMINI_CHAT_TITLE_MODEL", "gemini-2.5-flash-lite")


def maybe_new_chat_title(
    *, user_message: str, history: list[dict[str, Any]] | None = None
) -> str | None:
    history = history or []
    if any(str(row.get("role")) == "user" for row in history):
        return None
    client = _get_client()
    model_name = _chat_title_model_name()
    try:
        title_response = client.models.generate_content(
            model=model_name,
            contents=(
                "Generate a short chat title (max 5 words, plain text only) for: "
                f"{user_message}"
            ),
        )
    except Exception:
        return None
    candidate = (title_response.text or "").strip().strip('"').strip("'")
    return candidate or None


def analyze_uploaded_document(
    *,
    uid: str,
    file_name: str,
    content_type: str,
    payload: bytes,
) -> dict[str, str]:
    client = _get_client()
    model_name = _model_name()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    uploaded = _upload_bytes(
        client,
        file_name=file_name,
        content_type=content_type or "application/octet-stream",
        payload=payload,
    )
    context_block = _profile_context(uid)
    prompt = dedent(
        """
        Analyze this medical document and provide practical prevention and lifestyle guidance.
        Focus on clear, non-alarmist language and mention when medical follow-up is recommended.
        End with a brief disclaimer that this is not a diagnosis.
        """
    ).strip()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                uploaded,
                f"USER PROFILE:\n{context_block}\n\n{prompt}",
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Document analysis failed: {exc}") from exc
    analysis = (response.text or "").strip()
    if not analysis:
        raise HTTPException(status_code=502, detail="Gemini returned empty analysis")
    return {"model": model_name, "analysis": analysis}


def analyze_stored_document(*, uid: str, doc_id: str) -> dict[str, str]:
    from fambot_backend.services.document_storage import get_user_document

    doc = get_user_document(uid, doc_id)
    storage_path = doc.get("storage_path")
    if not isinstance(storage_path, str) or not storage_path:
        raise HTTPException(status_code=500, detail="Document storage path missing")
    payload = get_user_document_payload(storage_path)
    return analyze_uploaded_document(
        uid=uid,
        file_name=str(doc.get("filename") or doc_id),
        content_type=str(doc.get("content_type") or "application/octet-stream"),
        payload=payload,
    )


def generate_chat_turn(
    *,
    uid: str,
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    upload_name: str | None = None,
    upload_content_type: str | None = None,
    upload_payload: bytes | None = None,
) -> dict[str, Any]:
    try:
        model_name, text, citations = run_chat_text_and_citations(
            uid=uid,
            user_message=user_message,
            history=history,
            upload_name=upload_name,
            upload_content_type=upload_content_type,
            upload_payload=upload_payload,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat analysis failed: {exc}") from exc
    return {
        "model": model_name,
        "content": text,
        "citations": citations,
        "new_title": maybe_new_chat_title(user_message=user_message, history=history),
    }


def generate_chat_turn_stream(
    *,
    uid: str,
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    upload_name: str | None = None,
    upload_content_type: str | None = None,
    upload_payload: bytes | None = None,
) -> Iterator[str]:
    try:
        _m, text, _c = run_chat_text_and_citations(
            uid=uid,
            user_message=user_message,
            history=history,
            upload_name=upload_name,
            upload_content_type=upload_content_type,
            upload_payload=upload_payload,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat analysis failed: {exc}") from exc
    step = max(1, min(48, max(len(text) // 48, 1)))
    for i in range(0, len(text), step):
        yield text[i : i + step]


def chat_with_documents(uid: str, user_query: str | None = None) -> dict[str, str]:
    result = generate_chat_turn(uid=uid, user_message=user_query or "Summarize my health.", history=[])
    return {
        "recommendations_text": str(result.get("content") or "").strip(),
        "model": str(result.get("model") or _model_name()),
        "query_used": user_query or "General Summary",
    }
