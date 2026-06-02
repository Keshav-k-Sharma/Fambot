"""Gemini orchestration with client and storage I/O mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from fambot_backend.providers.gemini_provider import GeminiProvider
from fambot_backend.providers.model_provider import ProviderContext
from fambot_backend.schemas import UserProfileOut
from fambot_backend.services import gemini_document_analysis as gda


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis.get_user_profile")
@patch("fambot_backend.services.gemini_document_analysis._get_client")
@patch("fambot_backend.services.gemini_document_analysis._upload_bytes", return_value="fake-file-ref")
def test_analyze_uploaded_document_returns_model_and_analysis(
    _upload: MagicMock,
    get_client: MagicMock,
    get_profile: MagicMock,
) -> None:
    get_profile.return_value = UserProfileOut(uid="u1", onboarding_complete=False)

    gen = MagicMock()

    def fake_generate(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(text="Eat well. Follow up with your clinician.")

    gen.models.generate_content.side_effect = fake_generate
    get_client.return_value = gen

    out = gda.analyze_uploaded_document(
        uid="u1",
        file_name="r.pdf",
        content_type="application/pdf",
        payload=b"%PDF",
    )
    assert out["model"]
    assert "Eat well" in out["analysis"]
    gen.models.generate_content.assert_called_once()


class _FakeModels:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self.calls = 0

    def generate_content(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


class _FakeGenaiClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.models = _FakeModels(responses)


class _FakeStreamModels:
    def generate_content_stream(self, *_args: object, **kwargs: object) -> list[SimpleNamespace]:
        config = kwargs["config"]
        tools = list(getattr(config, "tools", []) or [])
        has_file_search = any(getattr(tool, "file_search", None) is not None for tool in tools)
        has_functions = any(bool(getattr(tool, "function_declarations", None)) for tool in tools)
        assert has_functions
        assert not (has_file_search and has_functions)
        return [SimpleNamespace(text="ok", candidates=[])]


class _FakeStreamClient:
    def __init__(self) -> None:
        self.models = _FakeStreamModels()


@pytest.mark.unit
def test_chat_tools_do_not_mix_file_search_with_function_declarations() -> None:
    tools = gda._build_tools_list("u1")

    assert any(bool(getattr(tool, "function_declarations", None)) for tool in tools)
    assert all(getattr(tool, "file_search", None) is None for tool in tools)


@pytest.mark.unit
@patch("fambot_backend.providers.gemini_provider._user_message_text", return_value="USER_MESSAGE:\nhello")
@patch("fambot_backend.providers.gemini_provider._model_name", return_value="gemini-test")
@patch("fambot_backend.providers.gemini_provider._get_client", return_value=_FakeStreamClient())
def test_gemini_provider_sends_function_tools_without_builtin_file_search(
    _client: MagicMock,
    _model: MagicMock,
    _message_text: MagicMock,
) -> None:
    events = list(
        GeminiProvider().stream_turn(
            context=ProviderContext(
                uid="u1",
                user_message="hello",
                history=[],
                upload_name=None,
                upload_content_type=None,
                upload_payload=None,
            ),
            tool_dispatch=lambda *_args: ({}, None),
        )
    )

    assert [event.kind for event in events] == ["token", "done"]


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis.get_user_profile")
@patch("fambot_backend.services.gemini_document_analysis.list_user_documents", return_value=[])
@patch("fambot_backend.services.gemini_document_analysis._get_client")
@patch("fambot_backend.services.gemini_document_analysis._upload_bytes", return_value="fake")
def test_generate_chat_turn_returns_content_and_calls_model(
    _upload: MagicMock,
    get_client: MagicMock,
    _list_docs: MagicMock,
    get_profile: MagicMock,
) -> None:
    get_profile.return_value = UserProfileOut(uid="u1", age=40, onboarding_complete=True)
    # First call: main reply; optional second call: title when no prior user in history
    get_client.return_value = _FakeGenaiClient([SimpleNamespace(text="  Here is help.  ")])

    out = gda.generate_chat_turn(
        uid="u1",
        user_message="What about BP?",
        history=[{"role": "user", "content": "Hi", "created_at": None}],
    )
    assert out["content"] == "Here is help."
    assert out.get("citations") is None


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis.get_user_profile")
@patch("fambot_backend.services.gemini_document_analysis.list_user_documents", return_value=[])
@patch("fambot_backend.services.gemini_document_analysis._get_client")
@patch("fambot_backend.services.gemini_document_analysis._upload_bytes", return_value="fake")
def test_generate_chat_turn_empty_response_uses_fallback(
    _upload: MagicMock,
    get_client: MagicMock,
    _list_docs: MagicMock,
    get_profile: MagicMock,
) -> None:
    get_profile.return_value = UserProfileOut(uid="u1", onboarding_complete=False)
    get_client.return_value = _FakeGenaiClient(
        [SimpleNamespace(text="   "), SimpleNamespace(text="t")]
    )

    out = gda.generate_chat_turn(uid="u1", user_message="x", history=[])
    assert "try again" in out["content"].lower()


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis._get_client")
def test_maybe_new_chat_title_uses_dedicated_title_model(get_client: MagicMock) -> None:
    gen = MagicMock()
    gen.models.generate_content.return_value = SimpleNamespace(text="Heart Health")
    get_client.return_value = gen

    with patch.dict("os.environ", {"GEMINI_CHAT_TITLE_MODEL": "gemini-2.5-flash-lite"}, clear=False):
        out = gda.maybe_new_chat_title(user_message="How do I reduce risk?", history=[])
    assert out == "Heart Health"
    assert gen.models.generate_content.call_args.kwargs["model"] == "gemini-2.5-flash-lite"


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis._get_client")
def test_maybe_new_chat_title_skips_after_first_user_message(get_client: MagicMock) -> None:
    out = gda.maybe_new_chat_title(
        user_message="follow-up",
        history=[{"role": "user", "content": "first message"}],
    )
    assert out is None
    get_client.assert_not_called()


@pytest.mark.unit
@patch("fambot_backend.services.gemini_document_analysis.get_user_profile")
@patch("fambot_backend.services.gemini_document_analysis._upload_bytes")
@patch("fambot_backend.services.gemini_document_analysis._get_client")
def test_analyze_uploaded_document_empty_gemini_raises(
    get_client: MagicMock,
    upload_bytes: MagicMock,
    get_profile: MagicMock,
) -> None:
    get_profile.return_value = UserProfileOut(uid="u1", onboarding_complete=False)
    upload_bytes.return_value = "ref"
    gen = MagicMock()
    gen.models.generate_content.return_value = SimpleNamespace(text="")
    get_client.return_value = gen

    with pytest.raises(HTTPException) as ei:
        gda.analyze_uploaded_document(
            uid="u1",
            file_name="x.pdf",
            content_type="application/pdf",
            payload=b"x",
        )
    assert ei.value.status_code == 502
