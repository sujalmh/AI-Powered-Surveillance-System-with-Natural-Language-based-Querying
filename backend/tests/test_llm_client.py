"""Tests for the provider-agnostic LLM client and the parser's JSON coercion.

These run offline: no LLM call is made, they only exercise the extraction,
prompt rendering, transport selection and normalisation logic.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import types
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services import llm_client, nl_parser
from backend.app.services.nl_parser import NLFilter, coerce_to_nlfilter


# ---------------------------------------------------------------------------
# Helpers (fake CLI output / fake HTTP answers - no network, no subprocess)
# ---------------------------------------------------------------------------

class _FakeProc:
    """Minimal stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _FakeResponse:
    """Minimal stand-in for the object returned by ``urlopen``."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _cli_text_event(text: str) -> str:
    return json.dumps({"type": "text", "part": {"type": "text", "text": text}}) + "\n"


def _cli_error_event(message: str) -> str:
    return json.dumps({"type": "error", "error": {"message": message}}) + "\n"


@pytest.fixture(autouse=True)
def _clean_transport_cache():
    """The "route not available" memory is global - keep the tests isolated."""
    llm_client.reset_transport_cache()
    yield
    llm_client.reset_transport_cache()


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def test_extract_plain_json():
    obj = llm_client.extract_json_object('{"a": 1, "b": "x"}')
    assert obj == {"a": 1, "b": "x"}


def test_extract_from_markdown_fence():
    reply = 'Here you go:\n```json\n{"camera_id": 0, "color": "Olive"}\n```\nHope that helps.'
    assert llm_client.extract_json_object(reply) == {"camera_id": 0, "color": "Olive"}


def test_extract_with_trailing_prose():
    reply = '{"query_type": "visual"}\n\nNote: `{reference_date}` was filled in above.'
    assert llm_client.extract_json_object(reply) == {"query_type": "visual"}


def test_extract_repairs_stray_escapes_and_unquoted_keys():
    reply = r'{ ok\: true, n\: 42 }'
    obj = llm_client.extract_json_object(reply)
    assert obj == {"ok": True, "n": 42}


def test_extract_raises_on_garbage():
    with pytest.raises(llm_client.LLMError):
        llm_client.extract_json_object("I cannot answer that.")


def test_extract_prefers_the_last_object():
    reply = (
        'Here is the schema you gave me: {"properties": {"query_type": {}}, "title": "NLFilter"}\n'
        'And here is the answer:\n{"query_type": "visual", "semantic_query": "person in red"}'
    )
    obj = llm_client.extract_json_object(reply)
    assert obj["semantic_query"] == "person in red"
    assert obj["query_type"] == "visual"


def test_extract_prefers_an_object_with_filter_fields():
    reply = '{"query_type": "visual", "semantic_query": "x"} {"note": "that was the example"}'
    assert llm_client.extract_json_object(reply)["query_type"] == "visual"


def test_extract_returns_the_only_object_when_it_is_an_example():
    reply = '{"properties": {"query_type": {"type": "string"}}, "title": "NLFilter"}'
    obj = llm_client.extract_json_object(reply)
    assert obj["title"] == "NLFilter"


def test_extract_still_tolerates_fences_and_prose():
    reply = (
        "Sure!\n```json\n{\"query_type\": \"visual\", \"semantic_query\": \"a person\"}\n```\n"
        "Hope that helps."
    )
    assert llm_client.extract_json_object(reply)["semantic_query"] == "a person"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def test_render_prompt_fills_placeholders_and_keeps_unknowns():
    out = llm_client.render_prompt(
        "Q={query} S={schema} F={format_instructions} R={reference_date} X={unknown}",
        query="hi", schema="{}", format_instructions="json", reference_date="2026-03-01",
    )
    assert "Q=hi" in out and "R=2026-03-01" in out
    assert "X={unknown}" in out


def test_render_prompt_rejects_missing_required_values():
    with pytest.raises(llm_client.LLMError):
        llm_client.render_prompt("R={reference_date}", required=("reference_date",))


def test_render_prompt_does_not_expand_placeholders_inside_values():
    out = llm_client.render_prompt("Q={query}", query="a {schema} b")
    assert out == "Q=a {schema} b"


def test_parser_prompt_has_no_unfilled_placeholder():
    from backend.app.services import nl_parser

    prompt = nl_parser.build_parser_prompt(
        "show me people", {"type": "object"}, "answer with json", "2026-03-01",
    )
    # every placeholder of the template was substituted
    assert nl_parser.unfilled_placeholders(
        nl_parser.PROMPT_TEMPLATE,
        {"query": "q", "schema": "{}", "format_instructions": "f", "reference_date": "d"},
    ) == []
    assert not re.search(r"\{(query|schema|format_instructions|reference_date)\}", prompt)
    assert "2026-03-01" in prompt
    assert "show me people" in prompt


def test_parser_prompt_tolerates_placeholders_in_the_user_query():
    """A user question about a placeholder must not be reported as unfilled."""
    from backend.app.services import nl_parser

    prompt = nl_parser.build_parser_prompt(
        "what about {reference_date} and {query}?", {"type": "object"}, "json", "2026-03-01",
    )
    assert "{reference_date}" in prompt      # the user's own text is kept
    assert "{query}" in prompt
    assert "2026-03-01" in prompt


# ---------------------------------------------------------------------------
# Schema coercion (small models answer with slightly different shapes)
# ---------------------------------------------------------------------------

def test_coerce_objects_from_plain_strings():
    out = coerce_to_nlfilter({"objects": ["person"], "camera_id": 0})
    assert isinstance(out, NLFilter)
    assert out.objects[0].name == "person"
    assert out.camera_id == 0


def test_coerce_time_aliases():
    out = coerce_to_nlfilter({"time": {"start": "2026-03-01T17:01:00", "end": "2026-03-01T17:03:00"}})
    assert out.time is not None
    assert out.time.from_iso == "2026-03-01T17:01:00"
    assert out.time.to_iso == "2026-03-01T17:03:00"


def test_coerce_count_constraint_from_bare_int():
    out = coerce_to_nlfilter({"count_constraint": 3})
    assert out.count_constraint == {"eq": 3}


def test_coerce_count_constraint_from_dict():
    out = coerce_to_nlfilter({"count_constraint": {"gte": 5}})
    assert out.count_constraint == {"gte": 5}


def test_coerce_count_constraint_from_phrase_and_aliases():
    assert coerce_to_nlfilter({"count_constraint": "more than 3"}).count_constraint == {"gt": 3}
    assert coerce_to_nlfilter({"count_constraint": "at least 4"}).count_constraint == {"gte": 4}
    assert coerce_to_nlfilter({"count_constraint": {"minimum": 2}}).count_constraint == {"gte": 2}


def test_coerce_time_drops_values_that_are_not_timestamps():
    out = coerce_to_nlfilter({"time": {"start": "yesterday", "end": "tomorrow"}})
    assert out.time is None


def test_coerce_time_accepts_a_relative_window():
    out = coerce_to_nlfilter({"time": "last 30 minutes"})
    assert out.time is not None
    assert out.time.last_minutes == 30


def test_coerce_keeps_valid_payload():
    out = coerce_to_nlfilter({
        "query_type": "visual",
        "color": "Olive",
        "objects": [{"name": "person"}],
        "time": {"from_iso": "2026-03-01T17:01:00", "to_iso": "2026-03-01T17:03:00"},
    })
    assert out.query_type == "visual"
    assert out.color == "Olive"
    assert out.time.from_iso.startswith("2026-03-01T17:01")


def test_coerce_fills_the_required_fields():
    out = coerce_to_nlfilter({"objects": ["person"]}, raw_query="show me people")
    assert out.semantic_query == "show me people"
    assert out.raw_query == "show me people"


def test_coerce_validates_the_small_model_shapes():
    out = coerce_to_nlfilter(
        {
            "objects": ["person"],
            "time": {"start": "2026-03-01T17:01:00", "end": "2026-03-01T17:03:00"},
            "count_constraint": 5,
        },
        raw_query="how many people",
    )
    assert isinstance(out, NLFilter)
    assert out.objects[0].name == "person"
    assert out.time.from_iso == "2026-03-01T17:01:00"
    assert out.count_constraint == {"eq": 5}
    assert out.semantic_query == "how many people"


def test_coerce_normalises_query_type_and_keeps_required_fields_on_error():
    out = coerce_to_nlfilter({"query_type": "INFORMATIONAL"}, raw_query="how many cameras")
    assert out.query_type == "informational"
    assert out.raw_query == "how many cameras"


def test_bare_count_constraint_is_upgraded_by_the_question():
    out = coerce_to_nlfilter({"count_constraint": 3}, raw_query="show me more than 3 people")
    assert out.count_constraint == {"gt": 3}
    out = coerce_to_nlfilter({"count_constraint": 3}, raw_query="show me at least 3 people")
    assert out.count_constraint == {"gte": 3}
    out = coerce_to_nlfilter({"count_constraint": 3}, raw_query="show me exactly 3 people")
    assert out.count_constraint == {"eq": 3}
    out = coerce_to_nlfilter({"count_constraint": 3}, raw_query="how many people are there")
    assert out.count_constraint == {"eq": 3}


def test_filter_level_count_constraint_is_upgraded_by_the_question():
    from backend.app.services import nl_parser

    f = {"count_constraint": {"eq": 3}}
    nl_parser._augment_count_constraint_from_query("show me more than 3 people", f)
    assert f["count_constraint"] == {"gt": 3}

    f = {"count_constraint": {"eq": 3}}
    nl_parser._augment_count_constraint_from_query("show me at least 3 people", f)
    assert f["count_constraint"] == {"gte": 3}

    f = {"count_constraint": {"eq": 3}}
    nl_parser._augment_count_constraint_from_query("show me exactly 3 people", f)
    assert f["count_constraint"] == {"eq": 3}


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_opencode_backend_ignores_stale_openai_model(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "opencode")
    monkeypatch.setattr(settings, "OPENCODE_MODEL", "opencode/deepseek-v4.1-flash-free")
    monkeypatch.setattr(settings, "OPENCODE_BASE_URL", "http://127.0.0.1:4096")
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "gpt-4o-mini", "api_key": "",
    })
    assert isinstance(backend, llm_client.OpencodeBackend)
    assert backend.model == "opencode/deepseek-v4.1-flash-free"


def test_opencode_backend_ignores_a_stale_model_everywhere(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENCODE_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(settings, "OPENCODE_MODEL_FALLBACKS", "gpt-4o,opencode-go/minimax-m3.1-flash-free")
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "gpt-4o-mini", "api_key": "",
    })
    assert backend.model == llm_client.DEFAULT_OPENCODE_MODEL
    assert backend.fallbacks == ["opencode-go/minimax-m3.1-flash-free"]


def test_opencode_backend_does_not_need_the_cli_at_build_time(monkeypatch):
    from backend.app.config import settings

    def _no_cli():
        raise llm_client.LLMError("opencode CLI not found")

    monkeypatch.setattr(llm_client.OpencodeBackend, "_find_cli", staticmethod(_no_cli))
    monkeypatch.setattr(settings, "OPENCODE_MODEL", "opencode-go/beam-free")
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "opencode-go/beam-free", "api_key": "",
    })
    assert isinstance(backend, llm_client.OpencodeBackend)


def test_openai_compat_backend_uses_configured_base_url(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "http://localhost:8000/v1")
    backend = llm_client.build_backend({
        "provider": "openai_compatible", "model": "my-model", "api_key": "k",
    })
    assert isinstance(backend, llm_client.OpenAICompatBackend)
    assert backend.base_url == "http://localhost:8000/v1"
    assert backend.model == "my-model"


def test_chat_json_rejects_unsupported_provider():
    with pytest.raises(llm_client.LLMError):
        llm_client.build_backend({"provider": "nonsense-provider", "model": "m", "api_key": ""})


@pytest.mark.parametrize("provider", sorted(nl_parser._LLM_CLIENT_PROVIDERS))
def test_build_backend_accepts_every_llm_client_provider(provider, monkeypatch):
    """Every provider the parser hands to llm_client must be buildable."""
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(settings, "OPENCODE_MODEL", "opencode-go/beam-free")
    monkeypatch.setattr(settings, "OPENCODE_MODEL_FALLBACKS", "")
    backend = llm_client.build_backend({
        "provider": provider, "model": "some-model", "api_key": "",
    })
    assert backend is not None
    if provider == "opencode":
        assert isinstance(backend, llm_client.OpencodeBackend)
    else:
        assert isinstance(backend, llm_client.OpenAICompatBackend)


def test_openai_compat_alias_is_accepted(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "http://localhost:8000/v1")
    backend = llm_client.build_backend({
        "provider": "openai_compat", "model": "my-model", "api_key": "k",
    })
    assert isinstance(backend, llm_client.OpenAICompatBackend)
    assert backend.model == "my-model"


def test_every_provider_has_its_own_default_model(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "NL_DEFAULT_MODEL", "opencode-go/beam-free")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test-key")

    openai_backend = llm_client.build_backend({
        "provider": "openai", "model": "", "api_key": "sk-test-key",
    })
    assert openai_backend.model == "gpt-4o-mini"

    openrouter_backend = llm_client.build_backend({
        "provider": "openrouter", "model": "", "api_key": "sk-or-test-key",
    })
    assert openrouter_backend.model == "openai/gpt-4o-mini"

    ollama_backend = llm_client.build_backend({
        "provider": "ollama", "model": "", "api_key": "",
    })
    assert ollama_backend.model == "llama3.1"

    opencode_backend = llm_client.build_backend({
        "provider": "opencode", "model": "opencode-go/beam-free", "api_key": "",
    })
    assert opencode_backend.model == "opencode-go/beam-free"


def test_set_opencode_model_reaches_the_backend(monkeypatch):
    """The evaluation scripts' ``--model`` override must reach the backend.

    Regression test: ``build_backend`` reads the *active* configuration,
    which prefers the opencode-specific model id, so setting only the
    generic ``NL_DEFAULT_MODEL`` left the backend on the configured primary
    while the result file claimed a different model.
    """
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENCODE_MODEL", "opencode-go/beam-free")
    monkeypatch.setattr(settings, "NL_DEFAULT_MODEL", "opencode-go/beam-free")
    monkeypatch.setattr(
        settings, "OPENCODE_MODEL_FALLBACKS",
        "opencode-go/minimax-m3.1-flash-free,opencode-go/deepseek-v4.1-flash",
    )

    settings.set_opencode_model("opencode-go/minimax-m3.1-flash-free")
    backend = llm_client.build_backend({
        "provider": "opencode", "model": settings.NL_DEFAULT_MODEL, "api_key": "",
    })
    assert backend.model == "opencode-go/minimax-m3.1-flash-free"
    # The model that is now primary must not also be a fallback.
    assert "opencode-go/minimax-m3.1-flash-free" not in backend.fallbacks
    assert "opencode-go/deepseek-v4.1-flash" in backend.fallbacks


def test_the_model_override_reaches_the_opencode_backend(monkeypatch):
    """``--model`` must change the model the parser actually uses.

    The evaluation scripts set the override and then record the model id in
    the result file.  When the override only reached the generic
    ``NL_DEFAULT_MODEL`` the backend stayed on the configured primary, so the
    run parsed with one model and reported another.
    """
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENCODE_MODEL", "opencode-go/beam-free")
    monkeypatch.setattr(settings, "NL_DEFAULT_MODEL", "opencode-go/beam-free")
    monkeypatch.setattr(settings, "_llm_config_checked", False)
    monkeypatch.setattr(settings, "_llm_config_cache", None)

    settings.set_opencode_model("opencode-go/minimax-m3.1-flash-free")
    try:
        backend = llm_client.build_backend()
        assert backend.model == "opencode-go/minimax-m3.1-flash-free"
        # The chosen model must not also be a fallback.
        assert "opencode-go/minimax-m3.1-flash-free" not in backend.fallbacks
    finally:
        settings.OPENCODE_MODEL = "opencode-go/beam-free"
        settings.NL_DEFAULT_MODEL = "opencode-go/beam-free"
        settings._llm_config_checked = False
        settings._llm_config_cache = None


# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

def test_root_is_the_repository_root():
    assert llm_client.ROOT == Path(__file__).resolve().parents[2]
    assert (llm_client.ROOT / "backend").is_dir()


def test_default_cache_path_points_into_the_repository():
    path = llm_client.default_cache_path("opencode-go/beam-free")
    assert path == (
        llm_client.ROOT / "backend" / "benchmarks" / "parsed_cache" / "opencode-go_beam-free.json"
    )
    assert path.parent.is_relative_to(llm_client.ROOT)


def test_opencode_backend_runs_in_the_repository_root():
    backend = llm_client.OpencodeBackend(model="opencode-go/beam-free", cli="opencode.cmd")
    assert backend.cwd == llm_client.ROOT
    assert (backend.cwd / "backend").is_dir()


# ---------------------------------------------------------------------------
# opencode transport order
# ---------------------------------------------------------------------------

def test_transport_order_is_http_session_cli(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENCODE_TRANSPORT", "auto")
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "opencode-go/beam-free", "api_key": "",
    })
    assert backend.transports == ["http", "session", "cli"]
    assert backend.transports[-1] == "cli"


def test_transport_order_can_be_restricted(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENCODE_TRANSPORT", "cli")
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "opencode-go/beam-free", "api_key": "",
    })
    assert backend.transports == ["cli"]


def test_http_transport_is_used_when_the_server_answers(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        body = json.dumps({"choices": [{"message": {"content": '{"query_type": "visual"}'}}]})
        return _FakeResponse(body.encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096",
        transports=["http", "session", "cli"],
    )
    reply = backend.chat("hello")
    assert reply.transport == "opencode-http"
    assert reply.text == '{"query_type": "visual"}'
    assert captured["url"].endswith("/v1/chat/completions")


def test_session_transport_is_used_when_the_server_answers(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        method = request.get_method()
        calls.append((method, request.full_url))
        if method == "DELETE":
            return _FakeResponse(b"")
        if request.full_url.endswith("/session"):
            return _FakeResponse(json.dumps({"id": "ses_1"}).encode())
        return _FakeResponse(json.dumps({
            "parts": [{"type": "text", "text": '{"query_type": "visual"}'}]
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096",
        transports=["http", "session", "cli"],
    )
    reply = backend.chat("hello")
    assert reply.transport == "opencode-session"
    assert reply.text == '{"query_type": "visual"}'
    assert any("/session/ses_1/message" in url for _method, url in calls)


def test_broken_http_and_session_transports_fall_back_to_the_cli(monkeypatch):
    requested = []

    def fake_urlopen(request, timeout=None):
        requested.append(request.full_url)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        llm_client.subprocess, "run",
        lambda cmd, **kwargs: _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}')),
    )
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096",
        cli="opencode.cmd", transports=["http", "session", "cli"],
    )
    reply = backend.chat("hello")
    assert reply.transport == "opencode-cli"
    assert reply.text == '{"query_type": "visual"}'
    assert any(url.endswith("/v1/chat/completions") for url in requested)
    assert any(url.endswith("/session") for url in requested)


def test_html_answer_is_treated_as_a_missing_route(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b"<!doctype html><html>v2 app</html>")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        llm_client.subprocess, "run",
        lambda cmd, **kwargs: _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}')),
    )
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096",
        cli="opencode.cmd", transports=["http", "session", "cli"],
    )
    assert backend.chat("hello").transport == "opencode-cli"


def test_unavailable_transport_is_not_probed_again(monkeypatch):
    requested = []

    def fake_urlopen(request, timeout=None):
        requested.append(request.full_url)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        llm_client.subprocess, "run",
        lambda cmd, **kwargs: _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}')),
    )
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096",
        cli="opencode.cmd", transports=["http", "session", "cli"],
    )
    backend.chat("hello")
    first = len(requested)
    backend.chat("hello")
    assert len(requested) == first


# ---------------------------------------------------------------------------
# Deadlines (one parse must never block for models x transports x timeout)
# ---------------------------------------------------------------------------

def test_chat_stops_when_the_deadline_is_exceeded(monkeypatch):
    clock = {"now": 1000.0}

    def fake_monotonic():
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(llm_client.time, "monotonic", fake_monotonic)
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(("http", timeout))
        raise TimeoutError("timed out")

    def fake_run(cmd, **kwargs):
        attempts.append(("cli", kwargs.get("timeout")))
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)

    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        fallbacks=["opencode-go/minimax-m3.1-flash-free"], timeout=2,
        transports=["http", "session", "cli"],
    )
    with pytest.raises(llm_client.LLMError) as excinfo:
        backend.chat("hello")
    assert "deadline" in str(excinfo.value)
    # the loop stopped instead of trying every model x transport
    assert len(attempts) < 6


def test_chat_total_time_is_bounded(monkeypatch):
    budgets = []

    def fake_urlopen(request, timeout=None):
        budgets.append(timeout)
        time.sleep(0.15)
        raise TimeoutError("timed out")

    def fake_run(cmd, **kwargs):
        budgets.append(kwargs.get("timeout"))
        time.sleep(0.15)
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)

    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        fallbacks=["opencode-go/minimax-m3.1-flash-free", "opencode-go/other-free"],
        timeout=0.2, transports=["http", "session", "cli"],
    )
    started = time.monotonic()
    with pytest.raises(llm_client.LLMError):
        backend.chat("hello")
    elapsed = time.monotonic() - started
    # 3 models x 3 transports x 0.2 s would be 1.8 s; the budget is 0.2 s.
    assert elapsed < 1.0
    # no attempt (CLI included) got more than the remaining budget
    assert budgets
    assert all(budget is not None and budget <= 0.2 for budget in budgets)


def test_a_socket_timeout_marks_the_transport_unavailable(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        llm_client.subprocess, "run",
        lambda cmd, **kwargs: _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}')),
    )
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        fallbacks=["opencode-go/minimax-m3.1-flash-free"], transports=["http", "session", "cli"],
    )
    reply = backend.chat("hello")
    assert reply.transport == "opencode-cli"
    # http (2 candidate routes) + session for the first model only: the
    # timeout is a transport problem, not a model problem.
    assert len(calls) == 3


def test_cli_uses_the_server_only_when_it_was_configured(monkeypatch):
    cmds = []

    def fake_run(cmd, **kwargs):
        cmds.append(cmd)
        return _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}'))

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)

    without_server = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        transports=["cli"],
    )
    without_server.chat("hello")
    assert "--server" not in cmds[0]

    with_server = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        transports=["cli"], cli_use_server=True,
    )
    with_server.chat("hello")
    assert "--server" in cmds[1]
    assert "http://127.0.0.1:4096" in cmds[1]


def test_build_backend_only_points_the_cli_at_an_explicit_server(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "OPENCODE_BASE_URL", "http://127.0.0.1:4096")
    monkeypatch.setattr(settings, "OPENCODE_BASE_URL_EXPLICIT", False)
    monkeypatch.setattr(settings, "OPENCODE_CLI_USE_SERVER", False)
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "opencode-go/beam-free", "api_key": "",
    })
    assert backend.cli_use_server is False

    monkeypatch.setattr(settings, "OPENCODE_BASE_URL_EXPLICIT", True)
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "opencode-go/beam-free", "api_key": "",
    })
    assert backend.cli_use_server is True

    monkeypatch.setattr(settings, "OPENCODE_BASE_URL_EXPLICIT", False)
    monkeypatch.setattr(settings, "OPENCODE_CLI_USE_SERVER", True)
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "opencode-go/beam-free", "api_key": "",
    })
    assert backend.cli_use_server is True


# ---------------------------------------------------------------------------
# opencode model fallback (the primary must always be tried first)
# ---------------------------------------------------------------------------

def test_fallback_does_not_replace_the_primary_model(monkeypatch):
    used_models = []

    def fake_run(cmd, **kwargs):
        model = cmd[cmd.index("-m") + 1]
        used_models.append(model)
        if model == "opencode-go/beam-free":
            return _FakeProc(stdout=_cli_error_event("Free usage limit reached"))
        return _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}'))

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        fallbacks=["opencode-go/minimax-m3.1-flash-free"], transports=["cli"],
    )
    reply = backend.chat("hello")
    assert reply.model == "opencode-go/minimax-m3.1-flash-free"
    assert backend.model == "opencode-go/beam-free"
    assert backend.primary_model == "opencode-go/beam-free"

    backend.chat("hello")
    assert used_models == [
        "opencode-go/beam-free", "opencode-go/minimax-m3.1-flash-free",
        "opencode-go/beam-free", "opencode-go/minimax-m3.1-flash-free",
    ]


def test_primary_model_is_used_again_after_a_successful_fallback(monkeypatch):
    used_models = []

    def fake_run(cmd, **kwargs):
        model = cmd[cmd.index("-m") + 1]
        used_models.append(model)
        return _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}'))

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        fallbacks=["opencode-go/minimax-m3.1-flash-free"], transports=["cli"],
    )
    assert backend.chat("hello").model == "opencode-go/beam-free"
    assert used_models == ["opencode-go/beam-free"]


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

def test_cli_call_uses_the_configured_timeout(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}'))

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        timeout=42, transports=["cli"],
    )
    backend.chat("hello")
    # the CLI may use the whole budget, but never more than it
    assert 41 <= captured["timeout"] <= 42


def test_build_backend_applies_the_configured_timeout(monkeypatch):
    from backend.app.config import settings

    monkeypatch.setattr(settings, "LLM_TIMEOUT", 77)
    monkeypatch.setattr(settings, "NL_TIMEOUT_SECONDS", 300)
    backend = llm_client.build_backend({
        "provider": "opencode", "model": "opencode-go/beam-free", "api_key": "",
    })
    assert backend.timeout == 77


def test_http_calls_use_the_configured_timeout(monkeypatch):
    from backend.app.config import settings

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps(
            {"choices": [{"message": {"content": '{"query_type": "visual"}'}}]}
        ).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(settings, "LLM_TIMEOUT", 33)
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "http://localhost:8000/v1")
    backend = llm_client.build_backend({
        "provider": "openai_compatible", "model": "m", "api_key": "",
    })
    backend.chat("hello")
    assert captured["timeout"] == 33


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def test_api_key_never_appears_in_an_error_message(monkeypatch):
    secret = "sk-test-SUPERSECRETKEY123456"

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = llm_client.OpenAICompatBackend("http://example.invalid/v1", secret, "m")
    with pytest.raises(llm_client.LLMError) as excinfo:
        backend.chat("hello")
    assert secret not in str(excinfo.value)


def test_api_key_never_appears_in_the_reply_payload(monkeypatch):
    secret = "sk-test-SUPERSECRETKEY123456"

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(json.dumps({
            "choices": [{"message": {"content": '{"query_type": "visual"}'}}],
            "echo": secret,
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = llm_client.OpenAICompatBackend("http://example.invalid/v1", secret, "m")
    reply = backend.chat("hello")
    assert secret not in reply.raw


def test_cli_prompt_file_is_removed_and_holds_no_secret(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        path = Path(cmd[cmd.index("--file") + 1])
        seen["path"] = path
        seen["content"] = path.read_text(encoding="utf-8")
        return _FakeProc(stdout=_cli_text_event('{"query_type": "visual"}'))

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        server_password="top-secret-password", transports=["cli"],
    )
    backend.chat("the parser prompt")
    assert not seen["path"].exists()
    assert "the parser prompt" in seen["content"]
    assert "top-secret-password" not in seen["content"]


def test_cli_prompt_file_is_removed_when_the_call_fails(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["path"] = Path(cmd[cmd.index("--file") + 1])
        raise OSError("boom")

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        transports=["cli"],
    )
    with pytest.raises(llm_client.LLMError):
        backend.chat("hello")
    assert not seen["path"].exists()


def test_cli_error_message_is_redacted(monkeypatch):
    secret = "sk-test-SUPERSECRETKEY123456"

    def fake_run(cmd, **kwargs):
        return _FakeProc(stdout=_cli_error_event(f"provider rejected {secret}"), stderr=secret)

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)
    backend = llm_client.OpencodeBackend(
        model="opencode-go/beam-free", server="http://127.0.0.1:4096", cli="opencode.cmd",
        transports=["cli"],
    )
    with pytest.raises(llm_client.LLMError) as excinfo:
        backend.chat("hello")
    assert secret not in str(excinfo.value)


def test_langchain_errors_are_redacted(monkeypatch):
    """The LangChain error text reaches logger.warning and the parse cache."""
    secret = "sk-test-SUPERSECRETKEY123456"
    monkeypatch.setattr(
        nl_parser.settings, "get_active_llm_config",
        lambda: {"provider": "openai", "model": "gpt-4o-mini", "api_key": secret},
    )

    fake_module = types.ModuleType("langchain_openai")

    def _boom(*args, **kwargs):
        raise RuntimeError(f"401 Unauthorized for key {secret}")

    fake_module.ChatOpenAI = _boom
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    with pytest.raises(RuntimeError) as excinfo:
        nl_parser._get_lc_components()
    assert secret not in str(excinfo.value)
    assert "***" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Parser integration (no network, no CLI)
# ---------------------------------------------------------------------------

def test_parse_nl_with_llm_uses_the_llm_client_path(monkeypatch):
    from backend.app.services import nl_parser

    captured = {}

    class _FakeBackend:
        def chat(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return llm_client.LLMReply(
                text=json.dumps({
                    "query_type": "visual",
                    "objects": ["person"],
                    "color": "red",
                    "time": {"start": "2026-03-01T17:01:00", "end": "2026-03-01T17:03:00"},
                    "count_constraint": 5,
                }),
                model="opencode-go/beam-free",
                transport="fake",
            )

    monkeypatch.setattr(
        nl_parser.settings, "get_active_llm_config",
        lambda: {"provider": "opencode", "model": "opencode-go/beam-free", "api_key": ""},
    )
    monkeypatch.setattr(llm_client, "build_backend", lambda: _FakeBackend())

    result = nl_parser.parse_nl_with_llm("how many people were there yesterday?")
    assert result["objects.object_name"] == "person"
    assert result["objects.color"] == "Red"
    assert result["count_constraint"] == {"eq": 5}
    assert result["timestamp"] == {
        "$gte": "2026-03-01T17:01:00", "$lte": "2026-03-01T17:03:00",
    }
    assert result["query_type"] == "visual"
    assert result["__semantic_query"]
    # The prompt that reached the backend is fully rendered.
    assert nl_parser.unfilled_placeholders(nl_parser.PROMPT_TEMPLATE, {
        "query": "q", "schema": "{}", "format_instructions": "f", "reference_date": "d",
    }) == []
    assert "{reference_date}" not in captured["prompt"]
    assert "2026-03-01" in captured["prompt"]


def test_parse_nl_with_llm_does_not_print_to_stdout(capsys, monkeypatch):
    class _FakeBackend:
        def chat(self, prompt, **kwargs):
            return llm_client.LLMReply(
                text=json.dumps({"query_type": "visual", "objects": ["person"]}),
                model="opencode-go/beam-free", transport="fake",
            )

    monkeypatch.setattr(
        nl_parser.settings, "get_active_llm_config",
        lambda: {"provider": "opencode", "model": "opencode-go/beam-free", "api_key": ""},
    )
    monkeypatch.setattr(llm_client, "build_backend", lambda: _FakeBackend())
    nl_parser.parse_nl_with_llm("show me people")
    captured = capsys.readouterr()
    assert "[NL Parser]" not in captured.out
    assert "[NL Parser]" not in captured.err


def test_get_lc_components_bypasses_langchain(monkeypatch):
    from backend.app.services import nl_parser

    monkeypatch.setattr(
        nl_parser.settings, "get_active_llm_config",
        lambda: {"provider": "opencode", "model": "opencode-go/beam-free", "api_key": ""},
    )
    # Make every LangChain import fail: the opencode path must not need it.
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", None)
    monkeypatch.setitem(sys.modules, "langchain_core.output_parsers", None)
    assert nl_parser._get_lc_components() == (None, None, None)


# ---------------------------------------------------------------------------
# Parse cache
# ---------------------------------------------------------------------------

def test_parse_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    cache = llm_client.ParseCache(path)
    cache.put("q1", {"parsed": {"camera_id": 0}})
    cache.save()
    again = llm_client.ParseCache(path)
    assert again.get("q1") == {"parsed": {"camera_id": 0}}
    assert again.get("missing") is None


def test_parse_cache_reports_a_corrupt_file(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json at all", encoding="utf-8")
    cache = llm_client.ParseCache(path)          # must not raise
    assert cache.get("q1") is None
    assert cache.load_error


def test_parse_cache_save_is_atomic(tmp_path):
    path = tmp_path / "cache.json"
    cache = llm_client.ParseCache(path)
    cache.put("q1", {"parsed": {"camera_id": 0}})
    cache.save()
    assert json.loads(path.read_text(encoding="utf-8"))["q1"] == {"parsed": {"camera_id": 0}}
    assert not (tmp_path / "cache.json.tmp").exists()
