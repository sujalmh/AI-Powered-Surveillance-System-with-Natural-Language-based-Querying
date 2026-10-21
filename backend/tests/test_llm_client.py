"""Tests for the provider-agnostic LLM client and the parser's JSON coercion.

These run offline: no LLM call is made, they only exercise the extraction,
prompt rendering and normalisation logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services import llm_client
from backend.app.services.nl_parser import NLFilter, coerce_to_nlfilter


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
