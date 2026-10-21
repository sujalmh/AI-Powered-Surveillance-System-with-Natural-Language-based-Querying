"""Person-index routing and parser-grounding tests (no MongoDB required)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from backend.app.config import settings
from backend.app.services.hybrid_search import HybridSearchEngine
from backend.app.services import nl_parser
from backend.app.services import unified_retrieval as ur


ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeEncoder:
    def __init__(self) -> None:
        self.texts = []

    def encode_text(self, text: str) -> np.ndarray:
        self.texts.append(text)
        vector = np.zeros((384,), dtype=np.float32)
        vector[0] = 1.0
        return vector


class FakePersonStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def vector_search(self, query_vec, top_k=50, filter_pred=None):
        self.calls.append((int(top_k), len(query_vec)))
        filtered = [
            hit for hit in self.hits
            if filter_pred is None or filter_pred(hit["meta"])
        ]
        return filtered[: int(top_k)]


def _scene_meta(tmp_path: Path) -> Path:
    path = tmp_path / "scene_meta.json"
    path.write_text(
        json.dumps(
            [
                {
                    "faiss_id": 0,
                    "camera_id": 0,
                    "clip_path": r"S:\data\clips\camera_0\2026-03-01T17-01-00-000000__to__2026-03-01T17-01-15-000000.mp4",
                    "clip_url": "/media/clips/camera_0/2026-03-01T17-01-00-000000__to__2026-03-01T17-01-15-000000.mp4",
                    "frame_ts": "2026-03-01T17:01:00.000000",
                    "frame_index": 0,
                    "caption": "corridor with people",
                },
                {
                    "faiss_id": 1,
                    "camera_id": 0,
                    "clip_path": r"S:\data\clips\camera_0\2026-03-01T17-02-00-000000__to__2026-03-01T17-02-15-000000.mp4",
                    "clip_url": "/media/clips/camera_0/2026-03-01T17-02-00-000000__to__2026-03-01T17-02-15-000000.mp4",
                    "frame_ts": "2026-03-01T17:02:00.000000",
                    "frame_index": 0,
                    "caption": "entrance with people",
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _hit(timestamp, camera_id=0, score=0.7, color="Red"):
    return {
        "score": score,
        "meta": {
            "faiss_id": int(score * 100),
            "detection_id": "det-1",
            "object_index": 0,
            "camera_id": camera_id,
            "track_id": -1,
            "timestamp": timestamp,
            "color": color,
            "attribute_text": f"person, wearing {str(color).lower()} clothing",
        },
    }


def _person_meta(tmp_path: Path) -> Path:
    path = tmp_path / "person_meta.json"
    path.write_text("[{\"faiss_id\": 0}]", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# HybridSearchEngine.search_meta (offline path)
# ---------------------------------------------------------------------------
def test_search_meta_maps_person_hits_to_scene_clips(tmp_path):
    scene_path = _scene_meta(tmp_path)
    person_meta_path = _person_meta(tmp_path)
    store = FakePersonStore(
        [
            _hit("2026-03-01T17:01:14.000000", score=0.80, color="Red"),
            _hit("2026-03-01T17:01:16.000000", score=0.70, color="Blue"),
            _hit("2026-03-01T17:02:05.000000", score=0.60, color="Red"),
        ]
    )
    encoder = FakeEncoder()
    engine = HybridSearchEngine(
        person_store=store,
        attr_encoder=encoder,
        scene_meta_path=scene_path,
        person_meta_path=person_meta_path,
        enable_mongo=False,
    )

    results = engine.search_meta("person wearing red clothing", top_k=3)

    assert len(results) == 2  # two person hits collapse into the first clip
    first, second = results
    assert first["clip_path"].endswith("17-01-15-000000.mp4")
    assert first["clip_url"].startswith("/media/clips/camera_0/")
    assert first["source"] == "person"
    assert first["retrieval_branch"] == "person"
    assert first["timestamp"] == "2026-03-01T17:01:14.000000"
    assert first["score_raw"] == pytest.approx(0.80)
    assert first["score_norm"] == pytest.approx(1.0)
    assert len(first["person_hits"]) == 2
    assert first["frames"][0]["frame_ts"] == "2026-03-01T17:01:14.000000"
    assert second["score_norm"] == pytest.approx(0.0)
    assert second["clip_path"].endswith("17-02-15-000000.mp4")
    # Text only: query vector is 1152-d fused (zero visual + MiniLM text).
    assert store.calls[0][1] == 1152
    assert encoder.texts == ["person wearing red clothing"]


def test_search_meta_honours_camera_and_colour_filters(tmp_path):
    scene_path = _scene_meta(tmp_path)
    store = FakePersonStore(
        [
            _hit("2026-03-01T17:01:14.000000", camera_id=0, color="Red"),
            _hit("2026-03-01T17:01:16.000000", camera_id=0, color="Blue"),
        ]
    )
    engine = HybridSearchEngine(
        person_store=store,
        attr_encoder=FakeEncoder(),
        scene_meta_path=scene_path,
        person_meta_path=_person_meta(tmp_path),
        enable_mongo=False,
    )

    red = engine.search_meta(
        "person in red clothing", top_k=5, parsed_filter={"objects.color": "Red"}
    )
    assert len(red) == 1
    assert len(red[0]["person_hits"]) == 1
    assert red[0]["person_hits"][0]["color"] == "Red"

    wrong_camera = engine.search_meta(
        "person in red clothing", top_k=5, parsed_filter={"camera_id": 1}
    )
    assert wrong_camera == []


def test_search_meta_gender_off(tmp_path, monkeypatch):
    hit = _hit("2026-03-01T17:01:14.000000", score=0.75)
    hit["meta"]["attribute_text"] += ", female"
    store = FakePersonStore([hit])
    engine = HybridSearchEngine(
        person_store=store,
        attr_encoder=FakeEncoder(),
        scene_meta_path=_scene_meta(tmp_path),
        person_meta_path=_person_meta(tmp_path),
        enable_mongo=False,
    )
    monkeypatch.setattr(settings, "INCLUDE_GENDER_ATTRIBUTES", False)
    results = engine.search_meta("person carrying bag", top_k=2)
    # Gender text is removed from nested person hits as well as top-level text,
    # without corrupting the non-gender attributes.
    assert results
    hit_text = results[0]["person_hits"][0]["attribute_text"]
    assert "female" not in hit_text
    assert "male" not in hit_text
    assert "wearing red clothing" in hit_text


def test_person_index_availability(tmp_path):
    empty = tmp_path / "empty"
    assert not HybridSearchEngine.person_index_available(empty)
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "index.bin").write_bytes(b"not-a-real-index")
    (ready / "meta.json").write_text('[{"faiss_id": 0}]', encoding="utf-8")
    assert HybridSearchEngine.person_index_available(ready)


# ---------------------------------------------------------------------------
# UnifiedRetrieval routing + fallback
# ---------------------------------------------------------------------------
def _person_result():
    return {
        "camera_id": 0,
        "clip_path": r"S:\data\clips\camera_0\2026-03-01T17-01-00-000000__to__2026-03-01T17-01-15-000000.mp4",
        "clip_url": "/media/clips/camera_0/2026-03-01T17-01-00-000000__to__2026-03-01T17-01-15-000000.mp4",
        "timestamp": "2026-03-01T17:01:14.000000",
        "start": "2026-03-01T17:01:00.000000",
        "end": "2026-03-01T17:01:15.000000",
        "score": 0.8,
        "score_raw": 0.8,
        "score_norm": 1.0,
        "source": "person",
        "retrieval_branch": "person",
        "frames": [{"frame_ts": "2026-03-01T17:01:14.000000", "score": 0.8}],
        "person_hits": [{"attribute_text": "person, wearing red clothing"}],
    }


def _no_structured(*args, **kwargs):
    return []


class _FakeAnswerGenerator:
    def generate(self, **kwargs):
        return "fake answer"


def _run_unified(monkeypatch, tmp_path, *, person_available: bool):
    import backend.app.services.hybrid_search as hybrid_module

    class FakePersonEngine:
        @staticmethod
        def person_index_available():
            return person_available

        def __init__(self):
            pass

        def search_meta(self, query, top_k=10, parsed_filter=None):
            return [_person_result()]

    monkeypatch.setattr(hybrid_module, "HybridSearchEngine", FakePersonEngine)
    monkeypatch.setattr(ur, "execute_structured_query", _no_structured)
    monkeypatch.setattr(ur, "AnswerGenerator", _FakeAnswerGenerator)
    monkeypatch.setattr(
        ur,
        "search_unstructured",
        lambda **kwargs: {"semantic_results": [_person_result()]},
    )
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)

    retriever = ur.UnifiedRetrieval()
    retriever._build_clips = lambda results, parsed_filter: results
    parsed = {
        "query_type": "visual",
        "objects.object_name": "person",
        "objects.color": "Red",
        "__raw": "show me the person in red clothing",
        "__semantic_query": "person wearing red clothing",
    }
    return retriever.search(
        parsed, semantic_query="person wearing red clothing", limit=5
    )


def test_unified_search_uses_person_branch(monkeypatch, tmp_path):
    response = _run_unified(monkeypatch, tmp_path, person_available=True)
    assert response["metadata"]["retrieval_branch"] == "person"
    assert response["metadata"]["semantic_count"] == 1
    assert response["metadata"]["person_fallback"] is None
    assert any(
        "person-index" in step["details"] for step in response["processing_steps"]
    )


def test_unified_search_falls_back_to_scene(monkeypatch, tmp_path):
    response = _run_unified(monkeypatch, tmp_path, person_available=False)
    assert response["metadata"]["retrieval_branch"] == "scene"
    assert response["metadata"]["person_fallback"] == "person index unavailable"
    assert any(
        "person fallback" in step["details"]
        for step in response["processing_steps"]
    )


def test_should_route_person():
    assert ur.should_route_person(
        {"query_type": "visual", "objects.object_name": "person", "__raw": "person"}
    )
    assert ur.should_route_person(
        {"query_type": "visual", "objects.color": "Red", "__raw": "red jacket near gate"}
    )
    assert not ur.should_route_person(
        {"query_type": "visual", "objects.object_name": "person", "action": "running"}
    )
    assert not ur.should_route_person(
        {
            "query_type": "visual",
            "objects.object_name": "person",
            "count_constraint": {"gte": 2},
        }
    )
    assert not ur.should_route_person(
        {"query_type": "visual", "objects.object_name": "car", "__raw": "red car"}
    )


# ---------------------------------------------------------------------------
# Parser grounding / classification + startup regression
# ---------------------------------------------------------------------------
def test_reference_date_grounding(monkeypatch):
    monkeypatch.setattr(settings, "DATA_EPOCH_DATE", "2026-03-01")
    assert nl_parser.resolve_reference_date() == "2026-03-01"
    assert "{reference_date}" in nl_parser.PROMPT_TEMPLATE
    assert "Never default a timestamp to 2023" in nl_parser.PROMPT_TEMPLATE

    now = datetime(2026, 10, 21, 12, 0, 0)
    assert nl_parser.resolve_reference_date("", now=now) == "2026-10-21"
    assert nl_parser.resolve_reference_date("not-a-date", now=now) == "2026-10-21"


def test_count_queries_are_visual():
    parsed = {}
    nl_parser._augment_count_constraint_from_query("more than 3 people", parsed)
    assert parsed["count_constraint"] == {"gt": 3}
    assert (
        nl_parser.correct_query_type_for_count(
            "informational", parsed["count_constraint"]
        )
        == "visual"
    )
    assert (
        nl_parser.correct_query_type_for_count(
            "informational", parsed["count_constraint"], "alerts"
        )
        == "informational"
    )
    assert nl_parser.correct_query_type_for_count("visual", None) == "visual"


def test_mongo_import_is_fast_and_non_fatal():
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", "import backend.app.db.mongo; print('ok')"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.perf_counter() - started
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert elapsed < 15.0, f"Mongo import took {elapsed:.1f}s"
    assert settings.MONGO_SERVER_SELECTION_TIMEOUT_MS <= 2000
