"""Tests for access control, audit logging, retention, face blur, gender-off."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.core import privacy, security
from backend.app.core.security import (
    AccessControlMiddleware,
    AccessDenied,
    Principal,
    Role,
    audit_log,
    authorize_camera,
    filter_authorized_results,
    is_camera_allowed,
    is_zone_allowed,
    principal_from_headers,
)


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
def test_role_hierarchy_and_camera_zone_grants():
    viewer = Principal(role=Role.VIEWER, actor="v", allowed_cameras=frozenset({0}),
                       allowed_zones=frozenset({"lobby"}))
    assert viewer.can(Role.VIEWER)
    assert not viewer.can(Role.OPERATOR)
    assert is_camera_allowed(viewer, 0)
    assert not is_camera_allowed(viewer, 1)
    assert is_zone_allowed(viewer, "Lobby")
    assert not is_zone_allowed(viewer, "parking")
    with pytest.raises(AccessDenied):
        authorize_camera(viewer, 2)

    admin = Principal(role=Role.ADMIN, actor="a")
    assert admin.can(Role.ADMIN)
    assert is_camera_allowed(admin, 99)
    assert is_zone_allowed(admin, "anywhere")

    records = [{"camera_id": 0, "zone": "lobby"}, {"camera_id": 1, "zone": "lobby"}]
    assert filter_authorized_results(viewer, records) == [records[0]]


def test_principal_from_headers_is_case_insensitive():
    headers = {
        "x-role": "operator",
        "x-actor": "alice",
        "x-allowed-cameras": "0, 2",
        "x-allowed-zones": "Gate 3,lobby",
    }
    principal = principal_from_headers(headers)
    assert principal.role == Role.OPERATOR
    assert principal.actor == "alice"
    assert principal.allowed_cameras == frozenset({0, 2})
    assert principal.allowed_zones == frozenset({"gate 3", "lobby"})

    with pytest.raises(AccessDenied):
        principal_from_headers({})
    with pytest.raises(AccessDenied):
        principal_from_headers({"X-Role": "superuser"})


def test_access_control_middleware_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_CONTROL_ENABLED", True)
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    app = FastAPI()
    app.add_middleware(AccessControlMiddleware)

    @app.get("/api/ping")
    def ping(camera_id: int | None = None, zone: str | None = None):
        return {"ok": True}

    @app.post("/api/write")
    def write():
        return {"ok": True}

    client = TestClient(app)

    assert client.get("/api/ping").status_code == 401
    assert client.get("/api/ping", headers={"X-Role": "viewer"}).status_code == 200
    assert client.post("/api/write", headers={"X-Role": "viewer"}).status_code == 403
    assert client.post("/api/write", headers={"X-Role": "operator"}).status_code == 200
    assert (
        client.get(
            "/api/ping?camera_id=2",
            headers={"X-Role": "viewer", "X-Allowed-Cameras": "0"},
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/ping?zone=parking",
            headers={"X-Role": "viewer", "X-Allowed-Zones": "lobby"},
        ).status_code
        == 403
    )

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 5
    records = [json.loads(line) for line in lines]
    assert any(r["outcome"] == "denied" for r in records)
    assert any(r["status_code"] == 200 for r in records)


def test_access_control_disabled_preserves_open_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_CONTROL_ENABLED", False)
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    app = FastAPI()
    app.add_middleware(AccessControlMiddleware)

    @app.get("/api/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/api/ping").status_code == 200
    assert not (tmp_path / "audit.jsonl").exists()


def test_audit_jsonl_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    record = audit_log(
        "access.request",
        actor="alice",
        role="operator",
        action="GET /api/chat",
        path="/api/chat",
        method="GET",
        outcome="success",
        status_code=200,
        camera_id=0,
        zone="lobby",
        details={"duration_ms": 1.25},
        log_path=path,
    )
    assert record["event"] == "access.request"
    parsed = json.loads(path.read_text(encoding="utf-8").strip())
    assert parsed["actor"] == "alice"
    assert parsed["role"] == "operator"
    assert parsed["camera_id"] == 0
    assert parsed["details"]["duration_ms"] == 1.25
    assert parsed["timestamp"].endswith("+00:00")


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------
def _touch(path: Path, when: datetime, payload: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    stamp = when.timestamp()
    os.utime(path, (stamp, stamp))


def test_retention_dry_run_then_real(tmp_path):
    now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    dirs = {
        "clips": tmp_path / "clips",
        "snapshots": tmp_path / "snapshots",
        "recordings": tmp_path / "recordings",
    }
    old = now - timedelta(hours=20)
    fresh = now - timedelta(hours=1)
    _touch(dirs["clips"] / "camera_0" / "old.mp4", old, b"old-clip")
    _touch(dirs["clips"] / "camera_0" / "new.mp4", fresh, b"new-clip")
    _touch(dirs["snapshots"] / "camera_0" / "old.jpg", old, b"old-snap")
    _touch(dirs["recordings"] / "old.mp4", old, b"old-rec")

    dry = privacy.prune_old_media(dirs, retention_hours=8, dry_run=True, now=now)
    assert dry["dry_run"] is True
    assert dry["candidate_files"] == 3
    assert dry["deleted_files"] == 0
    assert dirs["clips"] / "camera_0" / "old.mp4" in [Path(c["path"]) for c in dry["candidates"]]
    assert (dirs["clips"] / "camera_0" / "old.mp4").exists()

    real = privacy.prune_old_media(dirs, retention_hours=8, dry_run=False, now=now)
    assert real["deleted_files"] == 3
    assert real["bytes_reclaimed"] == len(b"old-clip") + len(b"old-snap") + len(b"old-rec")
    assert not (dirs["clips"] / "camera_0" / "old.mp4").exists()
    assert not (dirs["snapshots"] / "camera_0" / "old.jpg").exists()
    assert not (dirs["recordings"] / "old.mp4").exists()
    assert (dirs["clips"] / "camera_0" / "new.mp4").exists()
    # Directories are never removed by retention.
    assert (dirs["clips"] / "camera_0").is_dir()


def test_retention_policy_requires_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RETENTION_ENABLED", False)
    monkeypatch.setattr(settings, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(settings, "SNAPSHOTS_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(settings, "RECORDINGS_DIR", tmp_path / "recordings")
    _touch(tmp_path / "clips" / "old.mp4", datetime.now(timezone.utc) - timedelta(hours=30))
    report = privacy.run_retention_policy(dry_run=False)
    assert report["skipped"] is True
    assert (tmp_path / "clips" / "old.mp4").exists()


def test_retention_policy_runs_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RETENTION_ENABLED", True)
    monkeypatch.setattr(settings, "RETENTION_HOURS", 8.0)
    monkeypatch.setattr(settings, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(settings, "SNAPSHOTS_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(settings, "RECORDINGS_DIR", tmp_path / "recordings")
    _touch(tmp_path / "clips" / "old.mp4", datetime.now(timezone.utc) - timedelta(hours=30))
    report = privacy.run_retention_policy(dry_run=True)
    assert report["skipped"] is False
    assert report["dry_run"] is True
    assert report["candidate_files"] == 1


# ---------------------------------------------------------------------------
# Face blur
# ---------------------------------------------------------------------------
def test_face_blur_runs_without_a_face():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    blurred, count = privacy.blur_faces_with_count(image)
    assert count == 0
    assert blurred.shape == image.shape
    assert blurred.dtype == image.dtype
    assert np.array_equal(blurred, image)

    gray = np.zeros((32, 48), dtype=np.uint8)
    out = privacy.blur_faces(gray)
    assert out.shape == gray.shape
    assert out.dtype == gray.dtype


def test_face_blur_file_requires_switch_or_force(tmp_path, monkeypatch):
    import cv2

    source = tmp_path / "frame.jpg"
    image = np.full((48, 64, 3), 200, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)

    monkeypatch.setattr(settings, "FACE_BLUR_ENABLED", False)
    skipped = privacy.blur_faces_in_file(source)
    assert skipped["skipped"] is True
    assert skipped["faces_blurred"] == 0

    forced = privacy.blur_faces_in_file(source, force=True)
    assert forced["skipped"] is False
    assert forced["faces_blurred"] == 0
    assert source.exists()
    out = cv2.imread(str(source))
    assert out is not None and out.shape == image.shape

    monkeypatch.setattr(settings, "FACE_BLUR_ENABLED", True)
    enabled = privacy.blur_faces_in_file(source)
    assert enabled["skipped"] is False


# ---------------------------------------------------------------------------
# Gender-off
# ---------------------------------------------------------------------------
def test_gender_filter_strips_terms():
    text = "person, wearing red clothing, carrying bag, female"
    assert privacy.strip_gender_terms(text) == "person, wearing red clothing, carrying bag"
    record = {
        "caption": "person, male",
        "attribute_text": "person wearing blue coat, male",
        "objects": [{"attribute_text": "person, female"}],
        "person_hits": [{"attribute_text": "person, female"}],
        "frames": [{"caption": "person, male"}],
        "detection": {"caption": "person, female"},
    }
    cleaned = privacy.strip_gender_from_record(record)
    assert "male" not in cleaned["attribute_text"]
    assert "female" not in cleaned["objects"][0]["attribute_text"]
    assert "male" not in cleaned["caption"]
    assert "female" not in cleaned["person_hits"][0]["attribute_text"]
    assert "male" not in cleaned["frames"][0]["caption"]
    assert "female" not in cleaned["detection"]["caption"]


def test_gender_switch_controls_attribute_text(monkeypatch):
    from backend.app.services.attribute_encoder import AttributeEncoder

    encoder = AttributeEncoder.__new__(AttributeEncoder)  # no model load
    attrs = {"male_confidence": 0.9, "bag_confidence": 0.9}

    monkeypatch.setattr(settings, "INCLUDE_GENDER_ATTRIBUTES", True)
    with_gender = encoder.attributes_to_text(attrs, "Red")
    assert "male" in with_gender
    assert "carrying bag" in with_gender

    monkeypatch.setattr(settings, "INCLUDE_GENDER_ATTRIBUTES", False)
    with_gender_off = encoder.attributes_to_text(attrs, "Red")
    assert "male" not in with_gender_off
    assert "female" not in with_gender_off
    assert "wearing red clothing" in with_gender_off
