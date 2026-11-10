"""DETECTION_CLASSES allowlist (scene props must not become detections)."""
from __future__ import annotations

import importlib

import pytest


def _reload(monkeypatch, value):
    monkeypatch.setenv("DETECTION_CLASSES", value)
    from backend.app import config as config_mod

    settings = config_mod.Settings()
    monkeypatch.setattr(config_mod, "settings", settings, raising=False)
    import backend.object_detection as od

    return importlib.reload(od)


def test_default_allowlist_keeps_surveillance_classes(monkeypatch):
    od = _reload(monkeypatch, "person,bicycle,car,motorcycle,bus,truck")
    assert od.is_allowed_class("person")
    assert od.is_allowed_class("Car")
    assert od.is_allowed_class("bicycle")


def test_default_allowlist_rejects_scene_props(monkeypatch):
    od = _reload(monkeypatch, "person,bicycle,car,motorcycle,bus,truck")
    for junk in ("bowl", "tv", "frisbee", "cell phone", "chair"):
        assert not od.is_allowed_class(junk), junk


def test_all_disables_the_filter(monkeypatch):
    od = _reload(monkeypatch, "all")
    assert od.ALLOWED_DETECTION_CLASSES is None
    assert od.is_allowed_class("bowl")


def test_empty_value_disables_the_filter(monkeypatch):
    od = _reload(monkeypatch, "")
    assert od.ALLOWED_DETECTION_CLASSES is None
