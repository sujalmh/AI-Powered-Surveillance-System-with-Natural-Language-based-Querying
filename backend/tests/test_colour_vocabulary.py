"""Colour vocabulary must match what the detector can emit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.routers.chat import COLOR_ALIASES  # noqa: E402


def test_aliases_target_detector_colours():
    # Import lazily: object_detection loads YOLO at import time, which is slow
    # and unnecessary for the pure-data assertion below.
    from backend.object_detection import SURVEILLANCE_COLORS

    detector = {name for name in SURVEILLANCE_COLORS}
    parser = set(COLOR_ALIASES.values())
    unknown = parser - detector
    assert not unknown, f"parser emits colours the detector never produces: {sorted(unknown)}"


def test_every_alias_is_reachable():
    assert COLOR_ALIASES["olive"] == "Olive"
    assert COLOR_ALIASES["grey"] == "Gray"
    assert COLOR_ALIASES["navy"] == "Blue"
    assert COLOR_ALIASES["crimson"] == "Red"
    assert COLOR_ALIASES["teal"] == "Cyan"


def test_colour_names_are_title_cased():
    for value in COLOR_ALIASES.values():
        assert value == value.title(), value
