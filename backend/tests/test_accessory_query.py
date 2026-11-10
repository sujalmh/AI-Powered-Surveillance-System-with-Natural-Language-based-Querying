"""Accessory queries are not colour queries, and accessories reach the CLIP text."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.nl_parser import build_embedding_text  # noqa: E402


def test_accessory_is_appended_to_embedding_text():
    f = {"objects.object_name": "person", "__visual_attributes": ["carrying bag"]}
    assert build_embedding_text(f, "") == "person carrying bag"


def test_colour_and_accessory_together():
    f = {
        "objects.object_name": "person",
        "objects.color": "Red",
        "__visual_attributes": ["carrying bag"],
    }
    assert build_embedding_text(f, "") == "person wearing red clothing carrying bag"


def test_no_accessory_is_unchanged():
    f = {"objects.object_name": "person"}
    assert build_embedding_text(f, "") == "person"


def test_bag_is_not_a_colour_question():
    import re

    # Mirrors the regex in backend/app/routers/chat.py
    ask_color_re = r"\b(color|colour|clothes|clothing|wearing|shirt|jacket|dress|pants)\b"
    assert not re.search(ask_color_re, "show people carrying a bag")
    assert not re.search(ask_color_re, "find a backpack")
    assert re.search(ask_color_re, "show people wearing red")
