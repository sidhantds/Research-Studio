"""Shared setup. Every test here runs offline: no API key, no network, no model calls."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-used")  # agent.py loads .env; nothing calls the API

from agent import ToolStep  # noqa: E402


@pytest.fixture
def web_step():
    """A web search result in the exact format tools._fetch_web returns."""
    output = (
        "Sodium-ion battery - Overview\n"
        "Sodium-ion batteries use sodium ions as charge carriers. They are cheaper than lithium-ion "
        "cells because sodium is abundant. Their energy density is lower than lithium-ion.\n"
        "URL: https://www.example.org/sodium-ion\n\n"
        "Grid storage report\n"
        "Utilities are testing sodium-ion packs for grid storage in 2026.\n"
        "URL: https://energy.example.com/report"
    )
    return ToolStep(tool="duckduckgo_search", tool_input="sodium-ion batteries grid storage", output=output)
