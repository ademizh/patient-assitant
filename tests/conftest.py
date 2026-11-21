from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intake_assistant.config import load_settings  # noqa: E402
from intake_assistant.providers import DemoProvider  # noqa: E402
from intake_assistant.rag.store import DocumentStore  # noqa: E402

SAMPLE_DOCS = ROOT / "data" / "sample_documents"


@pytest.fixture()
def demo_settings(tmp_path):
    s = load_settings()
    return replace(
        s,
        app_mode="demo",
        openai_api_key=None,
        chroma_path=tmp_path / "chroma",
        library_path=tmp_path / "lib",
        appointments_path=tmp_path / "appointments",
    )


@pytest.fixture()
def demo_provider(demo_settings):
    return DemoProvider(demo_settings)


@pytest.fixture()
def store(demo_settings, demo_provider):
    return DocumentStore(demo_settings.chroma_path, demo_settings.library_path, "test_docs", demo_provider)


@pytest.fixture()
def indexed_store(store):
    for p in sorted(SAMPLE_DOCS.iterdir()):
        store.add_file(p.name, p.read_bytes(), 900, 150)
    return store
