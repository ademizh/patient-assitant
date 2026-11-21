"""Application settings, read from environment variables (optionally via a .env file).

Nothing here is logged. The API key is only held in memory and passed to the OpenAI client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root if present (does not override real env vars).
load_dotenv(PROJECT_ROOT / ".env", override=False)

# An empty variable (e.g. "OPENAI_BASE_URL=" copied from .env.example) would be picked up by the
# OpenAI SDK as an empty server address. Treat empty values as "not set".
for _var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
    if _var in os.environ and not os.environ[_var].strip():
        del os.environ[_var]


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # "auto" -> live when OPENAI_API_KEY is set, otherwise demo. "demo" / "live" force a mode.
    app_mode: str = "auto"
    openai_api_key: str | None = field(default=None, repr=False)
    openai_base_url: str | None = None

    chat_model: str = "gpt-6-luna"
    transcription_model: str = "gpt-transcribe"
    embedding_model: str = "text-embedding-3-small"

    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"
    library_path: Path = PROJECT_ROOT / "data" / "library_files"
    # Patient appointment records (JSON + uploads) - kept apart from the knowledge base.
    appointments_path: Path = PROJECT_ROOT / "data" / "appointments"
    # Urgent-care pathway rules. The bundled file is a DEMONSTRATION rule set (not validated).
    urgent_rules_path: Path = PROJECT_ROOT / "data" / "pathways" / "DEMO_urgent_rules_ru.json"
    collection_prefix: str = "medical_docs"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    # Cosine-similarity floor below which a retrieved chunk is not treated as evidence.
    # None -> mode-specific default (demo hash vectors score lower than real embeddings).
    min_relevance_override: float | None = None

    max_audio_mb: int = 25
    # ISO-639-1 hint for speech recognition; empty -> let the model detect the language.
    transcription_language: str | None = "ru"

    @property
    def mode(self) -> str:
        """Resolved mode: 'live' or 'demo'."""
        requested = (self.app_mode or "auto").lower()
        if requested == "demo":
            return "demo"
        if requested == "live":
            return "live"
        return "live" if self.openai_api_key else "demo"

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @property
    def min_relevance(self) -> float:
        if self.min_relevance_override is not None:
            return self.min_relevance_override
        return 0.12 if self.is_demo else 0.30


def load_settings() -> Settings:
    root = PROJECT_ROOT
    chroma = Path(os.getenv("CHROMA_PATH", root / "data" / "chroma"))
    library = Path(os.getenv("LIBRARY_PATH", root / "data" / "library_files"))
    if not chroma.is_absolute():
        chroma = root / chroma
    if not library.is_absolute():
        library = root / library
    appointments = Path(os.getenv("APPOINTMENTS_PATH", root / "data" / "appointments"))
    if not appointments.is_absolute():
        appointments = root / appointments
    rules = Path(os.getenv("URGENT_RULES_PATH", root / "data" / "pathways" / "DEMO_urgent_rules_ru.json"))
    if not rules.is_absolute():
        rules = root / rules
    return Settings(
        app_mode=os.getenv("APP_MODE", "auto"),
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        openai_base_url=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna"),
        transcription_model=os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-transcribe"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_path=chroma,
        library_path=library,
        appointments_path=appointments,
        urgent_rules_path=rules,
        collection_prefix=os.getenv("CHROMA_COLLECTION_PREFIX", "medical_docs"),
        chunk_size=_int("CHUNK_SIZE", 900),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        top_k=_int("RETRIEVAL_TOP_K", 4),
        min_relevance_override=_float("MIN_RELEVANCE", -1.0) if os.getenv("MIN_RELEVANCE") else None,
        max_audio_mb=_int("MAX_AUDIO_MB", 25),
        transcription_language=(os.getenv("TRANSCRIPTION_LANGUAGE", "ru") or "").strip() or None,
    )
