#!/usr/bin/env python3
"""Top-level config exports for Medical-RAG.

This file keeps a simple public import surface at repository root while
the actual implementation lives under ``src/medical_rag``.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_rag.config import (  # noqa: E402
    EmbeddingConfig,
    KnowledgeConfig,
    MilvusConfig,
    ProjectConfig,
    RerankerConfig,
    ServiceConfig,
    Settings,
    VLMConfig,
    load_settings,
)

__all__ = [
    "ProjectConfig",
    "KnowledgeConfig",
    "MilvusConfig",
    "EmbeddingConfig",
    "RerankerConfig",
    "VLMConfig",
    "ServiceConfig",
    "Settings",
    "load_settings",
]
