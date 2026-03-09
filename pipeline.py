#!/usr/bin/env python3
"""Top-level pipeline exports for Medical-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_rag.pipeline import MedicalRAGPipeline  # noqa: E402

__all__ = ["MedicalRAGPipeline"]
