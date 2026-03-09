#!/usr/bin/env python3
"""Top-level prompt exports for Medical-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_rag.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

__all__ = ["SYSTEM_PROMPT", "build_user_prompt"]
