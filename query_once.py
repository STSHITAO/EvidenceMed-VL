#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_rag.config import load_settings  # noqa: E402
from medical_rag.pipeline import MedicalRAGPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Medical-RAG query from repository root.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "settings.yaml"))
    parser.add_argument("--image", type=str, required=True, help="Path to medical image.")
    parser.add_argument("--question", type=str, required=True, help="Natural language question.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    pipeline = MedicalRAGPipeline(settings)
    result = pipeline.ask(image_path=args.image, question=args.question)

    print("\n===== ANSWER =====")
    print(result.answer)
    print("\n===== EVIDENCE =====")
    for i, e in enumerate(result.evidence, start=1):
        print(f"[{i}] score={e.score:.4f} source={e.source} chunk={e.chunk_index}")


if __name__ == "__main__":
    main()
