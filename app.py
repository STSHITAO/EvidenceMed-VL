#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medical_rag.app import build_demo  # noqa: E402
from medical_rag.config import load_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Medical-RAG web app from repository root.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "settings.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    demo = build_demo(settings)
    demo.launch(server_name=settings.service.host, server_port=settings.service.port, share=settings.service.share)


if __name__ == "__main__":
    main()
