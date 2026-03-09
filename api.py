#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastAPI backend from repository root.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "settings.yaml"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["MEDICAL_RAG_CONFIG"] = args.config
    uvicorn.run("medical_rag.api_server:app", host=args.host, port=args.port, reload=args.reload, workers=1)


if __name__ == "__main__":
    main()

