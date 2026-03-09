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
from medical_rag.retrieval import MilvusVectorStore, create_embedder  # noqa: E402
from medical_rag.utils.text_chunker import build_chunks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Medical-RAG Milvus index from repository root.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "settings.yaml"))
    parser.add_argument("--drop-old", action="store_true", help="Drop existing collection before indexing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    chunks = build_chunks(
        doc_dir=settings.knowledge.doc_dir,
        chunk_size=settings.knowledge.chunk_size,
        chunk_overlap=settings.knowledge.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError(f"No chunks generated from {settings.knowledge.doc_dir}. Add docs first.")

    embedder = create_embedder(settings.embedding)
    vectors = embedder.encode([c.content for c in chunks])
    dim = vectors.shape[1]

    store = MilvusVectorStore(settings.milvus)
    store.connect()
    if args.drop_old:
        store.recreate_collection(vector_dim=dim)
    else:
        store.ensure_collection(vector_dim=dim)

    inserted = store.insert_chunks(chunks, vectors)
    print(f"[Medical-RAG] chunks={len(chunks)} inserted={inserted} collection={settings.milvus.collection_name}")


if __name__ == "__main__":
    main()
