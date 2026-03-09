from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from medical_rag.retrieval.embedding import Embedder
from medical_rag.retrieval.vector_store import MilvusVectorStore
from medical_rag.schemas import RetrievedChunk


class MultiRouteRetriever:
    """Dense multi-route recall + reciprocal rank fusion."""

    def __init__(self, embedder: Embedder, store: MilvusVectorStore, route_top_k: int = 8) -> None:
        self.embedder = embedder
        self.store = store
        self.route_top_k = route_top_k

    @staticmethod
    def _expand_queries(query: str) -> List[str]:
        q = query.strip()
        return [
            q,
            f"影像征象 {q}",
            f"临床指南 {q}",
        ]

    def recall(self, query: str, top_k: int) -> List[RetrievedChunk]:
        queries = self._expand_queries(query)
        rrf_k = 60
        fused_score: Dict[str, float] = defaultdict(float)
        hit_cache: Dict[str, RetrievedChunk] = {}

        for route_query in queries:
            route_vec = self.embedder.encode([route_query])[0]
            route_hits = self.store.search(route_vec, top_k=self.route_top_k)
            for rank, hit in enumerate(route_hits, start=1):
                key = f"{hit.source}::{hit.chunk_index}"
                fused_score[key] += 1.0 / (rrf_k + rank)
                if key not in hit_cache:
                    hit_cache[key] = hit

        ranked = sorted(fused_score.items(), key=lambda x: x[1], reverse=True)
        merged: List[RetrievedChunk] = []
        for key, score in ranked[:top_k]:
            item = hit_cache[key]
            merged.append(
                RetrievedChunk(
                    doc_id=item.doc_id,
                    source=item.source,
                    chunk_index=item.chunk_index,
                    content=item.content,
                    score=score,
                )
            )
        return merged
