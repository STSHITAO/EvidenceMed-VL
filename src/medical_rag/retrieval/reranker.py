from __future__ import annotations

import os
from typing import List, Protocol

import requests

from medical_rag.config import RerankerConfig
from medical_rag.schemas import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        ...


class APIReranker:
    """HTTP API reranker for OpenAI-compatible or generic rerank endpoints."""

    def __init__(
        self,
        model_name: str,
        api_base_url: str,
        api_key: str,
        api_key_env: str,
        api_rerank_path: str,
        api_style: str,
        timeout_sec: int,
    ) -> None:
        self.model_name = model_name
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.api_rerank_path = api_rerank_path
        self.api_style = api_style
        self.timeout_sec = timeout_sec

        if not self.api_base_url:
            raise ValueError("reranker.api_base_url is required when reranker.provider=api")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_scores(self, body: dict, n_docs: int) -> List[float]:
        scores = [0.0] * n_docs

        # OpenAI-style rerank response: {"results": [{"index": i, "relevance_score": s}, ...]}
        if isinstance(body.get("results"), list):
            for item in body["results"]:
                idx = int(item.get("index", -1))
                if 0 <= idx < n_docs:
                    score = item.get("relevance_score", item.get("score", 0.0))
                    scores[idx] = float(score)
            return scores

        # Alternative style: {"data": [{"index": i, "score": s}, ...]}
        if isinstance(body.get("data"), list):
            for item in body["data"]:
                idx = int(item.get("index", -1))
                if 0 <= idx < n_docs:
                    score = item.get("relevance_score", item.get("score", 0.0))
                    scores[idx] = float(score)
            return scores

        # Simple style: {"scores": [s1, s2, ...]}
        if isinstance(body.get("scores"), list):
            arr = body["scores"]
            for i in range(min(n_docs, len(arr))):
                scores[i] = float(arr[i])
            return scores

        raise ValueError(f"Unsupported rerank API response format: keys={list(body.keys())}")

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not candidates:
            return []

        documents = [item.content for item in candidates]
        url = f"{self.api_base_url}{self.api_rerank_path}"

        if self.api_style == "openai":
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": max(top_k, len(documents)),
            }
        else:
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents,
            }

        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        body = resp.json()
        scores = self._extract_scores(body, n_docs=len(candidates))

        rescored: List[RetrievedChunk] = []
        for item, score in zip(candidates, scores):
            rescored.append(
                RetrievedChunk(
                    doc_id=item.doc_id,
                    source=item.source,
                    chunk_index=item.chunk_index,
                    content=item.content,
                    score=float(score),
                )
            )

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]


def create_reranker(config: RerankerConfig) -> Reranker:
    provider = (config.provider or "api").lower()
    if provider != "api":
        raise ValueError(
            f"Unsupported reranker.provider={config.provider}. "
            "This project only supports reranker.provider=api (vLLM /v1/rerank)."
        )

    return APIReranker(
        model_name=config.model_name,
        api_base_url=config.api_base_url,
        api_key=config.api_key,
        api_key_env=config.api_key_env,
        api_rerank_path=config.api_rerank_path,
        api_style=config.api_style,
        timeout_sec=config.timeout_sec,
    )
