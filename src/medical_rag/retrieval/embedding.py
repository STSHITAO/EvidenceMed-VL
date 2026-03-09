from __future__ import annotations

import os
from typing import Iterable, List, Protocol

import numpy as np
import requests

from medical_rag.config import EmbeddingConfig


class Embedder(Protocol):
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        ...


class APIEmbedder:
    """HTTP API embedder for OpenAI-compatible embedding endpoints."""

    def __init__(
        self,
        model_name: str,
        api_base_url: str,
        api_key: str,
        api_key_env: str,
        api_embedding_path: str,
        api_style: str,
        timeout_sec: int,
        batch_size: int,
    ) -> None:
        self.model_name = model_name
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.api_embedding_path = api_embedding_path
        self.api_style = api_style
        self.timeout_sec = timeout_sec
        self.batch_size = batch_size

        if not self.api_base_url:
            raise ValueError("embedding.api_base_url is required when embedding.provider=api")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _parse_embeddings(self, data: dict) -> List[List[float]]:
        if "data" in data and isinstance(data["data"], list):
            # OpenAI-compatible: [{index, embedding}, ...]
            ordered = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in ordered]
        if "embeddings" in data and isinstance(data["embeddings"], list):
            return data["embeddings"]
        if "output" in data and isinstance(data["output"], dict) and "embeddings" in data["output"]:
            return data["output"]["embeddings"]
        raise ValueError(f"Unsupported embedding API response format: keys={list(data.keys())}")

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list: List[str] = list(texts)
        if not text_list:
            return np.zeros((0, 1), dtype=np.float32)

        url = f"{self.api_base_url}{self.api_embedding_path}"
        all_vectors: List[List[float]] = []

        for i in range(0, len(text_list), self.batch_size):
            batch = text_list[i : i + self.batch_size]
            if self.api_style == "openai":
                payload = {"model": self.model_name, "input": batch}
            else:
                payload = {"model": self.model_name, "texts": batch}

            resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            body = resp.json()
            vectors = self._parse_embeddings(body)
            all_vectors.extend(vectors)

        return np.asarray(all_vectors, dtype=np.float32)


def create_embedder(config: EmbeddingConfig) -> Embedder:
    provider = (config.provider or "api").lower()
    if provider != "api":
        raise ValueError(
            f"Unsupported embedding.provider={config.provider}. "
            "This project only supports embedding.provider=api (vLLM /v1/embeddings)."
        )

    return APIEmbedder(
        model_name=config.model_name,
        api_base_url=config.api_base_url,
        api_key=config.api_key,
        api_key_env=config.api_key_env,
        api_embedding_path=config.api_embedding_path,
        api_style=config.api_style,
        timeout_sec=config.timeout_sec,
        batch_size=config.batch_size,
    )
