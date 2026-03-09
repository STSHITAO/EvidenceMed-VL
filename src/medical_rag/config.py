from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    name: str
    language: str
    top_k_recall: int
    top_k_rerank: int


@dataclass
class KnowledgeConfig:
    doc_dir: str
    chunk_size: int
    chunk_overlap: int


@dataclass
class MilvusConfig:
    uri: str = ""
    host: str = "127.0.0.1"
    port: int = 19530
    user: str = ""
    password: str = ""
    db_name: str = "default"
    collection_name: str = "medical_guidelines"
    consistency_level: str = "Strong"


@dataclass
class EmbeddingConfig:
    model_name: str
    provider: str = "api"
    use_fp16: bool = True
    batch_size: int = 16
    max_length: int = 1024
    api_base_url: str = ""
    api_key: str = ""
    api_key_env: str = "MODELSCOPE_API_TOKEN"
    api_embedding_path: str = "/v1/embeddings"
    api_style: str = "openai"
    timeout_sec: int = 60


@dataclass
class RerankerConfig:
    model_name: str
    provider: str = "api"
    use_fp16: bool = True
    batch_size: int = 16
    api_base_url: str = ""
    api_key: str = ""
    api_key_env: str = "MODELSCOPE_API_TOKEN"
    api_rerank_path: str = "/v1/rerank"
    api_style: str = "openai"
    timeout_sec: int = 60


@dataclass
class VLMConfig:
    base_model_path: str
    lora_adapter_path: str
    device: str
    dtype: str
    max_new_tokens: int
    temperature: float
    backend: str = "vllm_openai"
    model_name: str = ""
    api_base_url: str = ""
    api_key: str = ""
    api_key_env: str = "VLLM_API_KEY"
    api_chat_path: str = "/v1/chat/completions"
    request_timeout_sec: int = 120


@dataclass
class ServiceConfig:
    host: str
    port: int
    share: bool


@dataclass
class Settings:
    project: ProjectConfig
    knowledge: KnowledgeConfig
    milvus: MilvusConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    vlm: VLMConfig
    service: ServiceConfig


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(path: str | Path) -> Settings:
    data = _read_yaml(path)
    return Settings(
        project=ProjectConfig(**data["project"]),
        knowledge=KnowledgeConfig(**data["knowledge"]),
        milvus=MilvusConfig(**data["milvus"]),
        embedding=EmbeddingConfig(**data["embedding"]),
        reranker=RerankerConfig(**data["reranker"]),
        vlm=VLMConfig(**data["vlm"]),
        service=ServiceConfig(**data["service"]),
    )
