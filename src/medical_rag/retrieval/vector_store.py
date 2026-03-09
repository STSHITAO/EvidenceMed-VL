from __future__ import annotations

from typing import List

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from medical_rag.config import MilvusConfig
from medical_rag.schemas import DocChunk, RetrievedChunk


class MilvusVectorStore:
    def __init__(self, config: MilvusConfig) -> None:
        self.config = config
        self.collection: Collection | None = None

    def connect(self) -> None:
        if self.config.uri:
            connections.connect(alias="default", uri=self.config.uri)
            return

        connections.connect(
            alias="default",
            host=self.config.host,
            port=str(self.config.port),
            user=self.config.user or None,
            password=self.config.password or None,
            db_name=self.config.db_name,
        )

    def _build_schema(self, vector_dim: int) -> CollectionSchema:
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        ]
        return CollectionSchema(fields=fields, description="Medical guideline chunks")

    def ensure_collection(self, vector_dim: int) -> None:
        name = self.config.collection_name
        if utility.has_collection(name):
            self.collection = Collection(name)
            return

        schema = self._build_schema(vector_dim)
        self.collection = Collection(name=name, schema=schema, consistency_level=self.config.consistency_level)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
            "params": {},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)

    def recreate_collection(self, vector_dim: int) -> None:
        name = self.config.collection_name
        if utility.has_collection(name):
            utility.drop_collection(name)
        self.ensure_collection(vector_dim)

    def insert_chunks(self, chunks: List[DocChunk], vectors: np.ndarray) -> int:
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized.")
        if len(chunks) == 0:
            return 0
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors size mismatch")

        entities = [
            [c.doc_id for c in chunks],
            [c.source for c in chunks],
            [c.chunk_index for c in chunks],
            [c.content for c in chunks],
            vectors.tolist(),
        ]
        result = self.collection.insert(entities)
        self.collection.flush()
        return len(result.primary_keys)

    def load(self) -> None:
        if self.collection is None:
            self.collection = Collection(self.config.collection_name)
        self.collection.load()

    def search(self, query_vector: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        if self.collection is None:
            self.collection = Collection(self.config.collection_name)
            self.collection.load()

        search_params = {"metric_type": "COSINE", "params": {}}
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "source", "chunk_index", "content"],
        )

        merged: List[RetrievedChunk] = []
        for hit in results[0]:
            merged.append(
                RetrievedChunk(
                    doc_id=hit.entity.get("doc_id"),
                    source=hit.entity.get("source"),
                    chunk_index=int(hit.entity.get("chunk_index")),
                    content=hit.entity.get("content"),
                    score=float(hit.distance),
                )
            )
        return merged
