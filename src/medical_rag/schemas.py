from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DocChunk:
    doc_id: str
    source: str
    chunk_index: int
    content: str


@dataclass
class RetrievedChunk:
    doc_id: str
    source: str
    chunk_index: int
    content: str
    score: float


@dataclass
class RagAnswer:
    answer: str
    evidence: List[RetrievedChunk]
