from __future__ import annotations

from typing import List

from medical_rag.config import Settings
from medical_rag.retrieval import MilvusVectorStore, MultiRouteRetriever, create_embedder, create_reranker
from medical_rag.schemas import RagAnswer, RetrievedChunk
from medical_rag.vlm import QwenVLMReasoner


class MedicalRAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.embedder = create_embedder(settings.embedding)

        self.store = MilvusVectorStore(settings.milvus)
        self.store.connect()
        self.store.load()

        self.retriever = MultiRouteRetriever(
            embedder=self.embedder,
            store=self.store,
            route_top_k=max(4, settings.project.top_k_recall // 2),
        )

        self.reranker = create_reranker(settings.reranker)

        self.reasoner = QwenVLMReasoner(settings.vlm)

    @staticmethod
    def _build_evidence_blocks(items: List[RetrievedChunk]) -> List[str]:
        blocks: List[str] = []
        for item in items:
            blocks.append(
                f"来源: {item.source} | chunk: {item.chunk_index} | score: {item.score:.4f}\n{item.content}"
            )
        return blocks

    def ask(self, image_path: str, question: str) -> RagAnswer:
        recalled = self.retriever.recall(question, top_k=self.settings.project.top_k_recall)
        reranked = self.reranker.rerank(question, recalled, top_k=self.settings.project.top_k_rerank)

        if not reranked:
            reranked = [
                RetrievedChunk(
                    doc_id="none",
                    source="none",
                    chunk_index=-1,
                    content="未检索到可用医学证据，请补充指南文档并重建索引。",
                    score=0.0,
                )
            ]

        evidence_blocks = self._build_evidence_blocks(reranked)
        answer = self.reasoner.generate(image_path=image_path, question=question, evidence_blocks=evidence_blocks)
        return RagAnswer(answer=answer, evidence=reranked)
