from __future__ import annotations

import mimetypes
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from medical_rag.config import Settings, load_settings
from medical_rag.pipeline import MedicalRAGPipeline

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "settings.yaml"


class _PipelineRuntime:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self._settings: Settings | None = None
        self._pipeline: MedicalRAGPipeline | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def _ensure_loaded(self) -> MedicalRAGPipeline:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is None:
                self._settings = load_settings(self.config_path)
                self._pipeline = MedicalRAGPipeline(self._settings)
        return self._pipeline

    def ask(self, image_path: str, question: str) -> Dict[str, Any]:
        pipeline = self._ensure_loaded()
        result = pipeline.ask(image_path=image_path, question=question)
        evidence = [
            {
                "doc_id": x.doc_id,
                "source": x.source,
                "chunk_index": x.chunk_index,
                "score": x.score,
                "content": x.content,
            }
            for x in result.evidence
        ]
        return {"answer": result.answer, "evidence": evidence}


def _guess_suffix(upload: UploadFile) -> str:
    name_suffix = Path(upload.filename or "").suffix
    if name_suffix:
        return name_suffix
    mime = upload.content_type or "image/png"
    ext = mimetypes.guess_extension(mime)
    return ext or ".png"


def create_app(config_path: str | None = None) -> FastAPI:
    cfg_path = config_path or os.getenv("MEDICAL_RAG_CONFIG", str(DEFAULT_CONFIG_PATH))
    runtime = _PipelineRuntime(cfg_path)

    app = FastAPI(
        title="Medical-RAG API",
        description="FastAPI service for medical image + RAG question answering",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "config_path": cfg_path,
            "pipeline_ready": runtime.ready,
        }

    @app.post("/ask")
    async def ask(
        question: str = Form(..., description="Natural language medical question"),
        image: UploadFile = File(..., description="Medical image file"),
    ) -> Dict[str, Any]:
        q = question.strip()
        if not q:
            raise HTTPException(status_code=400, detail="question is empty")
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"unsupported image content_type={image.content_type}")

        suffix = _guess_suffix(image)
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="medical_rag_", suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                content = await image.read()
                tmp.write(content)

            output = runtime.ask(image_path=tmp_path, question=q)
            return {
                "answer": output["answer"],
                "evidence": output["evidence"],
                "meta": {"question": q, "config_path": cfg_path},
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            try:
                await image.close()
            except Exception:
                pass

    return app


app = create_app()
