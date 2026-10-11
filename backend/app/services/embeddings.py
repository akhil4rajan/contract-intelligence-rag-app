"""Lazy, failure-tolerant wrapper around a LangChain ``Embeddings`` implementation.

Defaults to ``HuggingFaceEmbeddings`` (sentence-transformers, runs locally). Any other
LangChain ``Embeddings`` (OpenAI, Cohere, ...) can be injected. If the model cannot be
loaded (e.g. offline first run) the service marks itself unavailable and retrieval
falls back to BM25-only.
"""

import logging
import threading

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self, model_name: str, enabled: bool = True, embeddings: Embeddings | None = None
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model: Embeddings | None = embeddings
        self._failed = False
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if self._failed:
            return "unavailable"
        return "loaded" if self._model is not None else "not_loaded"

    def _load(self) -> Embeddings | None:
        if not self.enabled or self._failed:
            return None
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    self._model = HuggingFaceEmbeddings(
                        model_name=self.model_name,
                        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
                    )
                    logger.info("Loaded embedding model %s", self.model_name)
                except Exception:  # noqa: BLE001 - any load failure degrades to BM25
                    logger.exception("Embedding model unavailable; falling back to BM25 only")
                    self._failed = True
        return self._model

    def warm_up(self) -> None:
        self._load()

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """Embed documents -> float32 matrix, or None if embeddings are unavailable."""
        model = self._load()
        if model is None or not texts:
            return None
        try:
            return np.asarray(model.embed_documents(texts), dtype=np.float32)
        except Exception:  # noqa: BLE001
            logger.exception("Embedding failed")
            return None

    def encode_query(self, text: str) -> np.ndarray | None:
        model = self._load()
        if model is None:
            return None
        try:
            return np.asarray(model.embed_query(text), dtype=np.float32)
        except Exception:  # noqa: BLE001
            logger.exception("Query embedding failed")
            return None


_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        s = get_settings()
        _service = EmbeddingService(s.embedding_model, s.embeddings_enabled)
    return _service
