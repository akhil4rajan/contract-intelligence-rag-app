"""Per-contract hybrid retrieval: BM25 + dense cosine, fused with Reciprocal Rank Fusion.

Chunks are exposed as LangChain ``Document`` objects whose metadata carries the page /
line / bbox provenance, and the retriever is available as a LangChain ``BaseRetriever``
(see :class:`ContractRetriever`) so it composes with the rest of the LangChain stack.
"""

import math
import re
import threading
from collections import Counter

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Chunk
from app.services.embeddings import EmbeddingService

RRF_K = 60
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "shall may any such which who whom what when where how do does did i you we they their our your".split()
)
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 with a Lucene-style IDF, ``log(1 + (N - n + .5) / (n + .5))``.

    ``rank_bm25.BM25Okapi`` (which LangChain's ``BM25Retriever`` wraps) yields zero or
    negative IDF on tiny corpora (a one-page contract is a single chunk), silently
    scoring every match <= 0. This variant is always non-negative.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.doc_len = np.array([len(d) for d in corpus], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf = [Counter(d) for d in corpus]
        df = Counter(t for d in corpus for t in set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.tf], dtype=np.float32)
            scores += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


def chunk_to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.id,
            "contract_id": chunk.contract_id,
            "chunk_index": chunk.chunk_index,
            "start_page": chunk.start_page,
            "start_line": chunk.start_line,
            "end_page": chunk.end_page,
            "end_line": chunk.end_line,
            "line_refs": chunk.line_refs,
        },
    )


class _ContractIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.docs = [chunk_to_document(c) for c in chunks]
        corpus = [tokenize(c.text) or ["_"] for c in chunks]
        self.bm25 = BM25(corpus) if chunks else None
        have_vecs = bool(chunks) and all(c.embedding for c in chunks)
        self.matrix = (
            np.vstack([np.frombuffer(c.embedding, dtype=np.float32) for c in chunks])
            if have_vecs
            else None
        )


class Retriever:
    def __init__(
        self, embeddings: EmbeddingService, candidates: int = 20, min_similarity: float = 0.15
    ) -> None:
        self.embeddings = embeddings
        self.candidates = candidates
        # Dense search always returns *something*; below this cosine a chunk is noise.
        # (all-MiniLM-L6-v2: relevant legal queries score >= ~0.29, unrelated <= ~0.08.)
        self.min_similarity = min_similarity
        self._cache: dict[str, _ContractIndex] = {}
        self._lock = threading.Lock()

    def invalidate(self, contract_id: str) -> None:
        with self._lock:
            self._cache.pop(contract_id, None)

    def _index(self, db: Session, contract_id: str) -> _ContractIndex:
        with self._lock:
            idx = self._cache.get(contract_id)
        if idx is not None:
            return idx
        chunks = list(
            db.scalars(
                select(Chunk).where(Chunk.contract_id == contract_id).order_by(Chunk.chunk_index)
            )
        )
        idx = _ContractIndex(chunks)
        with self._lock:
            self._cache[contract_id] = idx
        return idx

    def retrieve(self, db: Session, contract_id: str, query: str, top_k: int = 5) -> list[Document]:
        """Top-k chunks (RRF order) as Documents.

        Metadata gains ``score`` (0..1 relevance: cosine similarity when embeddings are
        available, else normalised BM25), ``bm25_rank`` and ``dense_rank``. Chunks that
        match neither lexically nor above the similarity floor are dropped, so an
        unrelated question yields no context rather than the "least bad" chunks.
        """
        idx = self._index(db, contract_id)
        if not idx.docs:
            return []

        n = min(self.candidates, len(idx.docs))
        bm25_rank: dict[int, int] = {}
        bm25_norm: dict[int, float] = {}
        if idx.bm25 is not None:
            q_tokens = tokenize(query)
            if q_tokens:
                scores = idx.bm25.get_scores(q_tokens)
                # Upper bound as tf -> infinity, to turn raw BM25 into a 0..1 relevance.
                ceiling = sum(
                    idx.bm25.idf[t] * (idx.bm25.k1 + 1) for t in set(q_tokens) if t in idx.bm25.idf
                )
                for rank, i in enumerate(np.argsort(-scores)[:n]):
                    if scores[i] > 0:
                        bm25_rank[int(i)] = rank
                        bm25_norm[int(i)] = float(scores[i]) / ceiling if ceiling else 0.0

        dense_rank: dict[int, int] = {}
        sims: np.ndarray | None = None
        if idx.matrix is not None:
            q_vec = self.embeddings.encode_query(query)
            if q_vec is not None:
                sims = idx.matrix @ q_vec
                for rank, i in enumerate(np.argsort(-sims)[:n]):
                    if sims[i] >= self.min_similarity:
                        dense_rank[int(i)] = rank

        fused: dict[int, float] = {}
        for ranks in (bm25_rank, dense_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[Document] = []
        for i, _ in ordered:
            relevance = float(sims[i]) if sims is not None else bm25_norm.get(i, 0.0)
            out.append(
                Document(
                    page_content=idx.docs[i].page_content,
                    metadata={
                        **idx.docs[i].metadata,
                        "score": round(max(0.0, min(1.0, relevance)), 4),
                        "bm25_rank": bm25_rank.get(i),
                        "dense_rank": dense_rank.get(i),
                    },
                )
            )
        return out

    def lead(self, db: Session, contract_id: str, k: int = 2) -> list[Document]:
        """The first ``k`` chunks in document order, for overview/summary questions.

        There is no query-relevance here, so ``score`` is None rather than a made-up value.
        """
        idx = self._index(db, contract_id)
        return [
            Document(
                page_content=d.page_content,
                metadata={**d.metadata, "score": None, "bm25_rank": None, "dense_rank": None},
            )
            for d in idx.docs[:k]
        ]

    def as_langchain(
        self, contract_id: str, top_k: int = 5, lead: bool = False
    ) -> "ContractRetriever":
        return ContractRetriever(
            retriever=self, contract_id=contract_id, top_k=top_k, lead=lead
        )


class ContractRetriever(BaseRetriever):
    """LangChain retriever scoped to one contract.

    Opens its own short-lived DB session per call, so concurrent ``ainvoke`` calls
    (LangChain runs sync retrievers in an executor) never share a session.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Retriever
    contract_id: str
    top_k: int = 5
    lead: bool = False  # return the document's opening chunks instead of searching

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        db = SessionLocal()
        try:
            if self.lead:
                return self.retriever.lead(db, self.contract_id, self.top_k)
            return self.retriever.retrieve(db, self.contract_id, query, self.top_k)
        finally:
            db.close()
