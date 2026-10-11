"""Exercises the LangChain-backed pieces offline (fake chat model, stub Embeddings)."""

import asyncio

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.llm.base import NOT_FOUND_PHRASE, ChatTurn, ContextChunk
from app.llm.langchain_provider import LangChainProvider
from app.schemas import QueryRequest
from app.services.embeddings import EmbeddingService
from app.services.retriever import Retriever

_CTX = [
    ContextChunk(
        tag="C1", chunk_id=1, text="Either party may terminate on sixty (60) days notice.",
        page_start=2, line_start=1, page_end=2, line_end=2,
    )
]


def _run(coro):
    return asyncio.run(coro)


# ---- LangChainProvider ----------------------------------------------------------


def test_provider_generate_runs_prompt_model_parser_chain():
    provider = LangChainProvider(FakeListChatModel(responses=["  Sixty days [C1].  "]), "fake")
    answer = _run(provider.generate("What is the notice period?", "msa.pdf", _CTX))
    assert answer == "Sixty days [C1]."


def test_provider_short_circuits_without_context():
    provider = LangChainProvider(FakeListChatModel(responses=["should not be used"]), "fake")
    assert _run(provider.generate("q", "msa.pdf", [])) == NOT_FOUND_PHRASE


def test_provider_condense_and_compare_use_bound_model():
    provider = LangChainProvider(
        FakeListChatModel(responses=["standalone query", "They differ: 60 vs 30 days."]), "fake"
    )
    history = [ChatTurn("user", "What is the notice period?"), ChatTurn("assistant", "60 days")]
    assert _run(provider.condense_question("and for renewal?", history)) == "standalone query"
    assert _run(provider.compare("q", {"a.pdf": "60 days", "b.pdf": "30 days"})) == (
        "They differ: 60 vs 30 days."
    )
    assert _run(provider.compare("q", {"a.pdf": "only one"})) is None


def test_condense_without_history_skips_the_model():
    provider = LangChainProvider(FakeListChatModel(responses=["unused"]), "fake")
    assert _run(provider.condense_question("plain question", [])) == "plain question"


def test_provider_errors_in_condense_fall_back_to_raw_question():
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("api down")

    provider = LangChainProvider(Boom(responses=["x"]), "fake")
    history = [ChatTurn("user", "earlier")]
    assert _run(provider.condense_question("follow up", history)) == "follow up"


# ---- Citation tag validation through the real RagService ------------------------


def test_rag_drops_hallucinated_citation_tags(client, pdf_a):
    from app.db import SessionLocal
    from app.deps import get_retriever
    from app.services.rag import RagService
    from tests.test_api import _upload

    res = _upload(client, **{"rag_tags.pdf": pdf_a + b"\n%rag-tags-variant"})
    contract_id = res.json()["results"][0]["contract"]["id"]

    fake = FakeListChatModel(responses=["Sixty (60) days notice [C1][C9]. Also [C1, C7]."])
    rag = RagService(get_retriever(), LangChainProvider(fake, "fake"))
    db = SessionLocal()
    try:
        out = _run(
            rag.query(
                db,
                QueryRequest(question="What is the termination notice?", contract_ids=[contract_id]),
            )
        )
    finally:
        db.close()

    result = out.results[0]
    assert [c.tag for c in result.citations] == ["C1"]  # C7 / C9 are not in the context
    assert "[C9]" not in result.answer and "[C7]" not in result.answer
    assert out.provider == "fake"


# ---- Dense path + RRF with an injected LangChain Embeddings ---------------------


class SynonymEmbeddings(Embeddings):
    """Tiny concept-space embedder: synonyms share a dimension (stands in for a real model)."""

    CONCEPTS = {
        "terminate": 0, "termination": 0, "cancel": 0, "end": 0, "cancellation": 0,
        "pay": 1, "payment": 1, "invoice": 1, "fees": 1,
        "law": 2, "governing": 2, "jurisdiction": 2,
    }

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(3, dtype=np.float32)
        for tok in text.lower().replace(".", " ").split():
            if tok in self.CONCEPTS:  # unknown words carry no signal -> orthogonal
                v[self.CONCEPTS[tok]] += 1.0
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture()
def dense_contract():
    from app.db import SessionLocal
    from app.models import Chunk, Contract

    emb = EmbeddingService("stub", embeddings=SynonymEmbeddings())
    texts = [
        "The vendor may terminate this agreement upon sixty days notice.",
        "Invoices are due for payment within forty five days of receipt.",
        "This agreement is governed by the law of New York jurisdiction.",
    ]
    vecs = emb.encode(texts)
    assert vecs is not None

    db = SessionLocal()
    contract = Contract(
        filename="dense.pdf", stored_path="n/a", size_bytes=1, page_count=1,
        sha256="dense-test", status="ready",
    )
    db.add(contract)
    db.flush()
    for i, (t, v) in enumerate(zip(texts, vecs)):
        db.add(
            Chunk(
                contract_id=contract.id, chunk_index=i, text=t, token_count=12,
                start_page=1, start_line=i + 1, end_page=1, end_line=i + 1,
                char_start=0, char_end=len(t),
                line_refs=[{"page": 1, "line_start": i + 1, "line_end": i + 1,
                            "bboxes": [[0.1, 0.1, 0.9, 0.12]]}],
                embedding=np.asarray(v, dtype=np.float32).tobytes(),
            )
        )
    db.commit()
    yield db, contract.id, emb
    db.close()


def test_dense_retrieval_finds_synonym_match_that_bm25_misses(dense_contract):
    db, contract_id, emb = dense_contract
    retriever = Retriever(emb, candidates=10)

    # "cancel early" shares no token with any clause ("terminate ..."): BM25 finds nothing.
    docs = retriever.retrieve(db, contract_id, "cancel early", top_k=3)
    assert docs, "dense path should surface a match"
    top = docs[0]
    assert "terminate" in top.page_content
    assert top.metadata["dense_rank"] == 0
    assert top.metadata["bm25_rank"] is None
    assert 0 < top.metadata["score"] <= 1


def test_hybrid_scores_agreement_higher_and_langchain_wrapper_matches(dense_contract):
    db, contract_id, emb = dense_contract
    retriever = Retriever(emb, candidates=10)

    both = retriever.retrieve(db, contract_id, "terminate this agreement", top_k=3)[0]
    assert both.metadata["bm25_rank"] == 0 and both.metadata["dense_rank"] == 0
    assert both.metadata["score"] == pytest.approx(1.0)

    # LangChain Runnable interface returns the same Documents.
    via_lc = retriever.as_langchain(contract_id, top_k=3).invoke("terminate this agreement")
    assert via_lc[0].metadata["chunk_id"] == both.metadata["chunk_id"]


def test_unrelated_query_yields_no_context(dense_contract):
    db, contract_id, emb = dense_contract
    retriever = Retriever(emb, candidates=10)
    # No lexical overlap and orthogonal in embedding space: nothing should be returned,
    # so the LLM is never asked to answer from irrelevant text.
    assert retriever.retrieve(db, contract_id, "weather forecast in paris", top_k=3) == []


def test_dense_score_is_cosine_not_rank_normalised(dense_contract):
    db, contract_id, emb = dense_contract
    retriever = Retriever(emb, candidates=10)
    docs = retriever.retrieve(db, contract_id, "cancel early", top_k=3)
    # Only the concept-matching chunk clears the floor; its score is the true cosine (1.0
    # for an identical concept vector), and weaker chunks are not padded in.
    assert [d.metadata["chunk_index"] for d in docs] == [0]
    assert docs[0].metadata["score"] == pytest.approx(1.0)


def test_bm25_only_when_embeddings_disabled(dense_contract):
    db, contract_id, _ = dense_contract
    retriever = Retriever(EmbeddingService("stub", enabled=False), candidates=10)
    docs = retriever.retrieve(db, contract_id, "payment invoices", top_k=2)
    assert docs and "Invoices" in docs[0].page_content
    assert docs[0].metadata["dense_rank"] is None
    assert 0 < docs[0].metadata["score"] <= 1  # normalised BM25
