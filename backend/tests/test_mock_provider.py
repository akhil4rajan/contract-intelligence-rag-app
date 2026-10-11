import asyncio

from app.llm.base import NOT_FOUND_PHRASE, ChatTurn, ContextChunk
from app.llm.mock_provider import MockProvider
from app.services.embeddings import EmbeddingService
from app.services.intent import is_overview_question
from tests.test_langchain_integration import SynonymEmbeddings

CERT = "\n".join(
    [
        "Hands-On AI: RAG using LlamaIndex",
        "Course completed by Akhil Rajan",
        "Dec 17, 2024 at 09:11AM UTC  6 hours 25 minutes",
        "Top skills covered",
        "Retrieval-Augmented Generation (RAG)",
        "Head of Global Content, Learning",
        "Certificate ID: 605e5ebad47b7d63c5673be79e067f2c",
    ]
)
PROSE = (
    "3. TERMINATION\n"
    "Either party may terminate this Agreement for convenience by giving sixty (60) days\n"
    "prior written notice to the other party.\n"
    "Either party may terminate immediately if the other party materially breaches this\n"
    "Agreement and fails to cure within thirty (30) days.\n"
    "4. GOVERNING LAW\n"
    "This Agreement is governed by the laws of the State of New York."
)


def _ctx(text: str) -> list[ContextChunk]:
    return [ContextChunk("C1", 1, text, 1, 1, 1, len(text.split("\n")))]


def _gen(provider: MockProvider, question: str, text: str) -> str:
    return asyncio.run(provider.generate(question, "doc.pdf", _ctx(text)))


def test_label_style_page_is_not_returned_as_one_blob():
    out = _gen(MockProvider(), "who completed the course?", CERT)
    assert "Course completed by Akhil Rajan" in out
    assert "Certificate ID" not in out  # not a dump of the whole page
    assert "Top skills covered" not in out


def test_different_questions_give_different_answers():
    mock = MockProvider()
    who = _gen(mock, "who completed the course?", CERT)
    when = _gen(mock, "what is the duration in hours and minutes?", CERT)
    cert = _gen(mock, "what is the certificate id?", CERT)
    assert len({who, when, cert}) == 3
    assert "6 hours 25 minutes" in when
    assert "605e5ebad47b7d63" in cert


def test_matched_section_header_pulls_in_the_lines_beneath_it():
    page = "\n".join(
        [
            "Hands-On AI: RAG using LlamaIndex",
            "Course completed by Akhil Rajan",
            "Dec 17, 2024 at 09:11AM UTC  6 hours 25 minutes",
            "Top skills covered",
            "LLaMA",
            "Retrieval-Augmented Generation (RAG)",
            "Artificial Intelligence (AI)",
            "Head of Global Content, Learning",
        ]
    )
    out = _gen(MockProvider(), "What skills are covered?", page)
    assert "Top skills covered" in out
    assert "LLaMA" in out and "Retrieval-Augmented Generation (RAG)" in out
    assert "Artificial Intelligence (AI)" in out
    assert "Course completed" not in out and "Head of Global" not in out


def test_weak_extra_matches_are_not_padded_into_the_answer():
    out = _gen(MockProvider(), "who completed the course?", CERT)
    assert out.count("\n- ") == 1  # only the line that answers it


def test_prose_is_answered_with_the_matching_sentence_only():
    out = _gen(MockProvider(), "what is the termination notice period?", PROSE)
    assert "sixty (60) days prior written notice" in out  # wrapped lines re-joined
    assert "New York" not in out


def test_overview_question_summarises_the_opening():
    out = _gen(MockProvider(), "what is this document about ?", CERT)
    assert out.startswith("This document appears to cover:")
    assert "Hands-On AI: RAG using LlamaIndex" in out and "Course completed by" in out
    assert "[C1]" in out


def test_unrelated_question_is_not_found():
    assert _gen(MockProvider(), "tell me about quarterly revenue forecasts", CERT) == NOT_FOUND_PHRASE


def test_semantic_ranking_when_embeddings_available():
    # "cancel" shares no word with "terminate": only the embedding path can match it.
    mock = MockProvider(EmbeddingService("stub", embeddings=SynonymEmbeddings()))
    out = _gen(mock, "can the vendor cancel?", "The vendor may terminate on sixty days notice.\n"
                                               "Invoices are due for payment within forty five days.")
    assert "terminate" in out and "Invoices" not in out


def test_standalone_question_does_not_inherit_previous_topic():
    mock = MockProvider()
    history = [ChatTurn("user", "Who are the parties?"), ChatTurn("assistant", "Acme and Globex")]
    run = lambda q: asyncio.run(mock.condense_question(q, history))  # noqa: E731

    assert run("what is this document about ?") == "what is this document about ?"
    assert run("what are the payment terms?") == "what are the payment terms?"
    assert run("hi") == "hi"
    assert run("and for renewal?") == "Who are the parties? and for renewal?"  # real follow-up
    assert run("why?") == "Who are the parties? why?"


def test_overview_intent_detection():
    yes = [
        "what is this document about ?", "What is this document about?", "summarize this",
        "Give me an overview", "what is this contract", "what does this agreement say?",
        "tell me about this document", "what's this about",
    ]
    no = [
        "what is the contract value?", "what is the termination notice period?",
        "who are the parties to this agreement?", "hi", "what is the contract term",
    ]
    assert all(is_overview_question(q) for q in yes), [q for q in yes if not is_overview_question(q)]
    assert not any(is_overview_question(q) for q in no), [q for q in no if is_overview_question(q)]
