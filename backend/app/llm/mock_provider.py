"""Deterministic, key-free provider: extractive answers built from the retrieved context.

It cannot *write* an answer, but it picks the most relevant sentences/lines from the
retrieved passages (semantically when an embedding model is available, lexically
otherwise) and quotes them with citation tags, so the whole pipeline stays demonstrable.
"""

import asyncio
import re
from dataclasses import dataclass

import numpy as np

from app.llm.base import NOT_FOUND_PHRASE, ChatTurn, ContextChunk
from app.llm.mock_extraction import extract_terms_heuristic
from app.schemas import TermsDraft
from app.services.embeddings import EmbeddingService
from app.services.intent import is_overview_question
from app.services.retriever import tokenize

_SENTENCE_END = re.compile(r"[.;!?]['\")\]]*\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+")
_FOLLOW_UP = re.compile(r"^(and|also|but|then|so|why|what about|how about)\b|^(it|that|they|those|these)\b", re.I)
_GREETING = re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay)\b[\s!.?]*$", re.I)

_MAX_ANSWER_UNITS = 3
_MAX_OVERVIEW_UNITS = 6
_MIN_COSINE = 0.25  # sentence-level; passage-level gating already happened in retrieval
_RELATIVE_CUTOFF = 0.7  # keep units scoring at least this fraction of the best one
_HEADER_FOLLOWERS = 3  # lines pulled in under a matched section header
_LABEL_STYLE_MAX_SENTENCE_RATIO = 0.25


# Words that occur on nearly every line of a contract, so matching on them says nothing.
_GENERIC = frozenset({"contract", "contracts", "agreement", "agreements", "document", "documents"})


def _stems(text: str) -> set[str]:
    """Crude 6-char prefix stems so 'terminate' / 'termination' overlap."""
    return {t[:6] for t in tokenize(text) if t not in _GENERIC}


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_header(text: str) -> bool:
    """A short, unpunctuated label such as "Top skills covered" or "Definitions:"."""
    words = text.split()
    if text.rstrip().endswith(":"):
        return True
    return len(words) <= 4 and not re.search(r"\d", text) and not _SENTENCE_END.search(text)


@dataclass
class _Unit:
    tag: str
    text: str
    order: int  # position across the whole context, for readable output


def _units(context: list[ContextChunk]) -> list[_Unit]:
    """Split passages into quotable units, deduplicated across overlapping chunks.

    Prose passages are split into sentences (physical PDF lines wrap mid-sentence, so
    lines are re-joined first). Label-style passages (certificates, forms, cover pages:
    few lines end in punctuation) are split per line instead; otherwise the whole page
    would collapse into a single "sentence".
    """
    out: list[_Unit] = []
    seen: set[str] = set()

    def add(tag: str, text: str, min_len: int) -> None:
        text = " ".join(text.split())
        key = _norm(text)
        if len(text) < min_len or key in seen or not re.search(r"[A-Za-z0-9]", text):
            return
        seen.add(key)
        out.append(_Unit(tag, text, len(out)))

    for chunk in context:
        lines = [ln.strip() for ln in chunk.text.split("\n") if ln.strip()]
        if not lines:
            continue
        ends = sum(1 for ln in lines if _SENTENCE_END.search(ln))
        if ends / len(lines) < _LABEL_STYLE_MAX_SENTENCE_RATIO:
            for ln in lines:
                add(chunk.tag, ln, 3)
        else:
            for sent in _SENTENCE_SPLIT.split(" ".join(lines)):
                add(chunk.tag, sent, 25)
    return out


class MockProvider:
    name = "mock"

    def __init__(self, embeddings: EmbeddingService | None = None) -> None:
        self._embeddings = embeddings

    async def condense_question(self, question: str, history: list[ChatTurn]) -> str:
        q = question.strip()
        prior = [t.content for t in history if t.role == "user"]
        if not prior or _GREETING.match(q):
            return question
        # Only rewrite genuinely elliptical follow-ups; a standalone question must not
        # inherit the previous question's topic.
        if len(q.split()) <= 3 or _FOLLOW_UP.search(q):
            return f"{prior[-1]} {question}"
        return question

    async def _similarities(self, question: str, texts: list[str]) -> np.ndarray | None:
        emb = self._embeddings
        if emb is None or emb.status in ("disabled", "unavailable"):
            return None

        def run() -> np.ndarray | None:
            vecs = emb.encode(texts)
            q = emb.encode_query(question)
            if vecs is None or q is None:
                return None
            return vecs @ q

        return await asyncio.to_thread(run)

    async def generate(
        self, question: str, contract_name: str, context: list[ContextChunk]
    ) -> str:
        units = _units(context)
        if not units:
            return NOT_FOUND_PHRASE

        if is_overview_question(question):
            head = units[:_MAX_OVERVIEW_UNITS]
            lines = [f"- {u.text} [{u.tag}]" for u in head]
            return "This document appears to cover:\n" + "\n".join(lines)

        q_stems = _stems(question)
        sims = await self._similarities(question, [u.text for u in units])

        scored: list[tuple[float, _Unit, int]] = []
        for i, unit in enumerate(units):
            overlap = len(q_stems & _stems(unit.text))
            cos = float(sims[i]) if sims is not None else 0.0
            if overlap == 0 and cos < _MIN_COSINE:
                continue
            scored.append((cos + 0.15 * overlap - 0.001 * unit.order, unit, overlap))
        if not scored:
            return NOT_FOUND_PHRASE

        scored.sort(key=lambda s: s[0], reverse=True)
        best = scored[0][0]
        picked = [s for s in scored[:_MAX_ANSWER_UNITS] if s[0] >= best * _RELATIVE_CUTOFF]
        chosen: dict[int, tuple[float, _Unit]] = {s[1].order: (s[0], s[1]) for s in picked}

        # A matched section header ("Top skills covered") is only a label: the answer is
        # the lines beneath it. Followers sit directly after their header.
        for score, unit, overlap in picked:
            if overlap and _is_header(unit.text):
                for follower in units[unit.order + 1 : unit.order + 1 + _HEADER_FOLLOWERS]:
                    chosen.setdefault(follower.order, (score, follower))

        # Most relevant first; document order within equal relevance.
        ordered = sorted(chosen.values(), key=lambda su: (-su[0], su[1].order))
        lines = [f"- {u.text} [{u.tag}]" for _, u in ordered]
        return "Based on the contract:\n" + "\n".join(lines)

    async def compare(self, question: str, answers: dict[str, str]) -> str | None:
        if len(answers) < 2:
            return None
        found = [n for n, a in answers.items() if not a.startswith(NOT_FOUND_PHRASE)]
        missing = [n for n in answers if n not in found]
        parts = [f"Relevant content was found in {len(found)} of {len(answers)} contracts."]
        if missing:
            parts.append("No supporting content in: " + ", ".join(missing) + ".")
        return " ".join(parts)

    async def extract_terms(self, contract_name: str, context: list[ContextChunk]) -> TermsDraft:
        return extract_terms_heuristic(context)
