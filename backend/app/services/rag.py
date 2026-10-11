"""RAG orchestration: retrieve per contract -> generate per contract (parallel) -> ground citations."""

import asyncio
import logging
import re
from datetime import datetime, timezone
from itertools import groupby

from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import ContractNotReady, NotFoundError
from app.llm.base import NOT_FOUND_PHRASE, ChatTurn, ContextChunk, LLMProvider
from app.models import Contract, Conversation, Message
from app.schemas import (
    Citation,
    ContractAnswer,
    Highlight,
    LineRange,
    QueryRequest,
    QueryResponse,
)
from app.services.intent import is_overview_question
from app.services.retriever import Retriever, tokenize

logger = logging.getLogger(__name__)

_TAG_GROUP = re.compile(r"\[\s*(C\d+(?:\s*[,;]\s*C\d+)+)\s*\]")  # [C1, C2] -> [C1][C2]
_TAG = re.compile(r"\[(C\d+)\]")
_HISTORY_TURNS = 6
_SNIPPET_CHARS = 500
_OVERVIEW_CHUNKS = 2


def _normalise_tags(answer: str) -> str:
    return _TAG_GROUP.sub(
        lambda m: "".join(f"[{t.strip()}]" for t in re.split(r"[,;]", m.group(1))), answer
    )


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET_CHARS else text[: _SNIPPET_CHARS - 1].rstrip() + "…"


_MAX_WINDOW_LINES = 8
_SUPPORT_THRESHOLD = 0.5  # fraction of a line's words that must appear in the answer

# (page, line_number, bbox)
_LineKey = tuple[int, int, list[float]]


def _flatten_lines(text: str, line_refs: list[dict]) -> list[tuple[str, _LineKey]] | None:
    """Pair each physical line of a chunk with its (page, line, bbox); None if misaligned."""
    texts = text.split("\n")
    keys: list[_LineKey] = [
        (ref["page"], n, ref["bboxes"][i])
        for ref in line_refs
        for i, n in enumerate(range(ref["line_start"], ref["line_end"] + 1))
    ]
    return list(zip(texts, keys)) if len(texts) == len(keys) else None


def _focus_window(
    text: str, line_refs: list[dict], terms: set[str]
) -> list[tuple[str, _LineKey]] | None:
    """The contiguous run of chunk lines best supported by the answer.

    A line is *supported* when most of its words appear in the answer/question terms
    (a verbatim quote is 100%). Of the contiguous runs of supported lines the one with the
    most support wins, capped at ``_MAX_WINDOW_LINES``. Returns None when no line is
    supported, so the caller cites the whole chunk instead.
    """
    lines = _flatten_lines(text, line_refs)
    if not lines or not terms:
        return None
    coverage: list[float] = []
    for line_text, _ in lines:
        toks = set(tokenize(line_text))
        coverage.append(len(toks & terms) / len(toks) if toks else 0.0)

    best: tuple[float, int, int] | None = None  # (support, lo, hi)
    i = 0
    while i < len(lines):
        if coverage[i] < _SUPPORT_THRESHOLD:
            i += 1
            continue
        j = i
        while j + 1 < len(lines) and coverage[j + 1] >= _SUPPORT_THRESHOLD:
            j += 1
        for lo in range(i, max(i, j - _MAX_WINDOW_LINES + 1) + 1):
            hi = min(j, lo + _MAX_WINDOW_LINES - 1)
            support = sum(coverage[lo : hi + 1])
            if best is None or support > best[0]:
                best = (support, lo, hi)
        i = j + 1
    if best is None:
        return None
    return lines[best[1] : best[2] + 1]


class RagService:
    def __init__(self, retriever: Retriever, provider: LLMProvider) -> None:
        self.retriever = retriever
        self.provider = provider

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _load_contracts(db: Session, ids: list[str]) -> list[Contract]:
        rows = {c.id: c for c in db.scalars(select(Contract).where(Contract.id.in_(ids)))}
        missing = [i for i in ids if i not in rows]
        if missing:
            raise NotFoundError("Unknown contract id(s)", details={"contract_ids": missing})
        not_ready = [i for i in ids if rows[i].status != "ready"]
        if not_ready:
            raise ContractNotReady(
                "Some contracts are not ready for querying yet", details={"contract_ids": not_ready}
            )
        return [rows[i] for i in ids]

    @staticmethod
    def _conversation(db: Session, conversation_id: str | None, question: str) -> Conversation:
        if conversation_id:
            conv = db.get(Conversation, conversation_id)
            if conv is None:
                raise NotFoundError(f"Conversation {conversation_id} not found")
            return conv
        title = question if len(question) <= 60 else question[:57].rstrip() + "…"
        conv = Conversation(title=title)
        db.add(conv)
        db.flush()
        return conv

    @staticmethod
    def _history(conv: Conversation) -> list[ChatTurn]:
        return [ChatTurn(m.role, m.content) for m in conv.messages[-_HISTORY_TURNS:]]

    @staticmethod
    def _build_citation(
        contract: Contract,
        tag: str,
        doc: Document,
        cited: bool,
        focus_terms: set[str],
        focused: bool | None = None,
    ) -> Citation:
        """Cite the tightest line window inside the chunk that supports the answer.

        Falls back to the whole chunk when no line overlaps the focus terms. ``focused``
        defaults to "the chunk has a relevance score"; pass True to narrow a chunk that
        has none (structured extraction quotes the exact lines it read).
        """
        m = doc.metadata
        # Overview answers (no relevance score) summarise the whole passage: cite all of it.
        if focused is None:
            focused = m.get("score") is not None
        window = _focus_window(doc.page_content, m["line_refs"], focus_terms) if focused else None
        if window is None:
            snippet = doc.page_content
            page_start, line_start = m["start_page"], m["start_line"]
            page_end, line_end = m["end_page"], m["end_line"]
            refs = [
                (ref["page"], ref["line_start"], ref["line_end"], ref["bboxes"])
                for ref in m["line_refs"]
            ]
        else:
            snippet = "\n".join(t for t, _ in window)
            first, last = window[0][1], window[-1][1]
            page_start, line_start = first[0], first[1]
            page_end, line_end = last[0], last[1]
            refs = []
            for page, group in groupby((w[1] for w in window), key=lambda x: x[0]):
                g = list(group)
                refs.append((page, g[0][1], g[-1][1], [x[2] for x in g]))

        return Citation(
            tag=tag,
            chunk_id=m["chunk_id"],
            contract_id=contract.id,
            contract_name=contract.filename,
            page_start=page_start,
            page_end=page_end,
            line_start=line_start,
            line_end=line_end,
            line_ranges=[LineRange(page=p, line_start=a, line_end=b) for p, a, b, _ in refs],
            snippet=_snippet(snippet),
            score=m["score"],
            cited=cited,
            highlights=[Highlight(page=p, bboxes=bb) for p, _, _, bb in refs],
        )

    # ---- main entrypoint -----------------------------------------------------

    async def query(self, db: Session, req: QueryRequest) -> QueryResponse:
        contracts = await run_in_threadpool(self._load_contracts, db, req.contract_ids)
        conv = await run_in_threadpool(self._conversation, db, req.conversation_id, req.question)
        history = self._history(conv)

        overview = is_overview_question(req.question)
        # Overview questions are answered from the document's opening, not by similarity.
        search_query = (
            req.question
            if overview
            else await self.provider.condense_question(req.question, history)
        )
        k = min(req.top_k, _OVERVIEW_CHUNKS) if overview else req.top_k

        # One LangChain retriever per contract, invoked concurrently (each in its own
        # executor thread with its own DB session).
        hit_lists = await asyncio.gather(
            *(
                self.retriever.as_langchain(c.id, k, lead=overview).ainvoke(search_query)
                for c in contracts
            )
        )
        retrieved: dict[str, list[Document]] = {
            c.id: hits for c, hits in zip(contracts, hit_lists)
        }

        async def answer_one(contract: Contract) -> ContractAnswer:
            hits = retrieved[contract.id]
            if not hits:
                return ContractAnswer(
                    contract_id=contract.id,
                    contract_name=contract.filename,
                    answer=NOT_FOUND_PHRASE,
                    not_found=True,
                )
            tagged = {f"C{i}": h for i, h in enumerate(hits, start=1)}
            context = [
                ContextChunk(
                    tag=tag,
                    chunk_id=h.metadata["chunk_id"],
                    text=h.page_content,
                    page_start=h.metadata["start_page"],
                    line_start=h.metadata["start_line"],
                    page_end=h.metadata["end_page"],
                    line_end=h.metadata["end_line"],
                )
                for tag, h in tagged.items()
            ]
            try:
                raw = await self.provider.generate(req.question, contract.filename, context)
            except Exception as exc:  # noqa: BLE001 - isolate per-contract failures
                logger.exception("LLM generation failed for %s", contract.id)
                return ContractAnswer(
                    contract_id=contract.id,
                    contract_name=contract.filename,
                    answer="",
                    error=f"Answer generation failed: {type(exc).__name__}",
                )

            answer = _normalise_tags(raw.strip())
            not_found = answer.startswith(NOT_FOUND_PHRASE)
            if not_found:
                return ContractAnswer(
                    contract_id=contract.id,
                    contract_name=contract.filename,
                    answer=answer,
                    not_found=True,
                )

            # Keep only tags that exist in the supplied context; drop hallucinated ones.
            used: list[str] = []
            for tag in _TAG.findall(answer):
                if tag in tagged and tag not in used:
                    used.append(tag)
            answer = _TAG.sub(lambda m: m.group(0) if m.group(1) in tagged else "", answer)

            q_terms = set(tokenize(req.question))
            if used:
                citations = []
                for t in used:
                    # answer sentences that carry this tag are the claim being supported
                    claim = " ".join(ln for ln in answer.split("\n") if f"[{t}]" in ln)
                    focus = set(tokenize(_TAG.sub(" ", claim))) | q_terms
                    citations.append(self._build_citation(contract, t, tagged[t], True, focus))
            else:  # model cited nothing valid: fall back to the best retrieved chunk
                citations = [self._build_citation(contract, "C1", tagged["C1"], False, q_terms)]
            return ContractAnswer(
                contract_id=contract.id,
                contract_name=contract.filename,
                answer=answer,
                citations=citations,
            )

        results = list(await asyncio.gather(*(answer_one(c) for c in contracts)))

        comparison = None
        if len(results) > 1:
            usable = {r.contract_name: r.answer for r in results if r.answer and not r.error}
            if len(usable) > 1:
                try:
                    comparison = await self.provider.compare(req.question, usable)
                except Exception:  # noqa: BLE001 - comparison is optional
                    logger.exception("Comparison step failed")

        def persist() -> int:
            db.add(
                Message(
                    conversation_id=conv.id,
                    role="user",
                    content=req.question,
                    contract_ids=[c.id for c in contracts],
                )
            )
            payload = QueryResponse(
                conversation_id=conv.id,
                message_id=0,
                question=req.question,
                results=results,
                comparison=comparison,
                provider=self.provider.name,
            )
            text = "\n\n".join(f"{r.contract_name}: {r.answer or r.error or ''}" for r in results)
            msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=text,
                contract_ids=[c.id for c in contracts],
                results=payload.model_dump(mode="json", exclude={"message_id"}),
            )
            db.add(msg)
            conv.updated_at = datetime.now(timezone.utc)
            db.commit()
            return msg.id

        message_id = await run_in_threadpool(persist)
        return QueryResponse(
            conversation_id=conv.id,
            message_id=message_id,
            question=req.question,
            results=results,
            comparison=comparison,
            provider=self.provider.name,
        )


# Public alias so other services can ground citations the same way.
build_citation = RagService._build_citation
