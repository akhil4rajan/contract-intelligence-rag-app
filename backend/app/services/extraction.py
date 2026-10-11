"""Structured extraction: retrieve -> provider.extract_terms -> ground quotes as citations -> cache.

The provider returns a ``TermsDraft`` (values + a verbatim quote per value). This service
resolves each quote to the page/line citation of the passage it was read from, producing the
``ContractTerms`` that the API serves. Successful results are cached per contract.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone

from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.errors import ContractNotReady, NotFoundError
from app.llm.base import ContextChunk, LLMProvider
from app.models import Contract, ContractExtraction
from app.schemas import (
    Citation,
    ContractTerms,
    DateDraft,
    DateField,
    Evidence,
    ExtractionOut,
    RateItem,
    TermsDraft,
    TextField,
)
from app.services.rag import build_citation
from app.services.retriever import Retriever, tokenize

logger = logging.getLogger(__name__)

# One search per field, plus the document's opening (parties and effective date live there).
_QUERIES = (
    "effective date commencement date start date of this agreement",
    "expiration date end date term of agreement initial term expires terminates renewal",
    "supplier provider vendor licensor contractor parties between",
    "rates fees charges prices payment amount per hour per month annual fee invoice",
)
_LEAD_CHUNKS = 2
_TOP_K = 3
_MAX_RATES = 15
_QUOTE_MATCH = 0.5  # share of a quote's words that must occur in the chunk it is attributed to


def _overlap(terms: set[str], doc: Document) -> int:
    return len(terms & set(tokenize(doc.page_content)))


def _best_tag(tagged: dict[str, Document], terms: set[str]) -> str | None:
    """The excerpt sharing the most words with `terms` (earliest on a tie), or None."""
    best, best_n = None, 0
    for tag, doc in tagged.items():
        n = _overlap(terms, doc)
        if n > best_n:
            best, best_n = tag, n
    return best


def _date_terms(d: DateDraft) -> set[str]:
    assert d.value is not None
    return {d.value.strftime("%B").lower(), str(d.value.day), str(d.value.year)}


class ExtractionService:
    def __init__(self, retriever: Retriever, provider: LLMProvider) -> None:
        self.retriever = retriever
        self.provider = provider
        self._locks: dict[str, asyncio.Lock] = {}  # one extraction at a time per contract

    # ---- entrypoint --------------------------------------------------------------------

    async def get(self, db: Session, contract_id: str, *, refresh: bool = False) -> ExtractionOut:
        """Cached extraction for a ready contract; computed (and cached) on first request."""
        contract = await run_in_threadpool(self._load, db, contract_id)
        async with self._locks.setdefault(contract_id, asyncio.Lock()):
            if not refresh:
                cached = await run_in_threadpool(self._cached, db, contract)
                if cached is not None:
                    return cached
            return await self._extract_and_store(db, contract)

    # ---- persistence -------------------------------------------------------------------

    @staticmethod
    def _load(db: Session, contract_id: str) -> Contract:
        contract = db.get(Contract, contract_id)
        if contract is None:
            raise NotFoundError(f"Contract {contract_id} not found")
        if contract.status != "ready":
            raise ContractNotReady(
                "Contract is not ready for extraction yet", details={"contract_ids": [contract_id]}
            )
        return contract

    def _cached(self, db: Session, contract: Contract) -> ExtractionOut | None:
        row = db.get(ContractExtraction, contract.id)
        # A result from another provider (e.g. the mock, before an API key was added) is stale.
        if row is None or row.provider != self.provider.name:
            return None
        try:
            terms = ContractTerms.model_validate(row.data)
        except ValidationError:  # schema changed since it was stored: recompute
            return None
        return ExtractionOut(
            contract_id=contract.id,
            contract_name=contract.filename,
            status="ready",
            provider=row.provider,
            extracted_at=row.extracted_at,
            terms=terms,
        )

    def _store(self, db: Session, contract_id: str, terms: ContractTerms, at: datetime) -> None:
        try:
            row = db.get(ContractExtraction, contract_id)
            data = terms.model_dump(mode="json")
            if row is None:
                db.add(
                    ContractExtraction(
                        contract_id=contract_id,
                        provider=self.provider.name,
                        data=data,
                        extracted_at=at,
                    )
                )
            else:
                row.provider, row.data, row.extracted_at = self.provider.name, data, at
            db.commit()
        except Exception:  # noqa: BLE001 - e.g. contract deleted meanwhile; the result is still valid
            db.rollback()
            logger.exception("Could not cache extraction for %s", contract_id)

    # ---- extraction --------------------------------------------------------------------

    async def _extract_and_store(self, db: Session, contract: Contract) -> ExtractionOut:
        base = {
            "contract_id": contract.id,
            "contract_name": contract.filename,
            "provider": self.provider.name,
        }
        try:
            context, tagged = await self._gather_context(contract.id)
            draft = await self.provider.extract_terms(contract.filename, context)
        except Exception as exc:  # noqa: BLE001 - isolate: never take the app down over one contract
            logger.exception("Extraction failed for %s", contract.id)
            return ExtractionOut(
                **base, status="failed", error=f"Extraction failed: {type(exc).__name__}"
            )

        terms = self._ground(contract, draft, tagged)
        now = datetime.now(timezone.utc)
        await run_in_threadpool(self._store, db, contract.id, terms, now)
        return ExtractionOut(**base, status="ready", extracted_at=now, terms=terms)

    async def _gather_context(
        self, contract_id: str
    ) -> tuple[list[ContextChunk], dict[str, Document]]:
        searches = [self.retriever.as_langchain(contract_id, _LEAD_CHUNKS, lead=True).ainvoke("")]
        searches += [
            self.retriever.as_langchain(contract_id, _TOP_K).ainvoke(q) for q in _QUERIES
        ]
        merged: dict[int, Document] = {}
        for docs in await asyncio.gather(*searches):
            for d in docs:
                merged.setdefault(d.metadata["chunk_id"], d)

        # Document order reads naturally; no single query "scored" these, so drop the score.
        ordered = sorted(merged.values(), key=lambda d: d.metadata["chunk_index"])
        tagged = {
            f"C{i}": Document(page_content=d.page_content, metadata={**d.metadata, "score": None})
            for i, d in enumerate(ordered, start=1)
        }
        context = [
            ContextChunk(
                tag=tag,
                chunk_id=d.metadata["chunk_id"],
                text=d.page_content,
                page_start=d.metadata["start_page"],
                line_start=d.metadata["start_line"],
                page_end=d.metadata["end_page"],
                line_end=d.metadata["end_line"],
            )
            for tag, d in tagged.items()
        ]
        return context, tagged

    # ---- grounding ---------------------------------------------------------------------

    @staticmethod
    def _citations(
        contract: Contract,
        evidence: Evidence | None,
        value_terms: set[str],
        tagged: dict[str, Document],
    ) -> list[Citation]:
        """Point a value at the tightest lines that state it.

        Trusts the model's tag when its quote really is in that excerpt; otherwise finds the
        excerpt the quote (or, failing that, the value) best matches. Returns [] when the
        value cannot be located at all, so the UI shows it without a source.
        """
        quote_terms = set(tokenize(evidence.quote)) if evidence else set()
        tag: str | None = None
        if evidence:
            claimed = re.sub(r"[^A-Za-z0-9]", "", evidence.tag).upper()
            if claimed in tagged and (
                not quote_terms or _overlap(quote_terms, tagged[claimed]) >= _QUOTE_MATCH * len(quote_terms)
            ):
                tag = claimed
            elif quote_terms:
                tag = _best_tag(tagged, quote_terms) or (claimed if claimed in tagged else None)
        if tag is None:
            tag = _best_tag(tagged, value_terms)
        if tag is None:
            return []
        focus = quote_terms or value_terms
        return [build_citation(contract, tag, tagged[tag], True, focus, focused=True)]

    def _ground(
        self, contract: Contract, draft: TermsDraft, tagged: dict[str, Document]
    ) -> ContractTerms:
        def date_field(d: DateDraft | None) -> DateField:
            if d is None or d.value is None:
                return DateField()
            return DateField(
                value=d.value,
                derived_from=(d.basis or "").strip() or None,
                citations=self._citations(contract, d.evidence, _date_terms(d), tagged),
            )

        supplier = TextField()
        if draft.supplier_name and (name := (draft.supplier_name.value or "").strip()):
            supplier = TextField(
                value=name,
                citations=self._citations(
                    contract, draft.supplier_name.evidence, set(tokenize(name)), tagged
                ),
            )

        rates: list[RateItem] = []
        seen: set[tuple[str, str, str]] = set()
        for r in draft.rates:
            amount = r.amount.strip()
            if not amount:
                continue
            unit = (r.unit or "").strip() or None
            description = r.description.strip() or amount
            key = (description.lower(), amount.lower(), (unit or "").lower())
            if key in seen:
                continue
            seen.add(key)
            rates.append(
                RateItem(
                    description=description,
                    amount=amount,
                    unit=unit,
                    citations=self._citations(contract, r.evidence, set(tokenize(amount)), tagged),
                )
            )
            if len(rates) >= _MAX_RATES:
                break

        return ContractTerms(
            start_date=date_field(draft.start_date),
            end_date=date_field(draft.end_date),
            supplier_name=supplier,
            rates=rates,
        )
