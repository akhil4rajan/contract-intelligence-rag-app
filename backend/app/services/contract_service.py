"""Upload validation, ingestion (extract -> chunk -> persist) and background indexing."""

import hashlib
import logging
import re
import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import SessionLocal
from app.errors import NotFoundError
from app.models import Chunk, Contract, ContractExtraction, Line, Page
from app.services.chunker import chunk_lines
from app.services.embeddings import EmbeddingService
from app.services.ocr import OcrConfig
from app.services.pdf_extractor import PdfExtractionError, extract_pdf
from app.services.retriever import Retriever

logger = logging.getLogger(__name__)


class UploadRejected(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DuplicateContract(Exception):
    def __init__(self, existing: Contract) -> None:
        super().__init__("duplicate")
        self.existing = existing


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def _clean_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    return _SAFE_NAME.sub("_", base).strip() or "contract.pdf"


class ContractService:
    def __init__(self, settings: Settings, embeddings: EmbeddingService, retriever: Retriever) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.retriever = retriever
        self.ocr = OcrConfig(
            enabled=settings.ocr_enabled,
            language=settings.ocr_language,
            dpi=settings.ocr_dpi,
            min_chars=settings.ocr_min_chars_per_page,
            max_pages=settings.ocr_max_pages,
            tessdata_dir=settings.ocr_tessdata_dir,
        )

    # ---- ingestion ---------------------------------------------------------

    def ingest(self, db: Session, filename: str, content_type: str | None, data: bytes) -> Contract:
        """Validate + extract + chunk + persist. Embedding is done later in `index_contract`."""
        if not data:
            raise UploadRejected("empty_file", "The file is empty")
        if len(data) > self.settings.max_upload_bytes:
            raise UploadRejected(
                "file_too_large", f"File exceeds the {self.settings.max_upload_mb} MB limit"
            )
        if not data.startswith(b"%PDF"):
            raise UploadRejected("not_a_pdf", "Only PDF files are supported")
        if not filename.lower().endswith(".pdf"):
            raise UploadRejected("not_a_pdf", "Only .pdf files are supported")

        digest = hashlib.sha256(data).hexdigest()
        existing = db.scalar(select(Contract).where(Contract.sha256 == digest))
        if existing is not None:
            raise DuplicateContract(existing)

        contract_id = uuid.uuid4().hex
        stored = self.settings.upload_dir / f"{contract_id}.pdf"
        stored.write_bytes(data)
        try:
            doc = extract_pdf(str(stored), self.ocr)
        except PdfExtractionError as exc:
            stored.unlink(missing_ok=True)
            raise UploadRejected("unreadable_pdf", str(exc)) from exc

        drafts = chunk_lines(
            doc.lines,
            target_tokens=self.settings.chunk_target_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
            min_tokens=self.settings.chunk_min_tokens,
        )

        contract = Contract(
            id=contract_id,
            filename=_clean_filename(filename),
            stored_path=str(stored),
            size_bytes=len(data),
            page_count=doc.page_count,
            sha256=digest,
            status="processing",
        )
        db.add(contract)
        db.add_all(
            Page(contract_id=contract_id, page_number=p.page, text=p.text) for p in doc.pages
        )
        db.add_all(
            Line(
                contract_id=contract_id,
                page_number=l.page,
                line_number=l.line_number,
                text=l.text,
                bbox=l.bbox,
                page_width=l.page_width,
                page_height=l.page_height,
                char_start=l.char_start,
                char_end=l.char_end,
            )
            for l in doc.lines
        )
        db.add_all(
            Chunk(
                contract_id=contract_id,
                chunk_index=d.index,
                text=d.text,
                token_count=d.token_count,
                start_page=d.start_page,
                start_line=d.start_line,
                end_page=d.end_page,
                end_line=d.end_line,
                char_start=d.char_start,
                char_end=d.char_end,
                line_refs=d.line_refs,
            )
            for d in drafts
        )
        db.commit()
        return contract

    def index_contract(self, contract_id: str) -> None:
        """Background job: embed chunks and flip status to ready (BM25 works regardless)."""
        db = SessionLocal()
        try:
            contract = db.get(Contract, contract_id)
            if contract is None:
                return
            try:
                chunks = list(
                    db.scalars(
                        select(Chunk).where(Chunk.contract_id == contract_id).order_by(Chunk.chunk_index)
                    )
                )
                vecs = self.embeddings.encode([c.text for c in chunks])
                if vecs is not None:
                    for chunk, vec in zip(chunks, vecs):
                        chunk.embedding = np.asarray(vec, dtype=np.float32).tobytes()
                contract.status = "ready"
                contract.error = None
            except Exception as exc:  # noqa: BLE001
                logger.exception("Indexing failed for %s", contract_id)
                contract.status = "failed"
                contract.error = f"Indexing failed: {exc}"
            db.commit()
            self.retriever.invalidate(contract_id)
        finally:
            db.close()

    # ---- queries -----------------------------------------------------------

    def chunk_counts(self, db: Session, contract_ids: list[str]) -> dict[str, int]:
        if not contract_ids:
            return {}
        rows = db.execute(
            select(Chunk.contract_id, func.count(Chunk.id))
            .where(Chunk.contract_id.in_(contract_ids))
            .group_by(Chunk.contract_id)
        )
        return {cid: n for cid, n in rows}

    def get(self, db: Session, contract_id: str) -> Contract:
        contract = db.get(Contract, contract_id)
        if contract is None:
            raise NotFoundError(f"Contract {contract_id} not found")
        return contract

    def delete(self, db: Session, contract_id: str) -> None:
        contract = self.get(db, contract_id)
        path = Path(contract.stored_path)
        for model in (Chunk, Line, Page, ContractExtraction):
            db.execute(delete(model).where(model.contract_id == contract_id))
        db.delete(contract)
        db.commit()
        path.unlink(missing_ok=True)
        self.retriever.invalidate(contract_id)
