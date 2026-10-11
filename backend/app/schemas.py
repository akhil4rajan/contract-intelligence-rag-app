from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ErrorInfo(BaseModel):
    code: str
    message: str


# ---- Contracts -------------------------------------------------------------


class ContractOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    size_bytes: int
    page_count: int
    chunk_count: int = 0
    status: Literal["processing", "ready", "failed"]
    error: str | None = None
    uploaded_at: datetime


class UploadResult(BaseModel):
    filename: str
    status: Literal["accepted", "duplicate", "rejected"]
    contract: ContractOut | None = None
    error: ErrorInfo | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]


class PageLine(BaseModel):
    line_number: int
    text: str
    bbox: list[float]


class PageOut(BaseModel):
    contract_id: str
    page_number: int
    page_count: int
    lines: list[PageLine]


# ---- Chat ------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    contract_ids: list[str] = Field(min_length=1, max_length=10)
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v

    @field_validator("contract_ids")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class LineRange(BaseModel):
    page: int
    line_start: int
    line_end: int


class Highlight(BaseModel):
    page: int
    bboxes: list[list[float]]  # normalised [x0, y0, x1, y1]


class Citation(BaseModel):
    tag: str
    chunk_id: int
    contract_id: str
    contract_name: str
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    line_ranges: list[LineRange]
    snippet: str
    score: float | None = None  # query relevance 0..1; None for overview answers
    cited: bool = True  # False when added as a fallback because the model cited nothing valid
    highlights: list[Highlight]


class ContractAnswer(BaseModel):
    contract_id: str
    contract_name: str
    answer: str
    not_found: bool = False
    citations: list[Citation] = []
    error: str | None = None


class QueryResponse(BaseModel):
    conversation_id: str
    message_id: int
    question: str
    results: list[ContractAnswer]
    comparison: str | None = None
    provider: str


# ---- Extraction -------------------------------------------------------------
#
# Two layers. The *Draft* models are what the LLM (or mock) must return: values plus a
# verbatim quote from a numbered excerpt. The public models (`ContractTerms`) are what the
# API serves: the same values with each quote resolved to a page/line citation.


class Evidence(BaseModel):
    """Where a value came from: an excerpt tag such as "C2" and a verbatim quote from it."""

    tag: str = Field(description='Excerpt tag the value was read from, e.g. "C2".')
    quote: str = Field(description="Exact words copied from that excerpt that state the value.")


class DateDraft(BaseModel):
    value: date | None = Field(default=None, description="ISO 8601 date (YYYY-MM-DD), or null.")
    basis: str | None = Field(
        default=None,
        description="Only when the date was calculated rather than stated: how it was derived.",
    )
    evidence: Evidence | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _lenient_date(cls, v: Any) -> Any:
        # A model that returns "2025-01-01T00:00:00" or prose must not fail the whole extraction.
        if v is None or isinstance(v, date):
            return v
        try:
            return date.fromisoformat(str(v).strip()[:10])
        except ValueError:
            return None


class TextDraft(BaseModel):
    value: str | None = None
    evidence: Evidence | None = None


class RateDraft(BaseModel):
    description: str = Field(description="What is charged, e.g. 'Annual licence fee'.")
    amount: str = Field(description="The amount exactly as written, e.g. 'USD 120,000' or '$150'.")
    unit: str | None = Field(default=None, description="Billing unit if stated, e.g. 'per hour'.")
    evidence: Evidence | None = None


class TermsDraft(BaseModel):
    """Structured output requested from the model for one contract."""

    start_date: DateDraft | None = Field(
        default=None, description="Date the agreement or its term starts (effective/commencement)."
    )
    end_date: DateDraft | None = Field(
        default=None, description="Date the agreement or its term ends (expiry/termination date)."
    )
    supplier_name: TextDraft | None = Field(
        default=None,
        description="Legal name of the party supplying the goods, services or licence.",
    )
    rates: list[RateDraft] = Field(
        default_factory=list, description="Every price, fee, rate or charge in the contract."
    )


class DateField(BaseModel):
    value: date | None = None
    # Set when the value was calculated (e.g. start date + stated term) rather than quoted.
    derived_from: str | None = None
    citations: list[Citation] = []


class TextField(BaseModel):
    value: str | None = None
    citations: list[Citation] = []


class RateItem(BaseModel):
    description: str
    amount: str
    unit: str | None = None
    citations: list[Citation] = []


class ContractTerms(BaseModel):
    """The extraction schema: the fields pulled out of every contract."""

    start_date: DateField = Field(default_factory=DateField, description="Effective/commencement date.")
    end_date: DateField = Field(default_factory=DateField, description="Expiry/termination date.")
    supplier_name: TextField = Field(default_factory=TextField, description="Supplying party.")
    rates: list[RateItem] = Field(default_factory=list, description="Prices, fees and rates.")


class ExtractionOut(BaseModel):
    contract_id: str
    contract_name: str
    status: Literal["ready", "failed"]
    provider: str
    extracted_at: datetime | None = None
    error: str | None = None
    terms: ContractTerms | None = None


# ---- Conversations ----------------------------------------------------------


class MessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    contract_ids: list[str]
    results: dict | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


# ---- Health -----------------------------------------------------------------


class HealthOut(BaseModel):
    status: Literal["ok"]
    llm_provider: str
    embeddings: str
    retrieval: str
