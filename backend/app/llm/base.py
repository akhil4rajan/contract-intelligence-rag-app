from dataclasses import dataclass
from typing import Protocol

from app.schemas import TermsDraft

NOT_FOUND_PHRASE = "I could not find this in the contract."

SYSTEM_PROMPT = f"""You are a contract analysis assistant. Answer the user's question using ONLY the \
numbered context excerpts from the contract provided.

Rules:
- Cite every claim with the excerpt tag(s) in square brackets, e.g. [C1] or [C2][C3].
- Never use information outside the excerpts and never invent clause numbers, dates or amounts.
- Quote figures, dates and parties exactly as written.
- If the excerpts do not contain the answer, reply with exactly: "{NOT_FOUND_PHRASE}"
- Be concise and precise. Use short bullet points for multi-part answers."""

EXTRACTION_PROMPT = """You extract key commercial terms from a contract using ONLY the numbered \
context excerpts provided.

Fields:
- start_date: the date the agreement (or its term) starts: the effective or commencement date.
- end_date: the date the agreement ends. Use an explicit expiry date when one is stated. If only a \
term length is stated (e.g. "three (3) years") and the start date is known, calculate it as the \
day before the anniversary and describe the calculation in `basis`. Otherwise null.
- supplier_name: the legal name of the party supplying the goods, services or licence (the \
supplier, provider, vendor, licensor, contractor or seller). Give the name only, exactly as \
written, without its role label or descriptions such as "a Delaware corporation".
- rates: every price, fee, rate or charge (at most 15). For each: what it is for, the amount \
exactly as written, and the billing unit if stated (e.g. "per hour", "annually"). Do not include \
liability caps, penalties or deposits unless they are charged as a fee.

Rules:
- Never use information outside the excerpts and never guess. Use null (or an empty list) for \
anything the excerpts do not state.
- Dates are ISO 8601 (YYYY-MM-DD).
- For every value give `evidence`: the excerpt tag (e.g. "C2") and a short quote copied \
verbatim from that excerpt (one sentence or line at most)."""


@dataclass
class ContextChunk:
    tag: str  # "C1", "C2", ...
    chunk_id: int
    text: str
    page_start: int
    line_start: int
    page_end: int
    line_end: int


@dataclass
class ChatTurn:
    role: str  # user | assistant
    content: str


class LLMProvider(Protocol):
    name: str

    async def condense_question(self, question: str, history: list[ChatTurn]) -> str:
        """Rewrite a follow-up into a standalone retrieval query."""

    async def generate(
        self, question: str, contract_name: str, context: list[ContextChunk]
    ) -> str:
        """Answer from context only, citing chunk tags like [C1]."""

    async def compare(self, question: str, answers: dict[str, str]) -> str | None:
        """Optional short comparison across contracts (contract name -> answer)."""

    async def extract_terms(self, contract_name: str, context: list[ContextChunk]) -> TermsDraft:
        """Structured extraction of start/end date, supplier and rates from the context."""
