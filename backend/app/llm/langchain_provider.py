"""Provider backed by any LangChain chat model, via LCEL chains: ``prompt | model | parser``."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm.base import EXTRACTION_PROMPT, NOT_FOUND_PHRASE, SYSTEM_PROMPT, ChatTurn, ContextChunk
from app.schemas import TermsDraft

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            'Contract: "{contract_name}"\n\nContext excerpts:\n{context}\n\nQuestion: {question}',
        ),
    ]
)

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question as a standalone search query that can be "
            "understood without the conversation. Output only the rewritten query.",
        ),
        ("human", "Conversation:\n{transcript}\n\nLatest question: {question}"),
    ]
)

_COMPARE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You compare per-contract answers to the same question. In at most 4 sentences, "
            "state the key similarities and differences. Use only the provided answers and "
            "name the contracts. Do not add new facts.",
        ),
        ("human", "Question: {question}\n\n{answers}"),
    ]
)

_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACTION_PROMPT),
        ("human", 'Contract: "{contract_name}"\n\nContext excerpts:\n{context}'),
    ]
)


def _format_context(context: list[ContextChunk]) -> str:
    blocks = []
    for c in context:
        where = (
            f"page {c.page_start}, lines {c.line_start}-{c.line_end}"
            if c.page_start == c.page_end
            else f"page {c.page_start} line {c.line_start} to page {c.page_end} line {c.line_end}"
        )
        blocks.append(f"[{c.tag}] ({where})\n{c.text}")
    return "\n\n".join(blocks)


class LangChainProvider:
    def __init__(self, model: BaseChatModel, name: str) -> None:
        self.name = name
        self._model = model
        self._extract = None  # built on first use: not every chat model supports tools
        parser = StrOutputParser()
        self._answer = _ANSWER_PROMPT | model | parser
        # Short outputs: cap tokens for the auxiliary chains.
        self._condense = _CONDENSE_PROMPT | model.bind(max_tokens=150) | parser
        self._compare = _COMPARE_PROMPT | model.bind(max_tokens=400) | parser

    async def condense_question(self, question: str, history: list[ChatTurn]) -> str:
        if not history:
            return question
        transcript = "\n".join(f"{t.role}: {t.content[:600]}" for t in history[-6:])
        try:
            rewritten = await self._condense.ainvoke({"transcript": transcript, "question": question})
            return rewritten.strip() or question
        except Exception:  # noqa: BLE001 - retrieval can proceed with the raw question
            return question

    async def generate(
        self, question: str, contract_name: str, context: list[ContextChunk]
    ) -> str:
        if not context:
            return NOT_FOUND_PHRASE
        out = await self._answer.ainvoke(
            {
                "contract_name": contract_name,
                "context": _format_context(context),
                "question": question,
            }
        )
        return out.strip()

    async def compare(self, question: str, answers: dict[str, str]) -> str | None:
        if len(answers) < 2:
            return None
        body = "\n\n".join(f'Contract "{n}":\n{a}' for n, a in answers.items())
        try:
            return (await self._compare.ainvoke({"question": question, "answers": body})).strip()
        except Exception:  # noqa: BLE001 - comparison is optional
            return None

    async def extract_terms(self, contract_name: str, context: list[ContextChunk]) -> TermsDraft:
        if not context:
            return TermsDraft()
        if self._extract is None:
            self._extract = _EXTRACT_PROMPT | self._model.with_structured_output(TermsDraft)
        out = await self._extract.ainvoke(
            {"contract_name": contract_name, "context": _format_context(context)}
        )
        if isinstance(out, dict):  # some models/versions return the raw tool arguments
            out = TermsDraft.model_validate(out)
        return out
