"""Process-wide singletons wired once and shared via FastAPI dependencies."""

from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLMProvider
from app.llm.factory import build_provider
from app.services.contract_service import ContractService
from app.services.embeddings import get_embedding_service
from app.services.extraction import ExtractionService
from app.services.rag import RagService
from app.services.retriever import Retriever


@lru_cache
def get_retriever() -> Retriever:
    s = get_settings()
    return Retriever(
        get_embedding_service(),
        candidates=s.retrieval_candidates,
        min_similarity=s.retrieval_min_similarity,
    )


@lru_cache
def get_provider() -> LLMProvider:
    return build_provider(get_settings(), get_embedding_service())


@lru_cache
def get_contract_service() -> ContractService:
    return ContractService(get_settings(), get_embedding_service(), get_retriever())


@lru_cache
def get_rag_service() -> RagService:
    return RagService(get_retriever(), get_provider())


@lru_cache
def get_extraction_service() -> ExtractionService:
    return ExtractionService(get_retriever(), get_provider())
