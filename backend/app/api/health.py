from fastapi import APIRouter

from app.deps import get_provider
from app.schemas import HealthOut
from app.services.embeddings import get_embedding_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    emb = get_embedding_service().status
    return HealthOut(
        status="ok",
        llm_provider=get_provider().name,
        embeddings=emb,
        retrieval="hybrid (BM25 + dense)" if emb in ("loaded", "not_loaded") else "bm25-only",
    )
