from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_extraction_service
from app.schemas import ExtractionOut
from app.services.extraction import ExtractionService

router = APIRouter(prefix="/contracts", tags=["extraction"])


@router.get("/{contract_id}/extraction", response_model=ExtractionOut)
async def get_extraction(
    contract_id: str,
    db: Session = Depends(get_db),
    service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionOut:
    """Start/end date, supplier name and rates for one contract (cached after first use)."""
    return await service.get(db, contract_id)


@router.post("/{contract_id}/extraction", response_model=ExtractionOut)
async def refresh_extraction(
    contract_id: str,
    db: Session = Depends(get_db),
    service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionOut:
    """Re-run the extraction, replacing the cached result."""
    return await service.get(db, contract_id, refresh=True)
