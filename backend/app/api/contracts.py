from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import get_contract_service
from app.errors import NotFoundError, ValidationFailed
from app.models import Contract, Line
from app.schemas import (
    ContractOut,
    ErrorInfo,
    PageLine,
    PageOut,
    UploadResponse,
    UploadResult,
)
from app.services.contract_service import ContractService, DuplicateContract, UploadRejected

router = APIRouter(prefix="/contracts", tags=["contracts"])


def _out(contract: Contract, chunk_count: int) -> ContractOut:
    out = ContractOut.model_validate(contract)
    out.chunk_count = chunk_count
    return out


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_contracts(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    service: ContractService = Depends(get_contract_service),
) -> UploadResponse:
    settings = get_settings()
    if not files:
        raise ValidationFailed("No files were uploaded")
    if len(files) > settings.max_files_per_upload:
        raise ValidationFailed(f"At most {settings.max_files_per_upload} files per upload")

    results: list[UploadResult] = []
    for f in files:
        name = f.filename or "contract.pdf"
        data = await f.read(settings.max_upload_bytes + 1)
        try:
            contract = await run_in_threadpool(service.ingest, db, name, f.content_type, data)
        except DuplicateContract as dup:
            counts = service.chunk_counts(db, [dup.existing.id])
            results.append(
                UploadResult(
                    filename=name,
                    status="duplicate",
                    contract=_out(dup.existing, counts.get(dup.existing.id, 0)),
                    error=ErrorInfo(code="duplicate", message="This file was already uploaded"),
                )
            )
        except UploadRejected as rej:
            results.append(
                UploadResult(
                    filename=name,
                    status="rejected",
                    error=ErrorInfo(code=rej.code, message=rej.message),
                )
            )
        else:
            counts = service.chunk_counts(db, [contract.id])
            background.add_task(service.index_contract, contract.id)
            results.append(
                UploadResult(
                    filename=name,
                    status="accepted",
                    contract=_out(contract, counts.get(contract.id, 0)),
                )
            )
    return UploadResponse(results=results)


@router.get("", response_model=list[ContractOut])
def list_contracts(
    db: Session = Depends(get_db), service: ContractService = Depends(get_contract_service)
) -> list[ContractOut]:
    contracts = list(db.scalars(select(Contract).order_by(Contract.uploaded_at.desc())))
    counts = service.chunk_counts(db, [c.id for c in contracts])
    return [_out(c, counts.get(c.id, 0)) for c in contracts]


@router.get("/{contract_id}", response_model=ContractOut)
def get_contract(
    contract_id: str,
    db: Session = Depends(get_db),
    service: ContractService = Depends(get_contract_service),
) -> ContractOut:
    contract = service.get(db, contract_id)
    return _out(contract, service.chunk_counts(db, [contract_id]).get(contract_id, 0))


@router.get("/{contract_id}/file")
def get_contract_file(
    contract_id: str,
    db: Session = Depends(get_db),
    service: ContractService = Depends(get_contract_service),
) -> FileResponse:
    contract = service.get(db, contract_id)
    return FileResponse(
        contract.stored_path,
        media_type="application/pdf",
        filename=contract.filename,
        content_disposition_type="inline",
    )


@router.get("/{contract_id}/pages/{page_number}", response_model=PageOut)
def get_page(
    contract_id: str,
    page_number: int,
    db: Session = Depends(get_db),
    service: ContractService = Depends(get_contract_service),
) -> PageOut:
    contract = service.get(db, contract_id)
    if not 1 <= page_number <= contract.page_count:
        raise NotFoundError(f"Page {page_number} not found (document has {contract.page_count} pages)")
    rows = db.scalars(
        select(Line)
        .where(Line.contract_id == contract_id, Line.page_number == page_number)
        .order_by(Line.line_number)
    )
    return PageOut(
        contract_id=contract_id,
        page_number=page_number,
        page_count=contract.page_count,
        lines=[PageLine(line_number=l.line_number, text=l.text, bbox=l.bbox) for l in rows],
    )


@router.delete("/{contract_id}", status_code=204)
def delete_contract(
    contract_id: str,
    db: Session = Depends(get_db),
    service: ContractService = Depends(get_contract_service),
) -> None:
    service.delete(db, contract_id)
