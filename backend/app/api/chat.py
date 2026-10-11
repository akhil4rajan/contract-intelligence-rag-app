from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_rag_service
from app.errors import NotFoundError
from app.models import Conversation, Message
from app.schemas import (
    ConversationOut,
    ConversationSummary,
    MessageOut,
    QueryRequest,
    QueryResponse,
)
from app.services.rag import RagService

router = APIRouter(tags=["chat"])


@router.post("/chat/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    db: Session = Depends(get_db),
    rag: RagService = Depends(get_rag_service),
) -> QueryResponse:
    return await rag.query(db, req)


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(db: Session = Depends(get_db)) -> list[ConversationSummary]:
    counts = dict(
        db.execute(select(Message.conversation_id, func.count(Message.id)).group_by(Message.conversation_id)).all()
    )
    convs = db.scalars(select(Conversation).order_by(Conversation.updated_at.desc()))
    return [
        ConversationSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=counts.get(c.id, 0),
        )
        for c in convs
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ConversationOut:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise NotFoundError(f"Conversation {conversation_id} not found")
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,  # type: ignore[arg-type]
                content=m.content,
                contract_ids=m.contract_ids or [],
                results=m.results,
                created_at=m.created_at,
            )
            for m in conv.messages
        ],
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> None:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise NotFoundError(f"Conversation {conversation_id} not found")
    db.delete(conv)
    db.commit()
