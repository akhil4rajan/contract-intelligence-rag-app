import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, contracts, extraction, health
from app.config import get_settings
from app.db import init_db
from app.deps import get_provider
from app.errors import register_error_handlers
from app.services.embeddings import get_embedding_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    get_provider()  # log which LLM provider is active
    # Warm the embedding model off the request path so the first upload is fast.
    threading.Thread(target=get_embedding_service().warm_up, daemon=True).start()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Contract Intelligence API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    for router in (health.router, contracts.router, extraction.router, chat.router):
        app.include_router(router, prefix="/api")
    return app


app = create_app()
