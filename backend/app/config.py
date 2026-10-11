from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", extra="ignore")

    data_dir: Path = Field(default=BACKEND_ROOT / "data")
    max_upload_mb: int = 25
    max_files_per_upload: int = 10

    # OCR for scanned PDFs (Tesseract, bundled with PyMuPDF; needs <lang>.traineddata)
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    ocr_min_chars_per_page: int = 20  # fewer extracted chars than this -> page is OCR'd
    ocr_max_pages: int = 150
    ocr_tessdata_dir: str | None = None  # None -> auto-detect

    # Chunking
    chunk_target_tokens: int = 400
    chunk_overlap_tokens: int = 60
    chunk_min_tokens: int = 80

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_enabled: bool = True
    retrieval_top_k: int = 5
    retrieval_candidates: int = 20
    retrieval_min_similarity: float = 0.15

    # LLM
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    llm_provider: str = "auto"  # auto | anthropic | mock
    llm_max_tokens: int = 1024

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'contracts.db').as_posix()}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
