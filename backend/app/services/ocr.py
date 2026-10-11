"""OCR for scanned (image-only) PDF pages, using the Tesseract engine bundled with PyMuPDF.

PyMuPDF ships the Tesseract library, so no ``tesseract`` executable is needed -- only the
language data (``<lang>.traineddata``). OCR output is a PyMuPDF TextPage, so it flows through
the same line/bbox extraction as a real text layer.
"""

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class OcrUnavailable(Exception):
    """OCR is wanted but the language data could not be found."""


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool = True
    language: str = "eng"  # Tesseract codes; join with "+" for several, e.g. "eng+deu"
    dpi: int = 300  # lower values (e.g. 150) are faster but lose small print and spacing
    min_chars: int = 20  # a page with fewer extracted characters has no usable text layer
    scanned_ratio: float = 0.5  # share of image-only pages at which a document counts as scanned
    max_pages: int = 150  # cap on OCR'd pages per document (bounds request time)
    tessdata_dir: str | None = None  # None -> auto-detect


_WELL_KNOWN_TESSDATA = (
    r"C:\Program Files\Tesseract-OCR\tessdata",
    r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tessdata",
    "/usr/local/share/tessdata",
    "/opt/homebrew/share/tessdata",
)

# Tesseract (via MuPDF) keeps per-call global state; OCR is CPU-bound anyway, so serialise it.
_OCR_LOCK = threading.Lock()


def _has_language(directory: str, language: str) -> bool:
    return all((Path(directory) / f"{lang}.traineddata").is_file() for lang in language.split("+"))


def _candidates(cfg: OcrConfig) -> list[str]:
    found: list[str] = []
    if cfg.tessdata_dir:
        return [cfg.tessdata_dir]  # an explicit setting is never second-guessed
    if os.environ.get("TESSDATA_PREFIX"):
        found.append(os.environ["TESSDATA_PREFIX"])
    try:
        found.append(fitz.get_tessdata())
    except Exception:  # noqa: BLE001  (raises when Tesseract is not on PATH)
        pass
    found.extend(_WELL_KNOWN_TESSDATA)
    return found


def resolve_tessdata(cfg: OcrConfig) -> str:
    """Return a tessdata directory containing every requested language, or raise."""
    for directory in _candidates(cfg):
        if directory and _has_language(directory, cfg.language):
            return directory
    where = cfg.tessdata_dir or "TESSDATA_PREFIX or a standard Tesseract install"
    raise OcrUnavailable(
        f"Tesseract language data '{cfg.language}.traineddata' not found in {where}. "
        "Install Tesseract OCR or set OCR_TESSDATA_DIR."
    )


# A sideways or upside-down scan makes Tesseract emit noise rather than fail. English contract
# prose is full of these words (~30-50% of tokens); noise scores below ~15%.
_COMMON_ENGLISH = frozenset(
    "the of and to in shall be by or for as with that this is are any party parties agreement "
    "on at from its it not which will may such under all other has have been their".split()
)
_WORD = re.compile(r"[^\W\d_]+")
_MIN_WORDS_TO_JUDGE = 8  # sparse pages (signature blocks, covers) are accepted as they are


def _plausibility(text: str, language: str) -> tuple[float, int]:
    """(score 0..1, word count). English: share of common words. Other languages: share of
    words with 3+ letters, which still separates prose from the 1-2 letter shards of noise."""
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return 0.0, 0
    if language.split("+")[0] == "eng":
        hits = sum(w in _COMMON_ENGLISH for w in words)
    else:
        hits = sum(len(w) >= 3 for w in words)
    return hits / len(words), len(words)


def ocr_textpage(page: "fitz.Page", cfg: OcrConfig, tessdata: str) -> "fitz.TextPage":
    """OCR the whole page (ignoring any text layer) and return a TextPage.

    If the result looks like noise, the page is retried rotated by 90/270/180 degrees and the
    most plausible reading wins. Line boxes are in unrotated page space whatever the rotation
    tried, so callers can map them with ``page.rotation_matrix`` as usual."""
    good = 0.25 if cfg.language.split("+")[0] == "eng" else 0.5
    original = page.rotation
    best: "fitz.TextPage | None" = None
    best_score = -1.0
    try:
        for extra in (0, 90, 270, 180):
            page.set_rotation((original + extra) % 360)
            with _OCR_LOCK:
                tp = page.get_textpage_ocr(
                    dpi=cfg.dpi, language=cfg.language, full=True, tessdata=tessdata
                )
            score, n_words = _plausibility(tp.extractText(), cfg.language)
            if n_words < _MIN_WORDS_TO_JUDGE or score >= good:
                if extra:
                    logger.info("OCR: page %s read correctly after rotating %d degrees", page.number + 1, extra)
                return tp
            if score > best_score:
                best, best_score = tp, score
    finally:
        page.set_rotation(original)
    logger.warning(
        "OCR: page %s: no orientation gave clearly readable text (best score %.2f)",
        page.number + 1,
        best_score,
    )
    assert best is not None
    return best
