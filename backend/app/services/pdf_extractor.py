"""PDF -> numbered physical lines with bounding boxes (the basis for page/line grounding).

Scanned pages (no text layer) go through OCR (see ``app.services.ocr``) and then the same
line/bbox pipeline, so downstream chunking and citations don't care where the text came from.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.services.ocr import OcrConfig, OcrUnavailable, ocr_textpage, resolve_tessdata

logger = logging.getLogger(__name__)


class PdfExtractionError(Exception):
    """Raised for unreadable, encrypted or text-less PDFs."""


@dataclass
class ExtractedLine:
    page: int  # 1-based
    line_number: int  # 1-based, restarts on every page
    text: str
    bbox: list[float]  # normalised [x0, y0, x1, y1], top-left origin
    page_width: float
    page_height: float
    block: int  # block id within the page; consecutive lines in a block form a paragraph
    char_start: int = 0  # offsets into ExtractedDocument.full_text
    char_end: int = 0


@dataclass
class ExtractedPage:
    page: int
    text: str
    width: float
    height: float


@dataclass
class ExtractedDocument:
    pages: list[ExtractedPage] = field(default_factory=list)
    lines: list[ExtractedLine] = field(default_factory=list)
    full_text: str = ""
    ocr_pages: list[int] = field(default_factory=list)  # 1-based pages recognised via OCR

    @property
    def page_count(self) -> int:
        return len(self.pages)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# Some generators draw each text run twice at (almost) the same spot (fake bold, shadow
# or accessibility overlays). Such copies are one visual line, not two.
_DUPLICATE_IOU = 0.6


# Where the first line sits for each text direction (unrotated space): sort keys that give
# reading order for text running left-to-right, upward, downward or upside-down.
_READING_ORDER = {
    (1, 0): lambda b: (b[1], b[0]),
    (0, -1): lambda b: (b[0], -b[3]),
    (0, 1): lambda b: (-b[2], b[1]),
    (-1, 0): lambda b: (-b[3], -b[2]),
}


def _in_reading_order(blocks: list[dict]) -> list[tuple[int, dict]]:
    """(block id, block) pairs. PyMuPDF sorts in unrotated space, which scrambles pages whose
    text runs sideways there (e.g. /Rotate 90 scans); re-sort along the dominant direction."""
    numbered = [(i, b) for i, b in enumerate(blocks) if b.get("type") == 0]
    dirs = Counter(
        (round(l["dir"][0]), round(l["dir"][1])) for _, b in numbered for l in b.get("lines", [])
    )
    dominant = dirs.most_common(1)[0][0] if dirs else (1, 0)
    key = _READING_ORDER.get(dominant)
    if key is None or dominant == (1, 0):
        return numbered  # PyMuPDF's own order is already right
    numbered.sort(key=lambda ib: key(ib[1]["bbox"]))
    for _, b in numbered:
        b["lines"] = sorted(b.get("lines", []), key=lambda l: key(l["bbox"]))
    return numbered


def _page_lines(
    data: dict, page_number: int, width: float, height: float, to_display: "fitz.Matrix"
) -> list[ExtractedLine]:
    """Numbered, de-duplicated lines (with normalised bboxes) from a PyMuPDF text dict.

    PyMuPDF reports boxes in *unrotated* page space; ``to_display`` (the page's rotation
    matrix) maps them to the page as displayed, which is what ``width``/``height`` describe."""
    lines: list[ExtractedLine] = []
    kept_boxes: dict[str, list[list[float]]] = {}  # normalised text -> bboxes kept
    for block_no, block in _in_reading_order(data.get("blocks", [])):  # images are skipped
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            shown = fitz.Rect(line["bbox"]) * to_display
            shown.normalize()
            x0, y0, x1, y1 = shown
            bbox = [
                _clamp(x0 / width),
                _clamp(y0 / height),
                _clamp(x1 / width),
                _clamp(y1 / height),
            ]
            key = " ".join(text.split()).lower()
            if any(_iou(bbox, seen) >= _DUPLICATE_IOU for seen in kept_boxes.get(key, [])):
                continue  # overprinted duplicate
            kept_boxes.setdefault(key, []).append(bbox)
            lines.append(
                ExtractedLine(
                    page=page_number,
                    line_number=len(lines) + 1,
                    text=text,
                    bbox=bbox,
                    page_width=width,
                    page_height=height,
                    block=block_no,
                )
            )
    return lines


def _char_count(lines: list[ExtractedLine]) -> int:
    return sum(len(l.text) for l in lines)


# A page counts as a scanned image when pictures cover at least this share of it.
_MIN_IMAGE_COVERAGE = 0.7


def _image_coverage(page: "fitz.Page") -> float:
    """Share (0..1) of the displayed page that is covered by embedded images."""
    area = page.rect.get_area()
    if area <= 0:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        shown = fitz.Rect(info["bbox"]) * page.rotation_matrix  # image boxes are unrotated
        shown &= page.rect
        if not shown.is_empty:
            covered += shown.get_area()
    return min(covered / area, 1.0)


def extract_pdf(path: str, ocr: OcrConfig | None = None) -> ExtractedDocument:
    """Extract numbered lines per page.

    OCR is a last resort and only runs for *scanned* documents: at least ``ocr.scanned_ratio``
    of the pages must be a full-page image with no text layer. A normal PDF -- including one
    with a blank page, a diagram or a logo-only page -- is read from its text layer alone and
    never reaches the OCR engine. In a scanned document only the image-only pages are OCR'd."""
    cfg = ocr or OcrConfig(enabled=False)
    try:
        doc = fitz.open(path)
    except Exception as exc:  # corrupt / not a PDF
        raise PdfExtractionError(f"Could not open PDF: {exc}") from exc

    result = ExtractedDocument()
    ocr_note = ""  # why a scanned document's pages were left empty
    is_scan = False

    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise PdfExtractionError("PDF is password-protected")
        if doc.page_count == 0:
            raise PdfExtractionError("PDF has no pages")

        pages: list[list[ExtractedLine]] = []
        image_only: list[int] = []  # 0-based pages with no text layer that are a full-page image
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            width, height = float(page.rect.width), float(page.rect.height)
            lines = _page_lines(
                page.get_text("dict", sort=True), page_index + 1, width, height, page.rotation_matrix
            )
            pages.append(lines)
            # Cheapest test first: a page with real text never needs the image check.
            if _char_count(lines) < cfg.min_chars and _image_coverage(page) >= _MIN_IMAGE_COVERAGE:
                image_only.append(page_index)
            result.pages.append(ExtractedPage(page=page_index + 1, text="", width=width, height=height))

        is_scan = bool(image_only) and len(image_only) / doc.page_count >= cfg.scanned_ratio
        if image_only and not is_scan:
            logger.info(
                "%s is a text PDF; %d image-only page(s) are not OCR'd (below the scanned-document threshold)",
                path,
                len(image_only),
            )

        if is_scan:
            if not cfg.enabled:
                ocr_note = "; OCR is disabled"
            elif len(image_only) > cfg.max_pages:
                raise PdfExtractionError(
                    f"{len(image_only)} scanned pages need OCR, but the limit is {cfg.max_pages} per document"
                )
            else:
                try:
                    tessdata = resolve_tessdata(cfg)
                except OcrUnavailable as exc:
                    logger.warning("OCR skipped for %s: %s", path, exc)
                    ocr_note = f"; OCR unavailable: {exc}"
                else:
                    for page_index in image_only:
                        page = doc.load_page(page_index)
                        tp = ocr_textpage(page, cfg, tessdata)
                        width, height = result.pages[page_index].width, result.pages[page_index].height
                        pages[page_index] = _page_lines(
                            page.get_text("dict", textpage=tp, sort=True),
                            page_index + 1,
                            width,
                            height,
                            page.rotation_matrix,
                        )
                        result.ocr_pages.append(page_index + 1)
                    logger.info("OCR'd %d of %d pages of %s", len(image_only), doc.page_count, path)

    for page_meta, lines in zip(result.pages, pages):
        page_meta.text = "\n".join(l.text for l in lines)
        result.lines.extend(lines)

    # Global offsets: lines are joined with "\n" so full_text[start:end] == line.text.
    cursor = 0
    parts: list[str] = []
    for line in result.lines:
        line.char_start = cursor
        line.char_end = cursor + len(line.text)
        parts.append(line.text)
        cursor = line.char_end + 1
    result.full_text = "\n".join(parts)

    if not result.full_text.strip():
        if result.ocr_pages:
            raise PdfExtractionError(
                "No extractable text found (OCR ran but recognised no text in the scanned pages)"
            )
        if is_scan:
            raise PdfExtractionError(
                f"No extractable text found (the PDF is a scanned image{ocr_note})"
            )
        raise PdfExtractionError("No extractable text found (the PDF has no text layer)")
    return result
