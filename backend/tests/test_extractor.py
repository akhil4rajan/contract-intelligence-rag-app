import pytest

from app.services.pdf_extractor import PdfExtractionError, extract_pdf
from tests.conftest import make_pdf


def test_extracts_numbered_lines_per_page(tmp_path, pdf_a):
    path = tmp_path / "a.pdf"
    path.write_bytes(pdf_a)
    doc = extract_pdf(str(path))

    assert doc.page_count == 2
    page1 = [l for l in doc.lines if l.page == 1]
    page2 = [l for l in doc.lines if l.page == 2]
    assert [l.line_number for l in page1] == list(range(1, len(page1) + 1))
    assert page2[0].line_number == 1  # numbering restarts on each page
    assert "sixty (60) days" in " ".join(l.text for l in page2)


def test_bboxes_are_normalised_and_offsets_round_trip(tmp_path, pdf_a):
    path = tmp_path / "a.pdf"
    path.write_bytes(pdf_a)
    doc = extract_pdf(str(path))

    for line in doc.lines:
        x0, y0, x1, y1 = line.bbox
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        assert doc.full_text[line.char_start : line.char_end] == line.text


def test_overprinted_duplicate_lines_are_collapsed(tmp_path):
    from tests.conftest import CERTIFICATE_LINES, make_label_pdf

    path = tmp_path / "cert.pdf"
    path.write_bytes(make_label_pdf(CERTIFICATE_LINES, overprint=True))
    doc = extract_pdf(str(path))

    assert [l.text for l in doc.lines] == CERTIFICATE_LINES  # each line exactly once
    assert [l.line_number for l in doc.lines] == list(range(1, len(CERTIFICATE_LINES) + 1))


def test_repeated_text_at_different_positions_is_kept(tmp_path):
    from tests.conftest import make_label_pdf

    path = tmp_path / "sig.pdf"
    path.write_bytes(make_label_pdf(["Signature:", "Name:", "Signature:", "Name:"], overprint=True))
    doc = extract_pdf(str(path))

    assert [l.text for l in doc.lines] == ["Signature:", "Name:", "Signature:", "Name:"]


def test_blank_pdf_is_rejected_as_no_text(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf([[]]))
    with pytest.raises(PdfExtractionError, match="No extractable text"):
        extract_pdf(str(path))


def test_corrupt_pdf_is_rejected(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(PdfExtractionError):
        extract_pdf(str(path))


# ---- OCR for scanned PDFs ---------------------------------------------------------------

from app.services.ocr import OcrConfig  # noqa: E402
from tests.conftest import make_scanned_pdf, requires_tesseract  # noqa: E402


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_scanned_pdf_without_ocr_is_rejected_with_a_clear_reason(tmp_path, pdf_a):
    path = _write(tmp_path, "scan.pdf", make_scanned_pdf(pdf_a))
    with pytest.raises(PdfExtractionError, match="No extractable text.*OCR is disabled"):
        extract_pdf(path)
    with pytest.raises(PdfExtractionError, match="OCR is disabled"):
        extract_pdf(path, OcrConfig(enabled=False))


def test_missing_language_data_is_reported_not_crashed(tmp_path, pdf_a):
    path = _write(tmp_path, "scan.pdf", make_scanned_pdf(pdf_a))
    cfg = OcrConfig(tessdata_dir=str(tmp_path / "nope"))
    with pytest.raises(PdfExtractionError, match="OCR unavailable.*eng.traineddata"):
        extract_pdf(path, cfg)


def test_ocr_page_limit_is_enforced_before_any_ocr_runs(tmp_path, pdf_a):
    path = _write(tmp_path, "scan.pdf", make_scanned_pdf(pdf_a))  # 2 scanned pages
    with pytest.raises(PdfExtractionError, match="limit is 1"):
        extract_pdf(path, OcrConfig(max_pages=1))


@requires_tesseract
def test_scanned_pdf_is_recognised_with_grounding(tmp_path, pdf_a):
    path = _write(tmp_path, "scan.pdf", make_scanned_pdf(pdf_a))
    doc = extract_pdf(path, OcrConfig())

    assert doc.ocr_pages == [1, 2]
    assert doc.page_count == 2
    page2 = [l for l in doc.lines if l.page == 2]
    assert page2[0].line_number == 1  # numbering restarts per page, as for digital PDFs
    assert "sixty (60) days" in " ".join(l.text for l in page2)
    assert "forty-five (45) days" in " ".join(l.text for l in doc.lines if l.page == 1)
    for line in doc.lines:
        x0, y0, x1, y1 = line.bbox
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        assert doc.full_text[line.char_start : line.char_end] == line.text


@requires_tesseract
def test_scanned_document_only_ocrs_its_image_only_pages(tmp_path, pdf_a):
    import fitz

    scanned = make_scanned_pdf(pdf_a)
    with fitz.open(stream=pdf_a, filetype="pdf") as digital, fitz.open(stream=scanned, filetype="pdf") as scan:
        digital.insert_pdf(scan)  # pages: digital, digital, scanned, scanned -> 50% scanned
        mixed = digital.tobytes()
    doc = extract_pdf(_write(tmp_path, "mixed.pdf", mixed), OcrConfig())

    assert doc.page_count == 4
    assert doc.ocr_pages == [3, 4]  # the two text-layer pages were not OCR'd
    flat = lambda text: " ".join(text.split())
    assert "sixty (60) days" in flat(doc.pages[1].text)  # untouched text layer
    assert "sixty (60) days" in flat(doc.pages[3].text)  # recognised page


@requires_tesseract
@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_sideways_or_upside_down_scan_is_recognised(tmp_path, pdf_a, rotation):
    path = _write(tmp_path, "rot.pdf", make_scanned_pdf(pdf_a, rotation=rotation))
    doc = extract_pdf(path, OcrConfig())
    assert "sixty (60) days" in " ".join(doc.full_text.split())


@requires_tesseract
def test_landscape_scan_boxes_are_in_displayed_page_space(tmp_path, pdf_a):
    from tests.conftest import make_landscape_scan

    doc = extract_pdf(_write(tmp_path, "land.pdf", make_landscape_scan(pdf_a)), OcrConfig())
    assert "sixty (60) days" in " ".join(doc.full_text.split())
    for line in doc.lines:  # horizontal text on an upright page => boxes wider than tall
        x0, y0, x1, y1 = line.bbox
        assert (x1 - x0) * line.page_width > (y1 - y0) * line.page_height


@requires_tesseract
def test_blank_scanned_page_is_still_rejected(tmp_path):
    path = _write(tmp_path, "blank.pdf", make_scanned_pdf(make_pdf([[]])))
    with pytest.raises(PdfExtractionError, match="No extractable text"):
        extract_pdf(path, OcrConfig())


# ---- proper (text) PDFs must never reach the OCR engine ---------------------------------


@pytest.fixture
def ocr_forbidden(monkeypatch):
    """Fail loudly if the OCR engine, or even tessdata discovery, is touched."""
    import app.services.pdf_extractor as mod

    def boom(*_a, **_k):
        raise AssertionError("OCR must not run for this document")

    monkeypatch.setattr(mod, "ocr_textpage", boom)
    monkeypatch.setattr(mod, "resolve_tessdata", boom)


def _digital_with_extras(pdf: bytes, *, blank_pages: int = 0, image_pages: int = 0) -> bytes:
    """A text PDF plus blank pages and/or pages holding only a full-page picture."""
    import fitz

    with fitz.open(stream=pdf, filetype="pdf") as doc:
        for _ in range(blank_pages):
            doc.new_page()
        for _ in range(image_pages):
            page = doc.new_page()
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 260), False)
            pix.clear_with(180)
            page.insert_image(page.rect, pixmap=pix)  # e.g. a full-bleed diagram
        return doc.tobytes()


def test_text_pdf_never_reaches_ocr(tmp_path, pdf_a, ocr_forbidden):
    doc = extract_pdf(_write(tmp_path, "a.pdf", pdf_a), OcrConfig())
    assert doc.ocr_pages == [] and "sixty (60)" in " ".join(doc.full_text.split())


def test_text_pdf_with_blank_pages_never_reaches_ocr(tmp_path, pdf_a, ocr_forbidden):
    path = _write(tmp_path, "blank.pdf", _digital_with_extras(pdf_a, blank_pages=1))
    doc = extract_pdf(path, OcrConfig())
    assert doc.page_count == 3 and doc.ocr_pages == []


def test_text_pdf_with_image_only_pages_is_not_treated_as_a_scan(tmp_path, pdf_a, ocr_forbidden):
    # 2 text pages + 1 picture page: a minority of image pages must not turn on OCR.
    path = _write(tmp_path, "figs.pdf", _digital_with_extras(pdf_a, image_pages=1))
    doc = extract_pdf(path, OcrConfig())
    assert doc.page_count == 3 and doc.ocr_pages == []
    assert "sixty (60)" in " ".join(doc.full_text.split())


def test_blank_digital_pdf_is_rejected_without_ocr(tmp_path, ocr_forbidden):
    with pytest.raises(PdfExtractionError, match="No extractable text.*no text layer"):
        extract_pdf(_write(tmp_path, "b.pdf", make_pdf([[]])), OcrConfig())


def test_text_pdf_with_a_stamp_style_short_page_never_reaches_ocr(tmp_path, ocr_forbidden):
    # Pages with only a few words (a cover or signature page) have a text layer: not a scan.
    path = _write(tmp_path, "cover.pdf", make_pdf([["CONTRACT FOR SERVICES - ACME CORP"], ["Signed."]]))
    doc = extract_pdf(path, OcrConfig())
    assert doc.ocr_pages == []
