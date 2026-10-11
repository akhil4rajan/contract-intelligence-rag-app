import os
import tempfile
import textwrap

# Isolate state and keep tests fast/offline *before* the app is imported.
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="contract-intel-tests-")
os.environ["EMBEDDINGS_ENABLED"] = "false"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["ANTHROPIC_API_KEY"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402


def make_pdf(pages: list[list[str]], path: str | None = None) -> bytes:
    """Each page is a list of paragraphs; paragraphs are wrapped and separated by a gap."""
    import io

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter
    for paragraphs in pages:
        y = height - 72
        for para in paragraphs:
            for line in textwrap.wrap(para, 90):
                c.setFont("Helvetica", 10)
                c.drawString(72, y, line)
                y -= 14
            y -= 14  # paragraph gap
        c.showPage()
    c.save()
    data = buf.getvalue()
    if path:
        with open(path, "wb") as fh:
            fh.write(data)
    return data


def make_label_pdf(lines: list[str], *, overprint: bool = True) -> bytes:
    """Certificate/form style page: short unpunctuated lines, each optionally drawn twice
    at (almost) the same spot, as some PDF generators do."""
    import io

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter
    y = height - 150
    for text in lines:
        c.setFont("Helvetica", 14)
        c.drawString(150, y, text)
        if overprint:
            c.drawString(150.04, y - 0.04, text)
        y -= 30
    c.showPage()
    c.save()
    return buf.getvalue()


def make_scanned_pdf(pdf: bytes, *, dpi: int = 200, rotation: int = 0) -> bytes:
    """Rasterise every page into an image-only PDF, i.e. what a scanner produces."""
    import fitz

    with fitz.open(stream=pdf, filetype="pdf") as src, fitz.open() as out:
        for page in src:
            pix = page.get_pixmap(dpi=dpi)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, pixmap=pix)
            if rotation:
                new.set_rotation(rotation)
        return out.tobytes()


def make_landscape_scan(pdf: bytes, *, dpi: int = 200) -> bytes:
    """What many scanners emit: the image is stored sideways and /Rotate 90 displays it upright."""
    import io

    import fitz
    from PIL import Image

    with fitz.open(stream=pdf, filetype="pdf") as src, fitz.open() as out:
        for page in src:
            img = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png"))).rotate(90, expand=True)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            new = out.new_page(width=page.rect.height, height=page.rect.width)  # unrotated: landscape
            new.insert_image(new.rect, stream=buf.getvalue())
            new.set_rotation(90)
        return out.tobytes()


def _tessdata_available() -> bool:
    from app.services.ocr import OcrConfig, OcrUnavailable, resolve_tessdata

    try:
        resolve_tessdata(OcrConfig())
        return True
    except OcrUnavailable:
        return False


requires_tesseract = pytest.mark.skipif(
    not _tessdata_available(), reason="Tesseract language data (eng.traineddata) not installed"
)


CERTIFICATE_LINES = [
    "Hands-On AI: RAG using LlamaIndex",
    "Course completed by Akhil Rajan",
    "Dec 17, 2024 at 09:11AM UTC  6 hours 25 minutes",
    "Top skills covered",
    "Retrieval-Augmented Generation (RAG)",
    "Artificial Intelligence (AI)",
    "Head of Global Content, Learning",
    "Certificate ID: 605e5ebad47b7d63c5673be79e067f2c",
]


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def pdf_a() -> bytes:
    return make_pdf(
        [
            [
                "MASTER SERVICES AGREEMENT",
                "This Master Services Agreement is entered into between Acme Corporation "
                "and Globex Limited effective as of January 1, 2025.",
                "Payment terms. The Customer shall pay all invoices within forty-five (45) days "
                "of receipt of a valid invoice.",
            ],
            [
                "Termination. Either party may terminate this Agreement for convenience by giving "
                "sixty (60) days prior written notice to the other party.",
                "Governing law. This Agreement is governed by the laws of the State of New York.",
            ],
        ]
    )


@pytest.fixture(scope="session")
def pdf_b() -> bytes:
    return make_pdf(
        [
            [
                "SOFTWARE LICENSE AGREEMENT",
                "This Software License Agreement is made between Initech Inc. and Umbrella LLC.",
            ],
            [
                "Term and termination. Licensee may terminate this license at any time by giving "
                "thirty (30) days written notice to Licensor.",
                "Fees. License fees are payable annually in advance within thirty (30) days.",
            ],
        ]
    )
