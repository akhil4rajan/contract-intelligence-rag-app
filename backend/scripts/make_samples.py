"""Generate small multi-page sample contracts for demos: ``python scripts/make_samples.py``."""

import sys
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[2] / "samples"

CONTRACTS: dict[str, list[list[str]]] = {
    "acme_master_services_agreement.pdf": [
        [
            "MASTER SERVICES AGREEMENT",
            "This Master Services Agreement (the Agreement) is entered into as of January 1, 2025 "
            "between Acme Corporation, a Delaware corporation (Provider), and Globex Limited, a company "
            "organised under the laws of England and Wales (Customer).",
            "1. SERVICES",
            "Provider shall perform the consulting and software implementation services described in "
            "each Statement of Work executed by the parties. Each Statement of Work shall form part of "
            "this Agreement and be governed by its terms.",
            "2. FEES AND PAYMENT",
            "Customer shall pay all undisputed invoices within forty-five (45) days of receipt of a valid "
            "invoice. Late payments shall accrue interest at one percent (1%) per month.",
        ],
        [
            "3. TERM AND TERMINATION",
            "This Agreement commences on the Effective Date and continues for an initial term of three (3) "
            "years. Either party may terminate this Agreement for convenience by giving sixty (60) days "
            "prior written notice to the other party.",
            "Either party may terminate immediately upon written notice if the other party materially "
            "breaches this Agreement and fails to cure such breach within thirty (30) days.",
            "4. LIMITATION OF LIABILITY",
            "In no event shall either party be liable for indirect, incidental or consequential damages. "
            "Each party's aggregate liability under this Agreement shall not exceed the fees paid in the "
            "twelve (12) months preceding the claim.",
            "5. GOVERNING LAW",
            "This Agreement is governed by the laws of the State of New York, without regard to its "
            "conflict of laws principles.",
        ],
    ],
    "initech_software_license.pdf": [
        [
            "SOFTWARE LICENSE AGREEMENT",
            "This Software License Agreement is made between Initech Inc. (Licensor) and Umbrella LLC "
            "(Licensee) and is effective on March 15, 2025.",
            "1. GRANT OF LICENSE",
            "Licensor grants Licensee a non-exclusive, non-transferable licence to use the Software "
            "for its internal business purposes during the term of this Agreement.",
        ],
        [
            "2. FEES",
            "Licensee shall pay annual licence fees of USD 120,000, payable annually in advance within "
            "thirty (30) days of invoice.",
            "3. TERMINATION",
            "Licensee may terminate this licence at any time by giving thirty (30) days written notice "
            "to Licensor. Licensor may terminate only for Licensee's uncured material breach.",
            "4. WARRANTY DISCLAIMER",
            "The Software is provided as is without warranty of any kind, express or implied. Licensor's "
            "total liability shall not exceed the licence fees paid in the preceding twelve (12) months.",
            "5. GOVERNING LAW",
            "This Agreement shall be governed by the laws of the State of California.",
        ],
    ],
}


def render(name: str, pages: list[list[str]]) -> None:
    c = canvas.Canvas(str(OUT / name), pagesize=letter)
    _, height = letter
    for paragraphs in pages:
        y = height - 72
        for para in paragraphs:
            is_heading = para.isupper() or para[:2].rstrip(".").isdigit() and para.upper() == para
            c.setFont("Helvetica-Bold" if is_heading else "Helvetica", 10)
            for line in textwrap.wrap(para, 92):
                c.drawString(72, y, line)
                y -= 14
            y -= 10
        c.showPage()
    c.save()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, pages in CONTRACTS.items():
        render(name, pages)
        print("wrote", OUT / name)


if __name__ == "__main__":
    sys.exit(main())
