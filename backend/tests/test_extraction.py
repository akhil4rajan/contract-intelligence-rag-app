"""Structured extraction: Pydantic schema, mock heuristics, grounding, caching and the API."""

import asyncio
from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from app.llm.base import ContextChunk
from app.llm.langchain_provider import LangChainProvider
from app.llm.mock_extraction import extract_terms_heuristic
from app.schemas import DateDraft, Evidence, RateDraft, TermsDraft, TextDraft
from tests.conftest import make_pdf
from tests.test_api import _upload

MSA = [
    [
        "MASTER SERVICES AGREEMENT",
        "This Master Services Agreement (the Agreement) is entered into as of January 1, 2025 "
        "between Acme Corporation, a Delaware corporation (Provider), and Globex Limited, a "
        "company organised under the laws of England and Wales (Customer).",
        "2. FEES AND PAYMENT",
        "Customer shall pay all undisputed invoices within forty-five (45) days of receipt. "
        "Late payments shall accrue interest at one percent (1%) per month.",
    ],
    [
        "3. TERM AND TERMINATION",
        "This Agreement commences on the Effective Date and continues for an initial term of "
        "three (3) years. Either party may terminate for convenience on sixty (60) days notice.",
        "4. LIMITATION OF LIABILITY",
        "Each party's aggregate liability shall not exceed USD 1,000,000 in any year.",
    ],
]

LICENSE = [
    [
        "SOFTWARE LICENSE AGREEMENT",
        "This Software License Agreement is made between Initech Inc. (Licensor) and Umbrella "
        "LLC (Licensee) and is effective on March 15, 2025.",
    ],
    [
        "2. FEES",
        "Licensee shall pay annual licence fees of USD 120,000, payable annually in advance "
        "within thirty (30) days of invoice.",
    ],
]

NOTHING = [["Meeting notes", "We discussed lunch options for the offsite and the parking situation."]]


def _chunk(text: str, tag: str = "C1") -> ContextChunk:
    return ContextChunk(
        tag=tag, chunk_id=1, text=text, page_start=1, line_start=1, page_end=1, line_end=9
    )


@pytest.fixture(scope="module")
def contracts(client):
    out = {}
    for name, pages in {"ex_msa.pdf": MSA, "ex_license.pdf": LICENSE, "ex_notes.pdf": NOTHING}.items():
        res = _upload(client, **{name: make_pdf(pages)}).json()["results"][0]
        assert res["status"] == "accepted", res
        out[name] = res["contract"]["id"]
    return out


# ---- schema ---------------------------------------------------------------------------


def test_draft_dates_are_parsed_leniently():
    assert DateDraft(value="2025-01-01").value == date(2025, 1, 1)
    assert DateDraft(value="2025-01-01T00:00:00Z").value == date(2025, 1, 1)
    assert DateDraft(value="sometime in spring").value is None
    assert TermsDraft().start_date is None and TermsDraft().rates == []


# ---- mock heuristics ------------------------------------------------------------------


def test_heuristics_read_the_usual_phrasings():
    t = extract_terms_heuristic([_chunk("\n".join(l for p in LICENSE for l in p))])
    assert t.start_date.value == date(2025, 3, 15)
    assert t.supplier_name.value == "Initech Inc."
    assert [(r.amount, r.unit) for r in t.rates] == [("USD 120,000", "annually")]
    assert t.end_date is None  # no term or expiry date stated: do not guess


def test_heuristics_explicit_end_date_and_quoted_role_label():
    text = (
        'Agreement dated 5 May 2024 by ABC Trading Ltd (the "Supplier") and Zed Corp (the "Buyer").\n'
        "This Agreement is effective from May 5, 2024 and expires on December 31, 2026.\n"
        "Rate: $150 per hour."
    )
    t = extract_terms_heuristic([_chunk(text)])
    assert t.supplier_name.value == "ABC Trading Ltd"
    assert t.start_date.value == date(2024, 5, 5)
    assert t.end_date.value == date(2026, 12, 31) and t.end_date.basis is None
    assert [(r.amount, r.unit) for r in t.rates] == [("$150", "per hour")]


def test_heuristics_second_party_and_calculated_end_date():
    text = (
        "This Agreement is made between Globex Limited (Customer) and Acme Services Inc. (Supplier), "
        "commencing on 1st day of February 2025. The initial term of 18 months renews for a "
        "renewal term of one (1) year."
    )
    t = extract_terms_heuristic([_chunk(text)])
    assert t.supplier_name.value == "Acme Services Inc."
    assert t.start_date.value == date(2025, 2, 1)
    assert t.end_date.value == date(2026, 7, 31)  # 18 months, day before the anniversary
    assert t.end_date.basis and "18 months" in t.end_date.basis


def test_heuristics_ignore_liability_caps_and_unrelated_text():
    t = extract_terms_heuristic([_chunk("Liability shall not exceed USD 1,000,000 per claim.")])
    assert t.rates == [] and t.supplier_name is None and t.start_date is None
    assert extract_terms_heuristic([]) == TermsDraft()


# ---- API (mock provider) --------------------------------------------------------------


def test_extraction_end_to_end(client, contracts):
    res = client.get(f"/api/contracts/{contracts['ex_msa.pdf']}/extraction")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "ready" and body["provider"] == "mock" and body["error"] is None
    terms = body["terms"]

    assert terms["start_date"]["value"] == "2025-01-01"
    cite = terms["start_date"]["citations"][0]
    assert cite["page_start"] == 1 and "January 1, 2025" in cite["snippet"]
    assert cite["highlights"] and cite["highlights"][0]["bboxes"]

    assert terms["supplier_name"]["value"] == "Acme Corporation"
    assert terms["supplier_name"]["citations"][0]["page_start"] == 1

    end = terms["end_date"]
    assert end["value"] == "2027-12-31" and end["derived_from"]
    assert end["citations"][0]["page_start"] == 2

    amounts = {r["amount"]: r for r in terms["rates"]}
    assert set(amounts) == {"1%"}  # the USD 1,000,000 liability cap is not a rate
    assert amounts["1%"]["unit"] == "per month" and amounts["1%"]["citations"]


def test_extraction_license_has_rate_and_no_end_date(client, contracts):
    terms = client.get(f"/api/contracts/{contracts['ex_license.pdf']}/extraction").json()["terms"]
    assert terms["supplier_name"]["value"] == "Initech Inc."
    assert terms["start_date"]["value"] == "2025-03-15"
    assert terms["end_date"] == {"value": None, "derived_from": None, "citations": []}
    (rate,) = terms["rates"]
    assert rate["amount"] == "USD 120,000" and rate["unit"] == "annually"
    assert rate["citations"][0]["page_start"] == 2


def test_extraction_of_unrelated_document_is_empty_not_an_error(client, contracts):
    body = client.get(f"/api/contracts/{contracts['ex_notes.pdf']}/extraction").json()
    assert body["status"] == "ready"
    terms = body["terms"]
    assert terms["start_date"]["value"] is None and terms["end_date"]["value"] is None
    assert terms["supplier_name"]["value"] is None and terms["rates"] == []


def test_extraction_is_cached_and_refreshable(client, contracts):
    url = f"/api/contracts/{contracts['ex_msa.pdf']}/extraction"
    first = client.get(url).json()
    assert client.get(url).json()["extracted_at"] == first["extracted_at"]
    again = client.post(url)
    assert again.status_code == 200
    assert again.json()["extracted_at"] != first["extracted_at"]
    assert again.json()["terms"] == first["terms"]


def test_extraction_errors(client):
    assert client.get("/api/contracts/nope/extraction").status_code == 404

    from app.db import SessionLocal
    from app.models import Contract

    with SessionLocal() as db:
        db.add(
            Contract(
                id="processing000000000000000000000",
                filename="p.pdf",
                stored_path="x",
                size_bytes=1,
                sha256="0" * 64,
                status="processing",
            )
        )
        db.commit()
    res = client.get("/api/contracts/processing000000000000000000000/extraction")
    assert res.status_code == 409 and res.json()["error"]["code"] == "contract_not_ready"


def test_deleting_a_contract_removes_its_extraction(client):
    from app.db import SessionLocal
    from app.models import ContractExtraction

    pdf = make_pdf([["Supply Agreement effective as of June 1, 2025 between Zeta Ltd (Supplier) and Y."]])
    cid = _upload(client, **{"del_ex.pdf": pdf}).json()["results"][0]["contract"]["id"]
    assert client.get(f"/api/contracts/{cid}/extraction").json()["terms"]["supplier_name"]["value"]
    with SessionLocal() as db:
        assert db.get(ContractExtraction, cid) is not None
    assert client.delete(f"/api/contracts/{cid}").status_code == 204
    with SessionLocal() as db:
        assert db.get(ContractExtraction, cid) is None
    assert client.get(f"/api/contracts/{cid}/extraction").status_code == 404


# ---- service: grounding, caching, failure ----------------------------------------------


class _StubProvider:
    def __init__(self, name: str, draft: TermsDraft | Exception) -> None:
        self.name, self.draft, self.calls = name, draft, 0

    async def extract_terms(self, contract_name, context):  # noqa: ANN001
        self.calls += 1
        if isinstance(self.draft, Exception):
            raise self.draft
        return self.draft


def _service(provider):
    from app.deps import get_retriever
    from app.services.extraction import ExtractionService

    return ExtractionService(get_retriever(), provider)


def _get(service, contract_id, **kw):
    from app.db import SessionLocal

    async def run():
        with SessionLocal() as db:
            return await service.get(db, contract_id, **kw)

    return asyncio.run(run())


def test_citation_is_found_from_the_quote_when_the_model_mislabels_the_tag(client, contracts):
    draft = TermsDraft(
        supplier_name=TextDraft(
            value="Initech Inc.",
            evidence=Evidence(tag="C99", quote="This Software License Agreement is made between"),
        ),
        rates=[
            RateDraft(
                description="Annual licence fee",
                amount="USD 120,000",
                unit="per year",
                evidence=Evidence(tag="[C1]", quote="Licensee shall pay annual licence fees of USD 120,000"),
            )
        ],
    )
    out = _get(_service(_StubProvider("stub-ground", draft)), contracts["ex_license.pdf"])
    terms = out.terms
    assert terms.supplier_name.citations[0].page_start == 1
    # "C1" is the right tag only by luck; the quote lives on page 2 and must win.
    assert terms.rates[0].citations[0].page_start == 2
    assert "120,000" in terms.rates[0].citations[0].snippet


def test_value_without_any_traceable_evidence_has_no_citation(client, contracts):
    draft = TermsDraft(supplier_name=TextDraft(value="Zzyzx Qwerty"))
    out = _get(_service(_StubProvider("stub-nocite", draft)), contracts["ex_license.pdf"])
    assert out.terms.supplier_name.value == "Zzyzx Qwerty"
    assert out.terms.supplier_name.citations == []


def test_cache_is_reused_per_provider_and_failures_are_not_cached(client, contracts):
    cid = contracts["ex_license.pdf"]
    draft = TermsDraft(supplier_name=TextDraft(value="Initech Inc."))

    a = _StubProvider("stub-cache-a", draft)
    svc = _service(a)
    _get(svc, cid)
    _get(svc, cid)
    assert a.calls == 1
    _get(svc, cid, refresh=True)
    assert a.calls == 2

    b = _StubProvider("stub-cache-b", draft)  # a different provider's result is stale
    _get(_service(b), cid)
    assert b.calls == 1

    boom = _StubProvider("stub-boom", RuntimeError("api down"))
    out = _get(_service(boom), cid)
    assert out.status == "failed" and out.terms is None and "RuntimeError" in out.error
    _get(_service(boom), cid)
    assert boom.calls == 2  # a failure is retried, never cached


# ---- LangChain provider ------------------------------------------------------------------


class _ToolModel(FakeListChatModel):
    """A fake chat model whose structured output is a fixed TermsDraft."""

    draft: TermsDraft

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001, ANN003
        assert schema is TermsDraft
        return RunnableLambda(lambda _prompt_value: self.draft)


def test_langchain_provider_extract_terms_uses_structured_output():
    draft = TermsDraft(supplier_name=TextDraft(value="Acme Corporation"))
    provider = LangChainProvider(_ToolModel(responses=["unused"], draft=draft), "fake")
    ctx = [_chunk("Acme Corporation (Provider) ...")]
    assert asyncio.run(provider.extract_terms("msa.pdf", ctx)) == draft
    assert asyncio.run(provider.extract_terms("msa.pdf", [])) == TermsDraft()


def test_langchain_provider_without_tool_support_fails_only_on_extraction():
    provider = LangChainProvider(FakeListChatModel(responses=["x"]), "fake")  # construction is fine
    with pytest.raises(NotImplementedError):
        asyncio.run(provider.extract_terms("msa.pdf", [_chunk("text")]))
