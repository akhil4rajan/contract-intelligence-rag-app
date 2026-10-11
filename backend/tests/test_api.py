import pytest


def _upload(client, **named_files):
    files = [("files", (name, data, "application/pdf")) for name, data in named_files.items()]
    return client.post("/api/contracts", files=files)


@pytest.fixture(scope="module")
def uploaded(client, pdf_a, pdf_b):
    res = _upload(client, **{"acme_msa.pdf": pdf_a, "initech_license.pdf": pdf_b})
    assert res.status_code == 201, res.text
    results = res.json()["results"]
    assert [r["status"] for r in results] == ["accepted", "accepted"]
    return {r["contract"]["filename"]: r["contract"] for r in results}


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "mock"


def test_upload_lists_ready_contracts_with_metadata(client, uploaded):
    contracts = {c["id"]: c for c in client.get("/api/contracts").json()}
    for c in uploaded.values():
        row = contracts[c["id"]]
        assert row["status"] == "ready"  # background indexing has completed
        assert row["page_count"] == 2
        assert row["chunk_count"] >= 1
        assert row["uploaded_at"]


def test_duplicate_upload_is_flagged(client, pdf_a, uploaded):
    res = _upload(client, **{"again.pdf": pdf_a})
    assert res.json()["results"][0]["status"] == "duplicate"


def test_upload_rejects_non_pdf_and_scanned_without_failing_batch(client, pdf_b):
    from tests.conftest import make_pdf

    res = client.post(
        "/api/contracts",
        files=[
            ("files", ("notes.txt", b"hello world", "text/plain")),
            ("files", ("blank.pdf", make_pdf([[]]), "application/pdf")),
        ],
    )
    assert res.status_code == 201
    statuses = [(r["status"], r["error"]["code"]) for r in res.json()["results"]]
    assert statuses == [("rejected", "not_a_pdf"), ("rejected", "unreadable_pdf")]


def test_page_lines_and_file_endpoints(client, uploaded):
    cid = uploaded["acme_msa.pdf"]["id"]
    page = client.get(f"/api/contracts/{cid}/pages/2").json()
    assert page["page_number"] == 2 and page["lines"][0]["line_number"] == 1
    assert client.get(f"/api/contracts/{cid}/pages/9").status_code == 404
    pdf = client.get(f"/api/contracts/{cid}/file")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


def test_multi_contract_query_returns_grounded_answers_per_contract(client, uploaded):
    ids = [c["id"] for c in uploaded.values()]
    res = client.post(
        "/api/chat/query",
        json={"question": "What is the termination notice period?", "contract_ids": ids},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert {r["contract_id"] for r in body["results"]} == set(ids)

    by_name = {r["contract_name"]: r for r in body["results"]}
    acme = by_name["acme_msa.pdf"]
    assert not acme["not_found"] and acme["citations"]
    cite = acme["citations"][0]
    assert cite["page_start"] == 2 and cite["line_start"] >= 1
    assert "sixty (60)" in cite["snippet"]
    assert cite["highlights"] and cite["highlights"][0]["bboxes"]
    assert "thirty (30)" in by_name["initech_license.pdf"]["citations"][0]["snippet"]


def test_conversation_history_is_persisted(client, uploaded):
    cid = uploaded["acme_msa.pdf"]["id"]
    first = client.post(
        "/api/chat/query", json={"question": "Who are the parties?", "contract_ids": [cid]}
    ).json()
    second = client.post(
        "/api/chat/query",
        json={
            "question": "What are the payment terms?",
            "contract_ids": [cid],
            "conversation_id": first["conversation_id"],
        },
    ).json()
    assert second["conversation_id"] == first["conversation_id"]

    conv = client.get(f"/api/conversations/{first['conversation_id']}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["messages"][1]["results"]["results"][0]["contract_id"] == cid
    assert any(c["id"] == conv["id"] for c in client.get("/api/conversations").json())

    assert client.delete(f"/api/conversations/{conv['id']}").status_code == 204
    assert client.get(f"/api/conversations/{conv['id']}").status_code == 404


def test_certificate_style_document_gives_distinct_grounded_answers(client):
    from tests.conftest import CERTIFICATE_LINES, make_label_pdf

    pdf = make_label_pdf(CERTIFICATE_LINES, overprint=True)
    contract = _upload(client, **{"certificate.pdf": pdf}).json()["results"][0]["contract"]
    cid = contract["id"]
    ask = lambda q, conv=None: client.post(  # noqa: E731
        "/api/chat/query", json={"question": q, "contract_ids": [cid], "conversation_id": conv}
    ).json()

    first = ask("who completed the course?")
    conv = first["conversation_id"]
    who = first["results"][0]
    overview = ask("what is this document about ?", conv)["results"][0]
    dur = ask("how many hours and minutes was the course?", conv)["results"][0]
    greeting = ask("hi", conv)["results"][0]

    assert "Akhil Rajan" in who["answer"] and "Certificate ID" not in who["answer"]
    assert who["citations"] and who["citations"][0]["score"] is not None

    assert overview["answer"].startswith("This document appears to cover:")
    assert not overview["not_found"] and overview["citations"]
    assert overview["citations"][0]["score"] is None  # no fake relevance for overviews

    assert "6 hours 25 minutes" in dur["answer"]
    assert len({who["answer"], overview["answer"], dur["answer"]}) == 3
    assert greeting["not_found"]

    # overprinted duplicates were removed: each line numbered once
    page = client.get(f"/api/contracts/{cid}/pages/1").json()
    assert [l["text"] for l in page["lines"]] == CERTIFICATE_LINES


def test_query_validation_and_errors(client, uploaded):
    cid = next(iter(uploaded.values()))["id"]
    assert client.post("/api/chat/query", json={"question": "  ", "contract_ids": [cid]}).status_code == 422
    assert client.post("/api/chat/query", json={"question": "hi", "contract_ids": []}).status_code == 422
    missing = client.post("/api/chat/query", json={"question": "hi", "contract_ids": ["nope"]})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert client.get("/api/contracts/nope").status_code == 404


def test_delete_contract_removes_it_from_listing(client, pdf_a):
    from tests.conftest import make_pdf

    unique = make_pdf([["Delete me contract with unique text to avoid duplicate detection xyz123."]])
    cid = _upload(client, **{"temp.pdf": unique}).json()["results"][0]["contract"]["id"]
    assert client.delete(f"/api/contracts/{cid}").status_code == 204
    assert client.get(f"/api/contracts/{cid}").status_code == 404


def test_scanned_pdf_is_ingested_via_ocr_and_answers_with_citations(client, pdf_a):
    from tests.conftest import make_scanned_pdf, requires_tesseract

    if requires_tesseract.args[0]:
        pytest.skip("Tesseract language data not installed")

    # A scan of different content, so the duplicate check doesn't hit pdf_a's upload.
    from tests.conftest import make_pdf

    scan = make_scanned_pdf(
        make_pdf([["Termination. Either party may terminate by giving ninety (90) days written notice."]])
    )
    res = _upload(client, **{"scanned_nda.pdf": scan})
    result = res.json()["results"][0]
    assert result["status"] == "accepted", result
    cid = result["contract"]["id"]
    assert client.get(f"/api/contracts/{cid}").json()["status"] == "ready"

    page = client.get(f"/api/contracts/{cid}/pages/1").json()
    assert "ninety (90) days" in " ".join(l["text"] for l in page["lines"])

    body = client.post(
        "/api/chat/query",
        json={"question": "What is the termination notice period?", "contract_ids": [cid]},
    ).json()
    answer = body["results"][0]
    assert not answer["not_found"] and answer["citations"]
    cite = answer["citations"][0]
    assert cite["page_start"] == 1 and "ninety (90)" in cite["snippet"]
    assert cite["highlights"] and cite["highlights"][0]["bboxes"]  # grounded on the scan
