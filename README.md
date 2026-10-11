# Contract Intelligence

Upload PDF contracts, ask questions across one or several of them, and get answers that are
**grounded in the source**: each answer shows the contract, page, line numbers and the matched
passage, and clicking a citation highlights those lines in the original PDF.

- [Stack and libraries](#1-stack-and-libraries)
- [Run it locally](#2-run-it-locally)
- [Test it](#3-test-it)
- [How chunking works](#4-how-chunking-works)
- [How retrieval works](#5-how-retrieval-works)
- [LLM logic: real and mock](#6-llm-logic-real-and-mock)
- [API](#7-api) · [Configuration](#8-configuration) · [Design decisions](#9-design-decisions-worth-knowing) · [Known limitations](#10-known-limitations)

```
PDF ─▶ PyMuPDF lines (page, line no., bbox)
       └ scanned document (image-only pages, no text layer)?
            ─▶ Tesseract OCR (auto-orients) ─▶ same lines/bboxes   [text PDFs never get here]
     ─▶ paragraph-aware chunker (~400 tok, overlap, exact offsets + line refs)
     ─▶ LangChain Documents ─▶ SQLite  (+ embeddings via HuggingFaceEmbeddings)

Question ─▶ [condense follow-up] ─▶ per contract, in parallel:
            BM25 + dense cosine ─▶ RRF ─▶ top-k  (LangChain BaseRetriever)
          ─▶ prompt | chat model | parser   (LangChain LCEL; mock if no key)
          ─▶ validate [C#] tags ─▶ narrow each citation to the supporting lines
```

---

## 1. Stack and libraries

| Layer | Library | Used for |
|---|---|---|
| API | **FastAPI** + **uvicorn** | REST endpoints, background indexing, OpenAPI docs at `/docs` |
| Validation / settings | **Pydantic v2**, **pydantic-settings** | Request/response schemas; `.env` configuration |
| Storage | **SQLAlchemy** + **SQLite** | Contracts, pages, lines, chunks (with embeddings), conversations |
| PDF parsing | **PyMuPDF** (`pymupdf`) | Per-line text with page, line number and bounding box |
| OCR | **Tesseract** (bundled in PyMuPDF) | Scanned / image-only PDFs only; needs `eng.traineddata`, see [Scanned PDFs](#scanned-pdfs-ocr) |
| Embeddings | **sentence-transformers** via **langchain-huggingface** | `all-MiniLM-L6-v2`, runs locally, ~90 MB download on first start |
| Retrieval | **NumPy** + own BM25 | Dense cosine, BM25, Reciprocal Rank Fusion |
| Orchestration | **langchain-core** | `Document`, `BaseRetriever`, `Embeddings`, prompt/LCEL chains |
| LLM | **langchain-anthropic** (`ChatAnthropic`) | Real answers when `ANTHROPIC_API_KEY` is set; default model `claude-sonnet-5-5` |
| Frontend | **React 18**, **TypeScript**, **Vite 5** | UI |
| Frontend data | **TanStack Query** | Server state, polling contract status |
| PDF viewer | **react-pdf** | Renders the source PDF with citation highlights |
| Tests | **pytest**, **httpx**, **reportlab** | API/unit tests; generates test and sample PDFs |

Exact backend dependencies are in [backend/requirements.txt](backend/requirements.txt); frontend
dependencies in [frontend/package.json](frontend/package.json).

## 2. Run it locally

**Prerequisites:** Python 3.12 (developed and tested on 3.12), Node.js 18+ (developed on 24).
No API key is required.

Open two terminals, both starting at the repository root.

**Terminal 1: backend**

```bash
cd backend
python -m venv .venv

# activate the virtualenv
.venv\Scripts\Activate.ps1         # Windows PowerShell   (cmd: .venv\Scripts\activate.bat)
source .venv/bin/activate          # macOS / Linux

pip install -r requirements.txt
cp .env.example .env               # Windows cmd: copy .env.example .env   (optional)
uvicorn app.main:app --reload --port 8001
```

**Terminal 2: frontend**

```bash
cd frontend
npm install
npm run dev
```

Then open **http://localhost:5173**.

| URL | What |
|---|---|
| http://localhost:5173 | The app |
| http://localhost:8001/docs | Interactive API docs (Swagger UI) |
| http://localhost:8001/api/health | Active LLM provider and retrieval mode |

> **Ports.** The frontend dev server proxies `/api` to `http://127.0.0.1:8001`
> ([frontend/vite.config.ts](frontend/vite.config.ts)). If you start the backend on a different
> port, change `target` there to match. The backend allows CORS from `localhost:5173`.

**First start:** the backend downloads the embedding model (~90 MB) in the background. The app is
usable immediately; retrieval is BM25-only until the model is loaded. See
[limitations](#10-known-limitations) for what happens if you are offline.

### Scanned PDFs (OCR)

Image-only (scanned) contracts are recognised automatically. OCR is a last resort: a document is
only OCR'd when at least half of its pages are a full-page picture with no text layer. Ordinary
PDFs, including ones with a blank page, a diagram or a logo, are read from their text layer and
**never touch the OCR engine**.

Tesseract itself is bundled inside PyMuPDF (no extra `pip` package and no `tesseract` executable
needed); it only needs the language data file `eng.traineddata`:

```bash
winget install UB-Mannheim.TesseractOCR      # Windows (auto-detected in Program Files)
sudo apt install tesseract-ocr               # Debian / Ubuntu
# or anywhere: put eng.traineddata in a folder and set OCR_TESSDATA_DIR in backend/.env
```

If the language data is missing the app still runs; a scanned file is rejected with a message
saying OCR is unavailable and why. Expect roughly 0.5 s per scanned page; it runs inside the
upload request.

### Using a real LLM

Put your key in `backend/.env` and restart the backend:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The header pill in the UI (and `/api/health`) shows `llm_provider: anthropic`. Without a key it
shows `mock`. To force the mock even when a key is present, set `LLM_PROVIDER=mock`.

## 3. Test it

### Automated (offline)

```bash
cd backend
pytest
```

58 tests across five files. They need no network, model download or API key: the suite forces
`EMBEDDINGS_ENABLED=false`, `LLM_PROVIDER=mock` and a temporary `DATA_DIR`, and uses a fake chat
model and a stub `Embeddings` class for the LangChain paths. Eight OCR tests need Tesseract's
language data and skip themselves when it isn't installed.

| File | Covers |
|---|---|
| `test_extractor.py` | Line/bbox extraction, overprint de-duplication, encrypted PDF rejection; OCR of scanned, rotated and mixed PDFs; text PDFs never reaching OCR |
| `test_chunker.py` | Provenance invariants, overlap, heading breaks, long-paragraph splitting |
| `test_mock_provider.py` | Extractive answers, label-style documents, overview questions, follow-ups |
| `test_langchain_integration.py` | LCEL chains, hybrid retrieval vs BM25, similarity floor, hallucinated tags |
| `test_api.py` | Upload (duplicates, non-PDF), query, citations, history, delete, validation errors |

### Manual walk-through

Two sample contracts are committed in [samples/](samples/) (regenerate with
`python backend/scripts/make_samples.py`).

1. Start both servers (above) and open http://localhost:5173.
2. Drag both PDFs from `samples/` onto the upload area. Each shows *processing*, then *ready*.
3. Select both contracts and ask the questions below. The outputs shown are from the **mock** LLM:

| Question | Expected result |
|---|---|
| *What is the notice period for termination?* | Acme: "…sixty (60) days prior written notice…" (page 2). Initech: "…thirty (30) days written notice…" (page 2) |
| *Which law governs this agreement?* | Acme: State of **New York**. Initech: State of **California** |
| *What is the capital of France?* | Both contracts: "I could not find this in the contract." (no LLM call is made) |
| *Summarize this document* | Each contract's opening passage, cited from page 1 |

4. Click a citation under an answer: the source viewer on the right opens the PDF at that page and
   highlights the cited lines.
5. Ask a follow-up such as *"and for Initech?"*: it is rewritten into a standalone query using the
   conversation so far. The **History** tab lists and reopens past conversations.

The same flow from the command line (bash; on Windows use `curl.exe` or the Swagger UI):

```bash
curl -F "files=@samples/acme_master_services_agreement.pdf" \
     -F "files=@samples/initech_software_license.pdf" http://localhost:8001/api/contracts

curl http://localhost:8001/api/contracts          # wait until each has "status": "ready"

curl -X POST http://localhost:8001/api/chat/query -H "Content-Type: application/json" \
     -d '{"question": "What is the notice period for termination?", "contract_ids": ["<id1>", "<id2>"]}'
```

## 4. How chunking works

Implemented in [chunker.py](backend/app/services/chunker.py), fed by
[pdf_extractor.py](backend/app/services/pdf_extractor.py).

1. **Extract physical lines.** PyMuPDF yields each text line with its page, a 1-based line number
   (restarting every page) and a normalised bounding box. Lines drawn twice at almost the same
   spot (fake bold, shadows; IoU ≥ 0.6 with identical text) are collapsed into one.
2. **Group into paragraphs.** Consecutive lines in the same PyMuPDF block on the same page.
3. **Split over-long paragraphs** (> target) on line boundaries, backing up to the nearest
   sentence end when one is close enough.
4. **Pack paragraphs into chunks** up to the target size, starting a new chunk early at a
   **heading** (a short ≤ 2-line paragraph that is ALL CAPS, or starts with `Article`/`Section`/
   `Clause`/`Schedule`/`Annex`/`Exhibit`/`1.`/`1.2` and has no sentence punctuation), provided the
   current chunk already has ≥ 80 tokens of new content.
5. **Overlap.** The last ~60 tokens (whole lines) of a chunk are repeated at the start of the next
   so a clause cut at a boundary is still retrievable.
6. **Merge a tiny tail.** A final chunk with < 40 tokens of new content is folded into its
   predecessor.

| Setting | Default | Env var |
|---|---|---|
| Target chunk size | 400 tokens | `CHUNK_TARGET_TOKENS` |
| Overlap | 60 tokens | `CHUNK_OVERLAP_TOKENS` |
| Minimum before a heading break | 80 tokens | `CHUNK_MIN_TOKENS` |

Tokens are estimated as ~1.3 × word count to avoid a tokenizer dependency.

**Provenance is exact.** Chunks are always whole, contiguous lines, so
`full_text[char_start:char_end] == chunk.text`. Each chunk stores start/end page and line plus
per-page `line_refs` (line ranges and bounding boxes); a chunk may cross a page boundary. This is
what lets the UI highlight the cited lines in the original PDF.

## 5. How retrieval works

Implemented in [retriever.py](backend/app/services/retriever.py); orchestrated in
[rag.py](backend/app/services/rag.py).

**Indexing** (background task after upload): chunks are embedded (normalised MiniLM vectors, stored
as float32 bytes in SQLite) and the contract flips from `processing` to `ready`. BM25 statistics
are built in memory per contract on first query and cached until the contract changes.

**Querying**, independently for each selected contract and in parallel:

1. **Follow-up rewrite.** If the conversation has history, the question is rewritten into a
   standalone query (last 6 turns). Skipped for overview questions.
2. **BM25** over the contract's chunks (lower-cased, stop-words removed; `k1=1.5`, `b=0.75`).
3. **Dense search**: cosine similarity between the query and chunk embeddings. Hits below
   **0.15** are discarded as noise.
4. **Reciprocal Rank Fusion** (`k=60`) merges the two rankings; the top **k** chunks are kept
   (default 5, range 1–10, from the top 20 candidates of each ranker).
5. **No match → no LLM call.** A chunk that matches neither lexically nor above the similarity
   floor is dropped; if nothing survives, the answer is "I could not find this in the contract."
6. **Overview questions** ("summarize this", "what is this document about?") are detected by
   pattern ([intent.py](backend/app/services/intent.py)) and answered from the document's first two
   chunks, since they have no overlap with any particular passage.

**Grounding the answer.** Retrieved chunks are tagged `[C1]`, `[C2]`… and passed to the model,
which must cite those tags. Afterwards:

- Tags the model invented are removed; `[C1, C2]` is normalised to `[C1][C2]`.
- If the model cited nothing valid, the best chunk is attached as an *uncited* fallback citation.
- Each citation is **narrowed** from the whole chunk (up to ~25 lines, maybe two pages) to the
  tightest run of lines supporting the claim: a line counts as supported when ≥ 50 % of its words
  appear in the answer sentence or question, and the best contiguous run (≤ 8 lines) is cited.
- `score` on a citation is the real cosine similarity (or normalised BM25 when embeddings are
  unavailable), not a rank value; it is `null` for overview answers.

When more than one contract is selected, a short **comparison** of the per-contract answers is
generated (by the LLM, or by a counting template in mock mode).

## 6. LLM logic: real and mock

The provider is chosen once at startup in [llm/factory.py](backend/app/llm/factory.py):

| `LLM_PROVIDER` | Key set | Result |
|---|---|---|
| `auto` (default) | yes | Anthropic via LangChain |
| `auto` | no | Mock |
| `anthropic` | yes | Anthropic via LangChain |
| `anthropic` | no | Mock (a warning is logged) |
| `mock` | any | Mock |

Both implement the same three-method interface (`condense_question`, `generate`, `compare`), so
the rest of the app doesn't know which is active.

### Real: LangChain chains

[langchain_provider.py](backend/app/llm/langchain_provider.py) builds three LCEL chains
(`prompt | chat model | StrOutputParser`): **answer**, **condense** (≤ 150 tokens) and **compare**
(≤ 400 tokens). The answer prompt instructs the model to use *only* the numbered excerpts, cite
every claim with its tag, quote figures and dates exactly, and reply with the exact not-found
sentence when the excerpts don't contain the answer. Defaults: `claude-sonnet-5-5`, 1024 max
output tokens, 2 retries. A failure in one contract's answer is reported on that contract only.
To use another vendor, build its LangChain chat model in `llm/factory.py` and wrap it in
`LangChainProvider`.

### Mock: deterministic and extractive

[mock_provider.py](backend/app/llm/mock_provider.py) is used when no key is configured. It
**cannot write or reason**; it selects and quotes the most relevant text from the retrieved
passages, always with a valid `[C#]` tag, so retrieval, citations, highlighting and the
multi-contract view all work end to end. For a given input it always returns the same output.

1. **Split the context into units.** Prose passages become sentences (lines are re-joined first,
   since PDF lines wrap mid-sentence). Label-style passages (certificates, forms, cover pages,
   where < 25 % of lines end in punctuation) become one unit per line. Duplicates from
   overlapping chunks are removed.
2. **Overview question →** the first 6 units, prefixed "This document appears to cover:".
3. **Otherwise, score each unit:** `cosine(question, unit)` (0 if embeddings are unavailable)
   `+ 0.15 × shared word stems` (6-character prefixes, ignoring generic words such as
   "contract"/"agreement") `− 0.001 × position`. Units with no shared stem and cosine < 0.25
   are dropped.
4. **Pick the answer:** up to 3 units scoring within 70 % of the best. If a picked unit is a
   section header (ends with `:`, or ≤ 4 words without digits or sentence end), the 3 lines
   beneath it are included, since the label alone isn't an answer.
5. **Format** as `Based on the contract:` followed by `- <quote> [C#]` bullets, or return the
   not-found sentence if nothing scored.
6. **Follow-ups:** a very short or elliptical question ("and for Initech?") is prefixed with the
   previous user question; a standalone question is left alone.
7. **Compare:** "Relevant content was found in *N* of *M* contracts." plus the names with no
   supporting content.

Example (mock, *"Which law governs this agreement?"* on the Acme sample):

```
Based on the contract:
- GOVERNING LAW This Agreement is governed by the laws of the State of New York, without regard
  to its conflict of laws principles. [C1]
- Each Statement of Work shall form part of this Agreement and be governed by its terms. [C2]
```

Because it is extractive it can return partial fragments on messy layouts and it won't join
facts across passages. Set `ANTHROPIC_API_KEY` for real answers.

## 7. API

All routes are under `/api`. Interactive docs: `/docs`.

| Method | Path | Notes |
|---|---|---|
| POST | `/contracts` | Multipart `files` (up to 10 PDFs, 25 MB each). Per-file result `accepted` / `duplicate` / `rejected`, so one bad file doesn't fail the batch. Duplicates are detected by SHA-256 |
| GET | `/contracts` | List. Poll `status`: `processing` → `ready` (or `failed`) |
| GET | `/contracts/{id}` | Metadata |
| GET | `/contracts/{id}/file` | Original PDF |
| GET | `/contracts/{id}/pages/{n}` | Page text with numbered lines and bboxes |
| DELETE | `/contracts/{id}` | Removes the contract, its chunks and the file |
| POST | `/chat/query` | `{question, contract_ids[1..10], conversation_id?, top_k?}` → one grounded answer per contract, plus `comparison` when > 1 |
| GET | `/conversations` | Conversation summaries |
| GET | `/conversations/{id}` | Full conversation with stored answers |
| DELETE | `/conversations/{id}` | Delete a conversation |
| GET | `/health` | Active LLM provider, embeddings status, retrieval mode |

Errors use one shape: `{"error": {"code", "message", "details"}}`.
Querying a contract that isn't `ready` returns `409 contract_not_ready`.

## 8. Configuration

Set in `backend/.env` (copy from [.env.example](backend/.env.example)) or as environment variables.
All are optional.

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | Enables the real LLM |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Model used when a key is set |
| `LLM_PROVIDER` | `auto` | `auto` \| `anthropic` \| `mock` |
| `LLM_MAX_TOKENS` | `1024` | Max output tokens for answers |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model |
| `EMBEDDINGS_ENABLED` | `true` | `false` = BM25-only, no model download |
| `RETRIEVAL_TOP_K` | `5` | Default chunks per contract |
| `RETRIEVAL_CANDIDATES` | `20` | Candidates taken from each ranker before fusion |
| `RETRIEVAL_MIN_SIMILARITY` | `0.15` | Cosine floor for dense hits |
| `CHUNK_TARGET_TOKENS` / `CHUNK_OVERLAP_TOKENS` / `CHUNK_MIN_TOKENS` | `400` / `60` / `80` | See [chunking](#4-how-chunking-works) |
| `OCR_ENABLED` | `true` | `false` = never OCR; scanned PDFs are rejected |
| `OCR_LANGUAGE` | `eng` | Tesseract language code(s), e.g. `eng+deu` |
| `OCR_TESSDATA_DIR` | *(auto-detect)* | Folder holding `<lang>.traineddata` |
| `OCR_DPI` | `300` | Render resolution for OCR; 150 is faster but loses small print |
| `OCR_MIN_CHARS_PER_PAGE` | `20` | A page with fewer extracted characters has no usable text layer |
| `OCR_MAX_PAGES` | `150` | Max pages OCR'd per document (bounds upload time) |
| `MAX_UPLOAD_MB` | `25` | Per-file upload limit |
| `MAX_FILES_PER_UPLOAD` | `10` | Files per upload request |
| `CORS_ORIGINS` | `["http://localhost:5173","http://127.0.0.1:5173"]` | JSON list |
| `DATA_DIR` | `backend/data` | SQLite database and uploaded PDFs |

## 9. Design decisions worth knowing

- **Custom extraction/chunking, LangChain for the rest.** Page/line/bbox provenance is the
  product's core, so it is built directly on PyMuPDF line geometry and carried through as
  `Document.metadata`. LangChain provides embeddings, the retriever interface and the LLM chain,
  so swapping vendors is a change in `llm/factory.py` only.
- **OCR only for scanned documents.** A page is "image-only" when it has fewer than
  `OCR_MIN_CHARS_PER_PAGE` extracted characters *and* pictures cover at least 70% of it. A
  document is "scanned" when at least half its pages are image-only. Only then does OCR run, and
  only on those pages. So a text PDF with a blank page, a figure or a logo page is never
  OCR'd, and a searchable scan (one with a text layer already) is read from that layer. Both
  checks are cheap: a page with real text skips the image test, and a text PDF never resolves
  language data or renders a page. Recognised lines go through the same line/bbox pipeline, so
  citations highlight the scanned page like any other. Pages that come out as noise
  (sideways / upside-down scans) are retried at other orientations and the most plausible
  reading wins. Measured on the sample contracts at 300 DPI: ~0.5 s/page, 99.9-100% of
  characters recovered, including a JPEG-compressed, slightly rotated scan.
- **Boxes are mapped through the page rotation.** PyMuPDF reports boxes in unrotated page
  space, so for `/Rotate` pages (typical of landscape scans) they are transformed to the page as
  displayed before being normalised; reading order follows the text direction.
- **Why Tesseract and not a pure-pip engine.** RapidOCR (ONNX) was tried first: it installs
  with pip but dropped whole lines of clean 300 DPI text (up to ~40% of a page, unpredictably)
  and ran words together in all-caps headings. For contracts, a silently missing date or fee is
  worse than a slower, installable dependency.
- **Own BM25 instead of `BM25Retriever`.** `rank_bm25` yields zero/negative IDF on tiny corpora
  (a one-page contract is a single chunk), silently dropping every match. `retriever.BM25` uses a
  non-negative IDF; this is covered by tests.
- **Similarity floor.** Dense search always returns *something*, so hits below cosine 0.15 are
  dropped (measured on MiniLM: relevant ≥ 0.29, unrelated ≤ 0.08). Unrelated questions return
  "not found" without an LLM call.
- **Citations are narrowed.** A chunk can span ~25 lines or two pages, so each citation is
  tightened to the (≤ 8) lines supporting the cited claim. Tags the model invents are discarded.
- **Per-contract answers.** Each selected contract is retrieved and answered independently and in
  parallel, so every answer is attributable and one failure doesn't sink the rest.
- **Graceful degradation.** No API key → mock LLM. No embedding model → BM25-only. The app runs
  end to end in both cases.

## 10. Known limitations

- Encrypted PDFs are rejected.
- OCR accuracy depends on scan quality, so check contract numbers, dates and amounts on poor
  scans against the highlighted page. The UI does not yet flag which contracts were OCR'd.
- OCR runs inside the upload request (~0.5 s/page, capped by `OCR_MAX_PAGES`), so large scans
  make the upload slow. Moving extraction to the background indexing job is the next step.
- A scan that also carries a digital text stamp of 20+ characters on every page is treated as a
  text PDF and not OCR'd. A text PDF with only a *minority* of scanned pages (under half) keeps
  those pages unread.
- Orientation fixing for a non-English `OCR_LANGUAGE` only catches sideways pages, not
  upside-down ones. English handles all four orientations.
- Handwriting and very low-resolution (< 150 DPI) scans are not supported.
- Multi-column layouts follow PyMuPDF's reading order, which can interleave columns.
- If the embedding model can't be loaded (e.g. first run while offline), contracts uploaded in
  that window are indexed **without vectors** and stay BM25-only, even after the model becomes
  available. Re-upload them (delete first) once the model has downloaded.
- Documents uploaded before overprint de-duplication was added keep the duplicate lines;
  delete and re-upload them.
- No authentication or multi-user separation; SQLite + local disk suit a demo, not production.
- Only PDF is supported; the mock LLM is extractive only (see above).
