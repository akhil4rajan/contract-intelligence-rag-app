from app.services.chunker import chunk_lines, estimate_tokens
from app.services.pdf_extractor import ExtractedLine


def _doc(paragraphs_per_page: list[list[str]], words_per_line: int = 12):
    """Synthesize extracted lines: paragraph = one block, wrapped into lines of N words."""
    lines: list[ExtractedLine] = []
    cursor = 0
    for page_no, paragraphs in enumerate(paragraphs_per_page, start=1):
        line_no = 0
        for block, para in enumerate(paragraphs):
            words = para.split()
            for i in range(0, len(words), words_per_line):
                text = " ".join(words[i : i + words_per_line])
                line_no += 1
                lines.append(
                    ExtractedLine(
                        page=page_no,
                        line_number=line_no,
                        text=text,
                        bbox=[0.1, 0.1, 0.9, 0.12],
                        page_width=612,
                        page_height=792,
                        block=block,
                        char_start=cursor,
                        char_end=cursor + len(text),
                    )
                )
                cursor += len(text) + 1
    return lines


def _para(n_words: int, seed: str = "term") -> str:
    return " ".join(f"{seed}{i}" for i in range(n_words)) + "."


def test_chunks_respect_token_budget_and_cover_everything():
    lines = _doc([[_para(120, f"p{i}") for i in range(8)], [_para(90, f"q{i}") for i in range(5)]])
    chunks = chunk_lines(lines, target_tokens=400, overlap_tokens=60, min_tokens=80)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 400 * 1.25 + 20
    covered = {(r["page"], n) for c in chunks for r in c.line_refs for n in range(r["line_start"], r["line_end"] + 1)}
    assert covered == {(l.page, l.line_number) for l in lines}


def test_offsets_round_trip_and_line_refs_match():
    lines = _doc([[_para(150, "a"), _para(60, "b")], [_para(200, "c")]])
    full_text = "\n".join(l.text for l in lines)
    chunks = chunk_lines(lines, target_tokens=200, overlap_tokens=30, min_tokens=40)

    for c in chunks:
        assert full_text[c.char_start : c.char_end] == c.text
        assert (c.start_page, c.start_line) == (c.line_refs[0]["page"], c.line_refs[0]["line_start"])
        assert (c.end_page, c.end_line) == (c.line_refs[-1]["page"], c.line_refs[-1]["line_end"])
        for ref in c.line_refs:
            assert len(ref["bboxes"]) == ref["line_end"] - ref["line_start"] + 1


def test_chunk_can_span_pages_with_both_page_refs():
    lines = _doc([[_para(40, "x")], [_para(40, "y")]])
    chunks = chunk_lines(lines, target_tokens=400)
    assert len(chunks) == 1
    assert [r["page"] for r in chunks[0].line_refs] == [1, 2]
    assert chunks[0].start_page == 1 and chunks[0].end_page == 2


def test_heading_starts_new_chunk_when_current_is_large_enough():
    body = _para(100, "b")
    lines = _doc([[body, "ARTICLE 2 TERMINATION", body]])
    chunks = chunk_lines(lines, target_tokens=400, overlap_tokens=0, min_tokens=80)
    assert len(chunks) == 2
    assert chunks[1].text.startswith("ARTICLE 2 TERMINATION")


def test_empty_input_and_token_estimate():
    assert chunk_lines([]) == []
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") >= 3
