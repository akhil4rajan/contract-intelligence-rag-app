"""Paragraph-aware chunking that keeps exact line/page/offset provenance.

Chunks are always made of whole, contiguous physical lines, so
``full_text[chunk.char_start:chunk.char_end] == chunk.text`` and every chunk maps
back to precise (page, line) ranges and bounding boxes.
"""

import re
from dataclasses import dataclass
from itertools import groupby

from app.services.pdf_extractor import ExtractedLine

_SENTENCE_END = re.compile(r"[.;:!?)\"']\s*$")
_HEADING_NUMBERING = re.compile(r"^(article|section|clause|schedule|annex|exhibit)\b|^\d+(\.\d+)*[.)]?\s", re.I)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per word) to avoid a tokenizer dependency."""
    words = len(text.split())
    return max(1, round(words * 1.3)) if words else 0


@dataclass
class ChunkDraft:
    index: int
    text: str
    token_count: int
    start_page: int
    start_line: int
    end_page: int
    end_line: int
    char_start: int
    char_end: int
    line_refs: list[dict]


def _paragraphs(lines: list[ExtractedLine]) -> list[list[ExtractedLine]]:
    return [list(group) for _, group in groupby(lines, key=lambda l: (l.page, l.block))]


def _tokens(lines: list[ExtractedLine]) -> int:
    return sum(estimate_tokens(l.text) for l in lines)


def _is_heading(par: list[ExtractedLine]) -> bool:
    if len(par) > 2:
        return False
    text = " ".join(l.text for l in par)
    words = text.split()
    if not words or len(words) > 14:
        return False
    if text.isupper() and len(words) <= 10:
        return True
    return bool(_HEADING_NUMBERING.match(text)) and not _SENTENCE_END.search(text)


def _build_refs(lines: list[ExtractedLine]) -> list[dict]:
    refs: list[dict] = []
    for page, group in groupby(lines, key=lambda l: l.page):
        g = list(group)
        refs.append(
            {
                "page": page,
                "line_start": g[0].line_number,
                "line_end": g[-1].line_number,
                "bboxes": [l.bbox for l in g],
            }
        )
    return refs


def _split_long(par: list[ExtractedLine], target: int) -> list[list[ExtractedLine]]:
    """Split an over-long paragraph on line boundaries, preferring sentence ends."""
    pieces: list[list[ExtractedLine]] = []
    current: list[ExtractedLine] = []
    tokens = 0
    for line in par:
        t = estimate_tokens(line.text)
        if current and tokens + t > target:
            # back up to the last sentence boundary if it is not too far back
            cut = len(current)
            for i in range(len(current) - 1, len(current) // 2, -1):
                if _SENTENCE_END.search(current[i].text):
                    cut = i + 1
                    break
            pieces.append(current[:cut])
            current = current[cut:]
            tokens = _tokens(current)
        current.append(line)
        tokens += t
    if current:
        pieces.append(current)
    return pieces


def chunk_lines(
    lines: list[ExtractedLine],
    *,
    target_tokens: int = 400,
    overlap_tokens: int = 60,
    min_tokens: int = 80,
) -> list[ChunkDraft]:
    if not lines:
        return []

    units: list[list[ExtractedLine]] = []
    for par in _paragraphs(lines):
        if _tokens(par) > target_tokens:
            units.extend(_split_long(par, target_tokens))
        else:
            units.append(par)

    # Each group: (lines, index where non-overlap lines begin)
    groups: list[tuple[list[ExtractedLine], int]] = []
    current: list[ExtractedLine] = []
    new_start = 0
    tokens = 0

    def flush() -> None:
        nonlocal current, new_start, tokens
        if len(current) > new_start:
            groups.append((current, new_start))
        # carry the tail of the finished chunk as overlap into the next one
        tail: list[ExtractedLine] = []
        acc = 0
        for line in reversed(current):
            t = estimate_tokens(line.text)
            if acc + t > overlap_tokens:
                break
            tail.insert(0, line)
            acc += t
        current = tail
        new_start = len(tail)
        tokens = acc

    for unit in units:
        unit_tokens = _tokens(unit)
        new_tokens = tokens  # includes overlap
        heading_break = _is_heading(unit) and (new_tokens - _tokens(current[:new_start])) >= min_tokens
        if current and len(current) > new_start and (new_tokens + unit_tokens > target_tokens or heading_break):
            flush()
        current.extend(unit)
        tokens += unit_tokens
    if len(current) > new_start:
        groups.append((current, new_start))

    # Merge a tiny trailing chunk (mostly overlap) back into its predecessor.
    if len(groups) > 1:
        last_lines, last_new = groups[-1]
        added = last_lines[last_new:]
        if _tokens(added) < min_tokens // 2:
            prev_lines, prev_new = groups[-2]
            if _tokens(prev_lines) + _tokens(added) <= target_tokens * 1.25:
                groups[-2] = (prev_lines + added, prev_new)
                groups.pop()

    drafts: list[ChunkDraft] = []
    for idx, (group_lines, _) in enumerate(groups):
        first, last = group_lines[0], group_lines[-1]
        text = "\n".join(l.text for l in group_lines)
        drafts.append(
            ChunkDraft(
                index=idx,
                text=text,
                token_count=estimate_tokens(text),
                start_page=first.page,
                start_line=first.line_number,
                end_page=last.page,
                end_line=last.line_number,
                char_start=first.char_start,
                char_end=last.char_end,
                line_refs=_build_refs(group_lines),
            )
        )
    return drafts
