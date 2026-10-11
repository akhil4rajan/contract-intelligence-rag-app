"""Tiny, dependency-free question-intent helpers."""

import re

_DOC = r"(?:document|doc|contract|agreement|file|pdf|certificate|paper|text)"
_END = r"\s*(?:about|for|all about)?\s*[?.!]*\s*$"

_OVERVIEW_PATTERNS = [
    re.compile(r"\b(?:summari[sz]e|summary|overview|tl;?\s?dr)\b", re.I),
    re.compile(rf"\bwhat(?:'s|\s+is|\s+does)\s+(?:this|the|that)\s+(?:\w+\s+){{0,2}}?{_DOC}{_END}", re.I),
    re.compile(
        rf"\bwhat\s+does\s+(?:this|the|that)\s+(?:\w+\s+){{0,2}}?{_DOC}\s+(?:say|cover|contain|include)"
        r"\s*[?.!]*\s*$",
        re.I,
    ),
    re.compile(r"\bwhat(?:'s|\s+is)\s+(?:this|it)\s+about\b", re.I),
    re.compile(
        rf"\b(?:tell\s+me\s+about|describe|explain)\s+(?:this|the)\s+(?:\w+\s+){{0,2}}?{_DOC}\s*[?.!]*\s*$",
        re.I,
    ),
]


def is_overview_question(question: str) -> bool:
    """True for 'what is this document about / summarise it' style questions.

    These have no lexical or semantic overlap with any one passage, so retrieval by
    similarity rightly finds nothing; they are answered from the document's opening.
    """
    q = question.strip()
    return any(p.search(q) for p in _OVERVIEW_PATTERNS)
