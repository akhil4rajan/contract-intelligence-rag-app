"""Rule-based term extraction for the key-free mock provider.

Like the rest of the mock it cannot understand a contract, but it finds the usual phrasings
("effective as of January 1, 2025", "Acme Corporation ... (Provider)", "USD 120,000") with
regular expressions and quotes the exact lines it read, so the whole extraction flow -
citations, highlighting, caching - is demonstrable without an API key.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

from app.llm.base import ContextChunk
from app.schemas import DateDraft, Evidence, RateDraft, TermsDraft, TextDraft

_MAX_QUOTE_LINES = 3
_MAX_RATES = 15
_MAX_RATE_SENTENCE = 220

# ---- dates -------------------------------------------------------------------------------

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_MONTH_NUM = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_DATE_RE = re.compile(
    rf"\b(?P<m1>{_MONTH})\.?\s+(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y1>\d{{4}})\b"
    rf"|\b(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?P<m2>{_MONTH})\.?,?\s+(?P<y2>\d{{4}})\b"
    r"|\b(?P<y3>\d{4})-(?P<m3>\d{2})-(?P<d3>\d{2})\b",
    re.I,
)
_START_CUE = re.compile(
    r"\b(?:effective|commenc\w*|start\w*|begin\w*|as of|dated|entered into|made on|from)\b", re.I
)
_END_CUE = re.compile(r"\b(?:expir\w*|terminat\w*|end(?:s|ing|ed)?|until|through)\b", re.I)
_CUE_WINDOW = 60

# "initial term of three (3) years", "period of 12 months" - but not a "renewal term".
_TERM_RE = re.compile(
    r"(?<!renewal )\b(?:initial\s+)?(?:term|period)\s+of\s+(?:[a-z]+(?:-[a-z]+)?\s+)?"
    r"\(?(?P<n>\d{1,3})\)?\s*[- ]?(?P<unit>year|month)s?\b",
    re.I,
)

# ---- supplier ----------------------------------------------------------------------------

_ROLE = r"(?:supplier|service provider|provider|vendor|licensor|contractor|seller|consultant|lessor|developer)"
_QUOTES = "\"'“”‘’"
_ROLE_MARK = re.compile(
    rf"\(\s*(?:hereinafter\s+)?(?:referred\s+to\s+as\s+)?(?:the\s+|a\s+)?[{_QUOTES}]*{_ROLE}[{_QUOTES}]*\s*\)",
    re.I,
)
_ROLE_LABEL = re.compile(rf"\b{_ROLE}(?:\s+name)?\s*[:\-–]\s*(?P<n>[^;(]{{2,80}})", re.I)
_DESCRIPTOR = re.compile(
    r",\s+(?:a|an|the|organi[sz]ed|incorporated|registered|existing|having|with|located|duly)\b|\s+\(",
    re.I,
)
_ABBREV = frozenset({"inc.", "ltd.", "co.", "corp.", "llc.", "l.l.c.", "s.a.", "plc.", "l.p.", "n.v."})

# ---- rates -------------------------------------------------------------------------------

_CURRENCY = r"(?:USD|EUR|GBP|CAD|AUD|CHF|JPY|INR|US\$|\$|£|€|₹)"
_AMOUNT_RE = re.compile(
    rf"(?<![\w$])(?:{_CURRENCY}\s?\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s?(?:million|billion|k)\b)?"
    rf"|\d(?:[\d,]*\d)?(?:\.\d+)?\s?{_CURRENCY}(?![\w$]))",
    re.I,
)
_PERCENT_RATE_RE = re.compile(
    r"\d+(?:\.\d+)?\s?%\)?(?=\s*(?:per|a|each|every)\s+(?:month|year|annum|week|day|quarter))", re.I
)
_UNIT_RE = re.compile(
    r"\b(?:(?:per|a|each|every)\s+(?:hour|day|week|month|quarter|year|annum|user|seat|licen[cs]e|unit|"
    r"transaction)|hourly|daily|weekly|monthly|quarterly|annual(?:ly)?|p\.a\.)\b",
    re.I,
)
_FEE_CUE = re.compile(r"\b(?:fees?|rates?|prices?|pric\w+|charges?|costs?|pay\w*|invoice\w*|interest)\b", re.I)
_NOT_A_RATE = re.compile(
    r"\b(?:liabilit\w*|shall not exceed|indemnif\w*|insur\w*|damages|penalt\w*|deposit)\b", re.I
)

# A section heading that ran into the sentence: "2. FEES AND PAYMENT Customer shall ..."
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(?:[A-Z][A-Z&/,'-]+\s+)+(?=[A-Z][a-z])")

_SENT_BREAK = re.compile(r'(?<=[.!?;])\s+(?=[A-Z0-9"“])')


@dataclass
class _Doc:
    tag: str
    joined: str  # physical lines re-joined with single spaces
    spans: list[tuple[int, int, str]]  # (start, end, text) of each line inside `joined`

    @classmethod
    def of(cls, chunk: ContextChunk) -> "_Doc":
        lines = [ln.strip() for ln in chunk.text.split("\n") if ln.strip()]
        spans: list[tuple[int, int, str]] = []
        pos = 0
        for ln in lines:
            spans.append((pos, pos + len(ln), ln))
            pos += len(ln) + 1
        return cls(chunk.tag, " ".join(lines), spans)

    def quote(self, a: int, b: int) -> str:
        """The physical lines overlapping [a, b): a verbatim, tightly scoped quote."""
        hit = [t for s, e, t in self.spans if s < b and e > a] or [self.spans[0][2]]
        return " ".join(hit[:_MAX_QUOTE_LINES])

    def sentence(self, a: int, b: int) -> str:
        start, end = 0, len(self.joined)
        for m in _SENT_BREAK.finditer(self.joined):
            if m.end() <= a:
                start = m.end()
            elif m.start() >= b:
                end = m.start()
                break
        return self.joined[start:end].strip()

    def evidence(self, a: int, b: int) -> Evidence:
        return Evidence(tag=self.tag, quote=self.quote(a, b))


def _add_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    for day in (d.day, 30, 29, 28):  # clamp e.g. Jan 31 + 1 month
        try:
            return date(year, month, day)
        except ValueError:
            continue
    return date(year, month, 28)


def _parse_dates(doc: _Doc) -> list[tuple[int, int, date]]:
    out: list[tuple[int, int, date]] = []
    for m in _DATE_RE.finditer(doc.joined):
        g = m.groupdict()
        try:
            if g["y1"]:
                d = date(int(g["y1"]), _MONTH_NUM[g["m1"][:3].lower()], int(g["d1"]))
            elif g["y2"]:
                d = date(int(g["y2"]), _MONTH_NUM[g["m2"][:3].lower()], int(g["d2"]))
            else:
                d = date(int(g["y3"]), int(g["m3"]), int(g["d3"]))
        except ValueError:
            continue
        out.append((m.start(), m.end(), d))
    return out


def _date_role(doc: _Doc, start: int) -> str | None:
    """"start" or "end", from the nearest cue word just before the date."""
    window = doc.joined[max(0, start - _CUE_WINDOW) : start]
    cues = [(m.end(), "start") for m in _START_CUE.finditer(window)]
    cues += [(m.end(), "end") for m in _END_CUE.finditer(window)]
    return max(cues)[1] if cues else None


def _find_dates(docs: list[_Doc]) -> tuple[DateDraft | None, DateDraft | None]:
    start: DateDraft | None = None
    end: DateDraft | None = None
    for doc in docs:
        for a, b, d in _parse_dates(doc):
            role = _date_role(doc, a)
            if role == "start" and start is None:
                start = DateDraft(value=d, evidence=doc.evidence(a, b))
            elif role == "end" and end is None:
                end = DateDraft(value=d, evidence=doc.evidence(a, b))

    if end is None and start is not None and start.value is not None:
        for doc in docs:
            m = _TERM_RE.search(doc.joined)
            if not m:
                continue
            n = int(m.group("n"))
            months = n * 12 if m.group("unit").lower() == "year" else n
            if months <= 0:
                continue
            last_day = _add_months(start.value, months) - timedelta(days=1)
            unit = m.group("unit").lower() + ("s" if n != 1 else "")
            end = DateDraft(
                value=last_day,
                basis=f"Calculated from the stated term of {n} {unit}, ending the day before "
                "the anniversary of the start date.",
                evidence=doc.evidence(m.start(), m.end()),
            )
            break
    return start, end


def _supplier_from_marker(doc: _Doc, idx: int) -> str | None:
    seg = doc.joined[max(0, idx - 220) : idx]
    seg = re.split(r"\)\s*,?\s*(?:and|&)\s+", seg)[-1]  # drop an earlier "(Customer) and"
    parts = re.split(r"\bbetween\b", seg, flags=re.I)
    if len(parts) > 1:
        seg = parts[-1]
    else:  # no "between": take the trailing run of capitalised words
        words: list[str] = []
        for w in reversed(seg.split()):
            ok = re.fullmatch(r"[A-Z][\w&.'’,-]*", w) or w in ("of", "&")
            if not ok or (words and w.endswith(".") and w.lower() not in _ABBREV):
                break
            words.append(w)
            if len(words) >= 8:
                break
        while words and words[-1] in ("of", "&"):
            words.pop()
        seg = " ".join(reversed(words))
    return _clean_name(seg)


def _clean_name(text: str) -> str | None:
    name = _DESCRIPTOR.split(text)[0].strip(" ,;:-–")
    if not re.search(r"[A-Za-z]", name) or not 2 <= len(name) <= 80 or len(name.split()) > 8:
        return None
    return name


def _find_supplier(docs: list[_Doc]) -> TextDraft | None:
    for doc in docs:
        for m in _ROLE_MARK.finditer(doc.joined):
            name = _supplier_from_marker(doc, m.start())
            if name:
                start = max(0, doc.joined.rfind(name, 0, m.start()))
                return TextDraft(value=name, evidence=doc.evidence(start, m.end()))
    for doc in docs:
        for s, e, line in doc.spans:
            m = _ROLE_LABEL.search(line)
            if m and (name := _clean_name(m.group("n"))):
                return TextDraft(value=name, evidence=doc.evidence(s, e))
    return None


def _find_rates(docs: list[_Doc]) -> list[RateDraft]:
    out: list[RateDraft] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        matches = list(_AMOUNT_RE.finditer(doc.joined)) + list(_PERCENT_RATE_RE.finditer(doc.joined))
        for m in sorted(matches, key=lambda x: x.start()):
            sentence = _HEADING.sub("", doc.sentence(m.start(), m.end()))
            if _NOT_A_RATE.search(sentence) or not _FEE_CUE.search(sentence):
                continue
            amount = m.group(0).strip().rstrip(")")
            tail = doc.joined[m.end() : m.end() + 60]
            unit_m = _UNIT_RE.search(tail) or _UNIT_RE.search(sentence)
            unit = unit_m.group(0).lower() if unit_m else None
            key = (re.sub(r"\s+", "", amount.lower()), unit or "")
            if key in seen:
                continue
            seen.add(key)
            description = sentence
            if len(description) > _MAX_RATE_SENTENCE:
                description = description[: _MAX_RATE_SENTENCE - 1].rstrip() + "…"
            out.append(
                RateDraft(
                    description=description,
                    amount=amount,
                    unit=unit,
                    evidence=doc.evidence(m.start(), m.end()),
                )
            )
            if len(out) >= _MAX_RATES:
                return out
    return out


def extract_terms_heuristic(context: list[ContextChunk]) -> TermsDraft:
    docs = [_Doc.of(c) for c in context if c.text.strip()]
    start, end = _find_dates(docs)
    return TermsDraft(
        start_date=start,
        end_date=end,
        supplier_name=_find_supplier(docs),
        rates=_find_rates(docs),
    )
