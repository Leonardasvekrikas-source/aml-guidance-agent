"""PDF text extraction, page by page.

Page numbers are preserved because a citation that cannot name a page is not
checkable by a human, and the point of this project is that citations are
checkable. Section headings are extracted on a best-effort basis: they improve
retrieval when present and are simply absent when the heuristic finds nothing,
which is honest and does not corrupt anything downstream.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Hyphen at a line break: "money laun-\ndering" -> "money laundering".
DEHYPHENATE = re.compile(r"(\w)-\n(\w)")
# Runs of whitespace, but never across a paragraph break.
COLLAPSE_SPACES = re.compile(r"[ \t\u00a0]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")

# Headings in this corpus are overwhelmingly numbered sections or short
# all-caps/title-case lines. Anything longer than this is prose, not a heading.
MAX_HEADING_CHARS = 90
NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+(\S.{2,%d})$" % MAX_HEADING_CHARS)
CAPS_HEADING = re.compile(r"^\s*([A-Z][A-Z0-9 ,&/\u2013\u2014'()-]{5,%d})\s*$" % MAX_HEADING_CHARS)

# Page furniture that adds nothing and pollutes lexical retrieval.
BOILERPLATE = re.compile(
    r"^\s*(page\s+\d+(\s+of\s+\d+)?|\d+\s*\|\s*.{0,60}|\u00a9.{0,80})\s*$",
    re.IGNORECASE,
)


@dataclass
class Page:
    number: int          # 1-based, as a reader would cite it
    text: str
    heading: str | None  # most recent heading at or before this page


class ExtractionError(RuntimeError):
    pass


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\u00ad", "")           # soft hyphens
    text = DEHYPHENATE.sub(r"\1\2", text)
    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if not BOILERPLATE.match(line)]
    text = "\n".join(lines)
    text = COLLAPSE_SPACES.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def find_heading(page_text: str) -> str | None:
    """Return the first plausible section heading on a page, or None.

    Deliberately conservative. A wrong heading attached to a chunk is worse
    than no heading, because it is displayed next to a citation and a reader
    will believe it.
    """
    for line in page_text.split("\n")[:25]:
        stripped = line.strip()
        if not stripped or len(stripped) > MAX_HEADING_CHARS:
            continue
        if stripped.endswith((".", ";", ",")):
            continue  # sentences are not headings

        numbered = NUMBERED_HEADING.match(stripped)
        if numbered:
            return f"{numbered.group(1)} {numbered.group(2).strip()}"

        caps = CAPS_HEADING.match(stripped)
        if caps:
            candidate = caps.group(1).strip()
            letters = [c for c in candidate if c.isalpha()]
            if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
                return candidate
    return None


def extract_pages(pdf_path, min_chars: int = 40) -> list[Page]:
    """Extract cleaned text per page.

    Pages with almost no extractable text are dropped. That is usually a scanned
    image or a cover page; either way it contributes nothing but would dilute
    BM25 statistics if kept.
    """
    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError, ValueError) as exc:
        raise ExtractionError(f"{pdf_path}: cannot read PDF ({exc})") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # many public PDFs carry an empty owner password
        except Exception as exc:
            raise ExtractionError(f"{pdf_path}: encrypted and cannot be opened") from exc

    pages: list[Page] = []
    current_heading: str | None = None

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            # One unreadable page should not lose the other three hundred.
            continue

        text = clean_text(raw)
        if len(text) < min_chars:
            continue

        heading = find_heading(text)
        if heading:
            current_heading = heading

        pages.append(Page(number=index, text=text, heading=current_heading))

    if not pages:
        raise ExtractionError(
            f"{pdf_path}: no extractable text on any page. This is almost "
            "certainly a scanned document with no text layer; it needs OCR or "
            "it needs to be dropped from the manifest."
        )

    return pages


def page_count(pdf_path) -> int:
    return len(PdfReader(str(pdf_path)).pages)
