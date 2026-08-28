"""Token-aware chunking with page attribution.

Chunk boundaries are computed in the embedding model's own tokens, using its
tokenizer, rather than estimated from characters. The constraint that matters
is the model's 512-token window, and characters-per-token varies enough across
legal prose, tables and citation-heavy text that an estimate would put some
chunks over the limit and waste capacity on others.

Each chunk records the page its text begins on, so a citation can name a page
a human can turn to.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from ..config import ChunkProfile, settings
from .extract import Page

# Below this, a trailing fragment is merged backwards instead of stored. A
# 12-token chunk is never independently useful and skews average-length stats.
MIN_TAIL_TOKENS = 32


@dataclass
class Chunk:
    document_id: str
    chunk_profile: str
    chunk_index: int
    page: int | None
    section_heading: str | None
    text: str
    char_count: int
    token_count: int


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str):
    """The embedding model's tokenizer, loaded once.

    Imported lazily so that reading the manifest or running the tests does not
    pull in transformers.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _join_pages(pages: Sequence[Page]) -> tuple[str, list[tuple[int, int, Page]]]:
    """Join pages into one string, remembering each page's character span."""
    parts: list[str] = []
    spans: list[tuple[int, int, Page]] = []
    cursor = 0
    separator = "\n\n"

    for position, page in enumerate(pages):
        text = page.text
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, page))
        if position < len(pages) - 1:
            parts.append(separator)
            cursor += len(separator)

    return "".join(parts), spans


def _page_at(char_offset: int, spans: list[tuple[int, int, Page]]) -> Page | None:
    for start, end, page in spans:
        if start <= char_offset < end:
            return page
    return spans[-1][2] if spans else None


def chunk_document(
    document_id: str,
    pages: Sequence[Page],
    profile: ChunkProfile,
    model_name: str | None = None,
) -> list[Chunk]:
    if not pages:
        return []

    tokenizer = get_tokenizer(model_name or settings.embedding_model)
    document_text, spans = _join_pages(pages)

    encoding = tokenizer(
        document_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets: list[tuple[int, int]] = [tuple(pair) for pair in encoding["offset_mapping"]]
    if not offsets:
        return []

    stride = profile.target_tokens - profile.overlap_tokens
    chunks: list[Chunk] = []
    start_token = 0
    index = 0

    while start_token < len(offsets):
        end_token = min(start_token + profile.target_tokens, len(offsets))

        # Merge a short tail into this window rather than emitting a stub.
        remaining = len(offsets) - end_token
        if 0 < remaining < MIN_TAIL_TOKENS:
            end_token = len(offsets)

        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        text = document_text[char_start:char_end].strip()

        if text:
            page = _page_at(char_start, spans)
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_profile=profile.name,
                    chunk_index=index,
                    page=page.number if page else None,
                    section_heading=page.heading if page else None,
                    text=text,
                    char_count=len(text),
                    token_count=end_token - start_token,
                )
            )
            index += 1

        if end_token >= len(offsets):
            break
        start_token += stride

    return chunks
