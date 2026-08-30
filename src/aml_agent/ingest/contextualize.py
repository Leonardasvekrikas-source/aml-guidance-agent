"""Contextual retrieval: situate each chunk in its document before embedding.

A chunk taken out of a 66-page report loses the thing that made it findable.
"These indicators apply to the buyer" is unsearchable on its own — which
buyer, which indicators, which typology? The passage answers a question no
retriever can match it to, because the words that would match live three pages
earlier.

The technique (Anthropic, 2024) is to prepend a short, model-written sentence
placing the chunk in its document, then embed and index *that*. The chunk text
a reader sees is unchanged; only the representation used for search grows.

**Cost is the whole design problem here.** Done naively this is one model call
per chunk with the entire document as context — for this corpus, 2,864 calls
each carrying ~40,000 tokens. Two things make it affordable:

  * **Prompt caching.** The document is sent once per document and read from
    cache thereafter, at a tenth of the input price.
  * **Batching.** Each call situates many chunks at once, so the cached
    document is read once per batch rather than once per chunk. This is the
    larger saving by far: it divides the dominant cost term by the batch size.

Together those turn roughly $30 of tokens into roughly $3. The estimate is
printed before anything is spent, and `--dry-run` prints it without spending.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import anthropic

from ..config import CHUNK_PROFILES, profile_by_name, settings
from ..cost import Spend
from ..db import connect, delete_chunks, insert_chunks
from ..llm import make_client
from .chunk import Chunk, chunk_document
from .embed import check_no_truncation, embed_passages
from .extract import extract_pages
from .manifest import load_manifest

# Chunks situated per model call, and the main cost lever. The cached document
# is re-read once per batch, so the dominant cost term scales with the number
# of batches rather than the number of chunks: going from 12 to 24 roughly
# halves the bill. Too large and the model starts mismatching contexts to
# excerpts, which is why the tool requires an explicit excerpt_number rather
# than trusting array order.
BATCH_SIZE = 24

# The suffix marking a contextualised profile. Boundaries are identical to the
# source profile, so chunk_index still lines up and the evaluation set's gold
# references resolve against either.
CTX_SUFFIX = "ctx"

SYSTEM = """You situate excerpts within the document they came from, so that \
each excerpt can be found by search on its own.

For each excerpt you are given, write ONE sentence, at most 30 words, stating \
what it is about in terms a reader searching the corpus would use. Name the \
specific subject the excerpt discusses — the typology, the obligation, the \
sector, the jurisdiction — and the section or context it sits in.

Rules:
- Describe the excerpt's own subject. Do not summarise the whole document.
- Use concrete nouns from the document, not vague framing. "Red flag \
indicators for over-invoicing in trade finance documentation" is useful; \
"This section discusses various matters" is not.
- Resolve what the excerpt leaves implicit. If it says "these indicators" or \
"the buyer", say which indicators and which buyer.
- Do not add facts that are not in the document.
- Write the sentence alone. No preamble, no numbering beyond what the tool \
requires."""

CONTEXT_TOOL: dict[str, Any] = {
    "name": "record_contexts",
    "description": "Record one situating sentence for each excerpt, in order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "contexts": {
                "type": "array",
                "description": (
                    "One entry per excerpt, in the same order as given. Must have "
                    "exactly as many entries as there were excerpts."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "excerpt_number": {"type": "integer"},
                        "context": {"type": "string"},
                    },
                    "required": ["excerpt_number", "context"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["contexts"],
        "additionalProperties": False,
    },
}


@dataclass
class Estimate:
    documents: int
    chunks: int
    calls: int
    cached_read_tokens: int
    cache_write_tokens: int
    output_tokens: int

    def usd(self, model: str) -> float:
        from ..cost import usd

        return usd(
            model,
            cache_write_tokens=self.cache_write_tokens,
            cache_read_tokens=self.cached_read_tokens,
            output_tokens=self.output_tokens,
        )


def estimate(profile_name: str, model: str) -> Estimate:
    """Predict the bill before spending it.

    Rough by design — it assumes 4 characters per token and a fixed output
    length — but the point is to catch an order-of-magnitude mistake before it
    is charged, not to be exact.
    """
    entries = load_manifest()
    profile = profile_by_name(profile_name)

    documents = 0
    chunks = 0
    calls = 0
    cache_write = 0
    cache_read = 0

    for entry in entries:
        path = entry.raw_path()
        if not path.exists():
            continue
        pages = extract_pages(path)
        doc_tokens = sum(len(p.text) for p in pages) // 4
        n = len(chunk_document(entry.id, pages, profile))
        if not n:
            continue

        documents += 1
        chunks += n
        batches = -(-n // BATCH_SIZE)
        calls += batches
        cache_write += doc_tokens
        cache_read += doc_tokens * batches

    return Estimate(
        documents=documents,
        chunks=chunks,
        calls=calls,
        cached_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        output_tokens=chunks * 40,
    )


def _situate_batch(
    client: anthropic.Anthropic,
    model: str,
    document_text: str,
    batch: list[Chunk],
    spend: Spend,
) -> list[str]:
    excerpts = "\n\n".join(
        f"EXCERPT {i}:\n{chunk.text[:2000]}" for i, chunk in enumerate(batch, start=1)
    )

    response = client.messages.create(  # type: ignore[call-overload]
        model=model,
        max_tokens=200 * len(batch) + 200,
        system=[
            {"type": "text", "text": SYSTEM},
            {
                # The document is the expensive part and is identical for every
                # batch from it, so it is the cache breakpoint. Everything
                # after it — the excerpts — changes per call.
                "type": "text",
                "text": f"DOCUMENT:\n\n{document_text}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        tools=[CONTEXT_TOOL],
        tool_choice={"type": "tool", "name": "record_contexts"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Situate each of the following {len(batch)} excerpts within the "
                    f"document above. Return exactly {len(batch)} entries.\n\n{excerpts}"
                ),
            }
        ],
    )

    spend.add(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    )

    contexts = [""] * len(batch)
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_contexts":
            payload = dict(block.input)
            for item in payload.get("contexts") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    position = int(item.get("excerpt_number", 0)) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= position < len(batch):
                    contexts[position] = str(item.get("context", "")).strip()
    return contexts


def build(profile_name: str, model: str, limit: int | None = None) -> int:
    target_profile = f"{profile_name}{CTX_SUFFIX}"
    profile = profile_by_name(profile_name)
    entries = load_manifest()
    if limit:
        entries = entries[:limit]

    client = make_client()
    spend = Spend(model=model)
    written = 0
    missing_context = 0

    with connect() as conn:
        for position, entry in enumerate(entries, start=1):
            path = entry.raw_path()
            if not path.exists():
                continue

            pages = extract_pages(path)
            chunks = chunk_document(entry.id, pages, profile)
            if not chunks:
                continue

            document_text = "\n\n".join(p.text for p in pages)

            contexts: list[str] = []
            for start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[start : start + BATCH_SIZE]
                try:
                    contexts.extend(_situate_batch(client, model, document_text, batch, spend))
                except anthropic.APIError as exc:
                    print(f"  ! {entry.id}: {type(exc).__name__}: {exc}")
                    contexts.extend([""] * len(batch))

            # A chunk whose context failed keeps its original text rather than
            # being dropped, so a partial failure degrades the experiment
            # instead of shrinking the corpus underneath it.
            rows = []
            for chunk, context in zip(chunks, contexts, strict=True):
                if not context:
                    missing_context += 1
                text = f"{context}\n\n{chunk.text}" if context else chunk.text
                rows.append(
                    {
                        "document_id": chunk.document_id,
                        "chunk_profile": target_profile,
                        "chunk_index": chunk.chunk_index,
                        "page": chunk.page,
                        "page_end": chunk.page_end,
                        "section_heading": chunk.section_heading,
                        "text": text,
                        "char_count": len(text),
                        "token_count": chunk.token_count,
                    }
                )

            # The row dicts hold mixed value types, so pull the texts out with
            # an explicit str() rather than leaving the element type as object.
            texts = [str(r["text"]) for r in rows]
            check_no_truncation(texts)
            vectors = embed_passages(texts)
            for row, vector in zip(rows, vectors, strict=True):
                row["embedding"] = vector

            delete_chunks(conn, entry.id, target_profile)
            insert_chunks(conn, rows)
            conn.commit()

            written += len(rows)
            print(
                f"  + [{position}/{len(entries)}] {entry.id:<42} "
                f"{len(rows):>4} chunks  ${spend.total_usd:.3f} so far",
                flush=True,
            )

    print(f"\nwrote {written} contextualised chunks to profile {target_profile!r}")
    if missing_context:
        print(f"{missing_context} chunks kept their original text (context generation failed)")
    print(f"spend: ${spend.total_usd:.2f} across {spend.calls} calls")

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "contextualize_spend.json").write_text(
        json.dumps({"profile": target_profile, **spend.to_dict()}, indent=2),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    profile_name = "t480"
    for name in (p.name for p in CHUNK_PROFILES):
        if f"--profile={name}" in sys.argv:
            profile_name = name

    model = settings.grader_model
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    est = estimate(profile_name, model)
    print(
        f"estimate for profile {profile_name!r} on {model}:\n"
        f"  {est.documents} documents, {est.chunks} chunks, {est.calls} calls "
        f"(batch size {BATCH_SIZE})\n"
        f"  ~{est.cache_write_tokens // 1000}k tokens cached, "
        f"~{est.cached_read_tokens // 1000}k cache reads, "
        f"~{est.output_tokens // 1000}k output\n"
        f"  ESTIMATED COST: ${est.usd(model):.2f}\n"
    )

    if "--dry-run" in sys.argv:
        print("dry run; nothing spent")
        return 0

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    return build(profile_name, model, limit)


if __name__ == "__main__":
    raise SystemExit(main())
