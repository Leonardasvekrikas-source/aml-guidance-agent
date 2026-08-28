"""The tool the agent may call.

There is exactly one: search. Tool calling is not magic — the model emits a
structured request naming a tool and its arguments, this code executes it, and
the result goes back into the conversation as a message. The model never
touches the database; it asks, and this code decides what actually runs.
"""

from __future__ import annotations

from typing import Any

from ..retrieval.base import Hit, Retriever

MAX_K = 10
MAX_SNIPPET_CHARS = 1200

SEARCH_TOOL: dict[str, Any] = {
    "name": "search",
    "description": (
        "Search the corpus of public AML regulatory and typology guidance. "
        "Returns passages with their source document and page number. "
        "Use specific regulatory or typology vocabulary rather than long "
        "natural-language sentences. If a search returns nothing relevant, "
        "reformulate with different terminology rather than repeating it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Keywords and phrases work better than full sentences.",
            },
            "k": {
                "type": "integer",
                "description": f"How many passages to return, 1 to {MAX_K}. Default 5.",
            },
        },
        "required": ["query"],
    },
}


def run_search(retriever: Retriever, arguments: dict[str, Any]) -> tuple[str, list[Hit]]:
    """Execute a search tool call.

    Arguments come from the model and are therefore untrusted input. A model
    that emits k=10000 or omits the query must produce a useful error message
    back into the conversation, not an exception that kills the loop.
    """
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return "ERROR: 'query' is required and must be a non-empty string.", []

    raw_k = arguments.get("k", 5)
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        k = 5
    k = max(1, min(k, MAX_K))

    hits = retriever.search(query.strip(), k)
    if not hits:
        return (
            f"No passages matched {query.strip()!r}. Try different terminology — "
            "the corpus uses regulatory and typology vocabulary.",
            [],
        )

    return format_hits(hits), hits


def format_hits(hits: list[Hit]) -> str:
    """Render passages for the model.

    Every passage is labelled with the chunk id, because the drafting step must
    cite chunk ids and validation checks those ids against what was actually
    retrieved. A citation format the model has to invent is a citation format
    it will get wrong.
    """
    blocks: list[str] = []
    for hit in hits:
        text = hit.text.strip()
        if len(text) > MAX_SNIPPET_CHARS:
            text = text[:MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + " ..."
        location = f"p.{hit.page}" if hit.page else "page unknown"
        heading = f" | {hit.section_heading}" if hit.section_heading else ""
        blocks.append(
            f"[chunk {hit.chunk_id}] {hit.title} ({hit.publisher}), {location}{heading}\n{text}"
        )
    return "\n\n---\n\n".join(blocks)
