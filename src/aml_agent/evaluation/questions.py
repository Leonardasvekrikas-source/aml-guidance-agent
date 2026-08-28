"""Loading and resolving the evaluation set.

Gold passages are named by `(document_id, chunk_index)`, not by database id.
Chunk ids are `bigserial` values that depend on ingestion order and history, so
they differ between one machine and the next; an evaluation set keyed on them
scores zero on anybody else's clone, and the benchmark that is the centrepiece
of this repository would not be reproducible. `(document_id, chunk_profile,
chunk_index)` is deterministic given the same corpus and chunker, and the
schema already enforces it as unique.

Resolution happens against the database at evaluation time, and a gold
reference that cannot be resolved is a hard error. A stale reference would
otherwise sink recall and look exactly like a retrieval failure — the most
misleading way for an evaluation to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import settings


@dataclass(frozen=True)
class GoldRef:
    """A gold passage, named stably."""

    document_id: str
    chunk_index: int

    def __str__(self) -> str:
        return f"{self.document_id}#{self.chunk_index}"


@dataclass
class Question:
    id: str
    question: str
    topic: str
    answerable: bool
    gold: tuple[GoldRef, ...] = ()
    answer_summary: str = ""
    note: str = ""
    # Filled in by resolve_gold_ids once the corpus is known.
    gold_chunk_ids: tuple[int, ...] = field(default=())


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or settings.questions_path
    if not path.exists():
        raise FileNotFoundError(f"no evaluation set at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("questions")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a non-empty 'questions:' list")

    questions: list[Question] = []
    seen: set[str] = set()

    for index, item in enumerate(items):
        qid = str(item.get("id", "")).strip()
        if not qid:
            raise ValueError(f"{path}: question {index} has no id")
        if qid in seen:
            raise ValueError(f"{path}: duplicate question id {qid!r}")
        seen.add(qid)

        answerable = bool(item.get("answerable", True))

        gold: list[GoldRef] = []
        for entry in item.get("gold") or []:
            if not isinstance(entry, dict) or "doc" not in entry or "chunk" not in entry:
                raise ValueError(
                    f"{path}: {qid} has a malformed gold entry {entry!r}; "
                    "expected {doc: <document_id>, chunk: <chunk_index>}"
                )
            gold.append(GoldRef(str(entry["doc"]).strip(), int(entry["chunk"])))

        if answerable and not gold:
            raise ValueError(
                f"{path}: {qid} is marked answerable but names no gold passages. "
                "An answerable question with no gold set cannot score recall."
            )
        if not answerable and gold:
            raise ValueError(
                f"{path}: {qid} is marked unanswerable but names gold passages. "
                "If the corpus contains the answer, the question is answerable."
            )

        questions.append(
            Question(
                id=qid,
                question=str(item["question"]).strip(),
                topic=str(item.get("topic", "")).strip(),
                answerable=answerable,
                gold=tuple(gold),
                answer_summary=str(item.get("answer_summary", "")).strip(),
                note=str(item.get("note", "")).strip(),
            )
        )

    return questions


def resolve_gold_ids(questions: list[Question], profile: str) -> list[str]:
    """Resolve every gold reference to a chunk id in the given profile.

    Mutates each Question's ``gold_chunk_ids`` in place and returns a list of
    unresolvable references. An empty list means the evaluation set is coherent
    with the ingested corpus.
    """
    from ..db import connect

    wanted = {(g.document_id, g.chunk_index) for q in questions for g in q.gold}
    if not wanted:
        return []

    documents = sorted({d for d, _ in wanted})
    indices = sorted({i for _, i in wanted})

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, document_id, chunk_index
            FROM chunks
            WHERE chunk_profile = %s
              AND document_id = ANY(%s)
              AND chunk_index  = ANY(%s)
            """,
            (profile, documents, indices),
        ).fetchall()

    lookup = {(r["document_id"], r["chunk_index"]): r["id"] for r in rows}

    problems: list[str] = []
    for question in questions:
        resolved: list[int] = []
        for ref in question.gold:
            chunk_id = lookup.get((ref.document_id, ref.chunk_index))
            if chunk_id is None:
                problems.append(
                    f"{question.id}: gold passage {ref} does not exist in profile "
                    f"{profile!r}. Has the corpus been ingested, and does the "
                    "manifest still contain that document?"
                )
            else:
                resolved.append(chunk_id)
        question.gold_chunk_ids = tuple(resolved)

    return problems
