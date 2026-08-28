"""Loading and validating the evaluation set.

Gold chunk ids are validated against the database at load time. A gold id that
does not exist would otherwise sink recall silently and look like a retrieval
failure, which is the most misleading way for an evaluation to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import settings


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    topic: str
    answerable: bool
    gold_chunk_ids: tuple[int, ...] = ()
    answer_summary: str = ""
    note: str = ""


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
        gold = tuple(int(g) for g in (item.get("gold_chunk_ids") or []))

        if answerable and not gold:
            raise ValueError(
                f"{path}: {qid} is marked answerable but has no gold_chunk_ids. "
                "An answerable question with no gold set cannot score recall."
            )
        if not answerable and gold:
            raise ValueError(
                f"{path}: {qid} is marked unanswerable but lists gold chunks. "
                "If the corpus contains the answer, the question is answerable."
            )

        questions.append(
            Question(
                id=qid,
                question=str(item["question"]).strip(),
                topic=str(item.get("topic", "")).strip(),
                answerable=answerable,
                gold_chunk_ids=gold,
                answer_summary=str(item.get("answer_summary", "")).strip(),
                note=str(item.get("note", "")).strip(),
            )
        )

    return questions


def validate_gold_chunks(questions: list[Question], profile: str) -> list[str]:
    """Check every gold chunk id exists in the given profile.

    Returns a list of human-readable problems. Empty means the evaluation set
    is coherent with the ingested corpus.
    """
    from ..db import connect

    wanted: set[int] = set()
    for question in questions:
        wanted.update(question.gold_chunk_ids)
    if not wanted:
        return []

    with connect() as conn:
        rows = conn.execute(
            "SELECT id FROM chunks WHERE id = ANY(%s) AND chunk_profile = %s",
            (list(wanted), profile),
        ).fetchall()
    present = {row["id"] for row in rows}

    problems: list[str] = []
    for question in questions:
        missing = [g for g in question.gold_chunk_ids if g not in present]
        if missing:
            problems.append(
                f"{question.id}: gold chunk id(s) {missing} not present in profile "
                f"{profile!r}"
            )
    return problems
