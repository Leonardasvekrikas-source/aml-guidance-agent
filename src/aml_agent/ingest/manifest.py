"""Reading and validating corpus/manifest.yaml.

The manifest is the corpus's source of truth. Every document in the database
traces back to an entry here, and every entry records where the document came
from and when it was retrieved. A manifest that validates loosely produces a
corpus whose provenance claims cannot be trusted, so this validates strictly
and refuses to guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from ..config import settings

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
REQUIRED = ("id", "title", "publisher", "source_url", "publication_date")


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    title: str
    publisher: str
    source_url: str
    publication_date: date | None
    doc_type: str

    @property
    def filename(self) -> str:
        return f"{self.id}.pdf"

    def raw_path(self) -> Path:
        return settings.raw_dir / self.filename


def _coerce_date(raw: object, doc_id: str) -> date | None:
    """Accept YYYY, YYYY-MM or a full date; reject anything else.

    A missing publication date is allowed and stored as NULL, because some
    genuinely undated guidance exists. A *wrong* one is not allowed, which is
    why a bare unparseable string is an error rather than a silent None.
    """
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    text = str(raw).strip()
    if not text or text.lower() == "unknown":
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        # A year-only or year-month date is stored as the first of the period.
        # The database column is a date, and inventing a day is better than
        # discarding the year entirely — but it is an approximation, so the
        # manifest keeps the string the publisher actually stated.
        return parsed if fmt == "%Y-%m-%d" else parsed.replace(day=1)

    raise ValueError(
        f"document {doc_id!r}: publication_date {text!r} is not YYYY, YYYY-MM or "
        "YYYY-MM-DD. Use 'unknown' if the document genuinely has no stated date."
    )


def load_manifest(path: Path | None = None) -> list[ManifestEntry]:
    path = path or settings.manifest_path
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest at {path}. The corpus is defined by this file; "
            "nothing can be ingested without it."
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    documents = raw.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{path}: expected a non-empty 'documents:' list")

    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()

    for index, item in enumerate(documents):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: entry {index} is not a mapping")

        missing = [key for key in REQUIRED if not str(item.get(key, "")).strip()]
        if missing:
            raise ValueError(
                f"{path}: entry {index} ({item.get('id', '?')}) is missing "
                f"required field(s): {', '.join(missing)}"
            )

        doc_id = str(item["id"]).strip()
        if not ID_PATTERN.match(doc_id):
            raise ValueError(
                f"{path}: id {doc_id!r} must be lowercase alphanumeric with "
                "hyphens, 3-64 characters. It becomes a filename and a primary key."
            )
        if doc_id in seen_ids:
            raise ValueError(f"{path}: duplicate document id {doc_id!r}")
        seen_ids.add(doc_id)

        url = str(item["source_url"]).strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"{path}: {doc_id} source_url must be http(s), got {url!r}")
        if url in seen_urls:
            raise ValueError(f"{path}: duplicate source_url for {doc_id!r}: {url}")
        seen_urls.add(url)

        entries.append(
            ManifestEntry(
                id=doc_id,
                title=str(item["title"]).strip(),
                publisher=str(item["publisher"]).strip(),
                source_url=url,
                publication_date=_coerce_date(item.get("publication_date"), doc_id),
                doc_type=str(item.get("doc_type", "guidance")).strip(),
            )
        )

    return entries
