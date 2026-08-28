"""Configuration, read once from the environment.

Every tunable that affects a number in the README lives here, so that the
settings a result was produced under can be recorded alongside it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class ChunkProfile:
    """One chunking configuration.

    ``target_tokens`` and ``overlap_tokens`` are counted with the embedding
    model's own tokenizer, not estimated from characters, because the limit
    that matters is the model's context window and a character estimate is
    wrong by a factor that varies with the text.
    """

    name: str
    target_tokens: int
    overlap_tokens: int

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError(
                f"profile {self.name}: overlap ({self.overlap_tokens}) must be "
                f"smaller than target ({self.target_tokens}), otherwise chunking "
                "does not advance"
            )


# bge-base-en-v1.5 truncates at 512 tokens. Both profiles sit below that on
# purpose: a profile larger than the window would be silently truncated at
# embedding time, so its tail would be searchable by BM25 and invisible to
# dense retrieval. That would not be a chunk-size comparison, it would be a
# comparison between one honest configuration and one broken one.
CHUNK_PROFILES: tuple[ChunkProfile, ...] = (
    ChunkProfile(name="t256", target_tokens=256, overlap_tokens=48),
    ChunkProfile(name="t480", target_tokens=480, overlap_tokens=64),
)

DEFAULT_PROFILE = "t480"


@dataclass(frozen=True)
class Settings:
    # --- database ---
    pg_user: str = field(default_factory=lambda: _env("POSTGRES_USER", "aml"))
    pg_password: str = field(default_factory=lambda: _env("POSTGRES_PASSWORD", "aml"))
    pg_db: str = field(default_factory=lambda: _env("POSTGRES_DB", "aml"))
    pg_host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "db"))
    pg_port: int = field(default_factory=lambda: _env_int("POSTGRES_PORT", 5432))

    # --- embeddings ---
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 768))
    embedding_device: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", "cuda"))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))

    # --- generation ---
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5"))

    # --- paths ---
    manifest_path: Path = REPO_ROOT / "corpus" / "manifest.yaml"
    raw_dir: Path = REPO_ROOT / "corpus" / "raw"
    results_dir: Path = REPO_ROOT / "results"
    traces_dir: Path = REPO_ROOT / "results" / "traces"
    questions_path: Path = REPO_ROOT / "eval" / "questions.yaml"

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )

    def require_api_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add a "
                "key. Retrieval and `make eval-retrieval` run without one; drafting, "
                "the agent loop and the judge do not."
            )
        return self.anthropic_api_key

    def provenance(self) -> dict[str, Any]:
        """The settings worth recording next to a result."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "anthropic_model": self.anthropic_model,
            "chunk_profiles": [asdict(p) for p in CHUNK_PROFILES],
        }


settings = Settings()


def profile_by_name(name: str) -> ChunkProfile:
    for profile in CHUNK_PROFILES:
        if profile.name == name:
            return profile
    known = ", ".join(p.name for p in CHUNK_PROFILES)
    raise KeyError(f"unknown chunk profile {name!r}; known profiles: {known}")
