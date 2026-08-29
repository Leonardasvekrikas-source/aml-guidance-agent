"""Configuration, read once from the environment.

Every tunable that affects a number in the README lives here, so that the
settings a result was produced under can be recorded alongside it.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
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
class EmbeddingSpec:
    """One embedding model, and everything that depends on choosing it.

    The column is part of the spec because a pgvector column has a fixed
    dimension: a 1024-wide model cannot be written into a 768-wide column, and
    the mismatch would surface deep inside a bulk insert rather than at the
    point the model was chosen.

    Query and passage prefixes are part of the spec because these models are
    asymmetric in different ways, and getting it wrong does not error — it
    quietly costs recall, which is the worst kind of bug in a retrieval system.
    """

    key: str
    hf_id: str
    dim: int
    column: str
    query_prefix: str = ""
    passage_prefix: str = ""


EMBEDDING_MODELS: dict[str, EmbeddingSpec] = {
    # The original baseline. Asymmetric: queries take an instruction prefix,
    # passages take none.
    "bge-base": EmbeddingSpec(
        key="bge-base",
        hf_id="BAAI/bge-base-en-v1.5",
        dim=768,
        column="embedding",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    # BGE-M3. Symmetric — its authors specify no instruction prefix on either
    # side — and an 8192-token window rather than 512.
    "bge-m3": EmbeddingSpec(
        key="bge-m3",
        hf_id="BAAI/bge-m3",
        dim=1024,
        column="embedding_lg",
    ),
    # E5 family. Symmetric in shape but both sides are prefixed, with
    # *different* prefixes.
    "e5-large": EmbeddingSpec(
        key="e5-large",
        hf_id="intfloat/multilingual-e5-large-instruct",
        dim=1024,
        column="embedding_lg",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
}

# bge-m3 is the default because it measurably raises the first-stage
# ceiling - the one number the reranker cannot improve. See
# results/embedding_ablation.md. bge-base remains selectable so the
# comparison stays reproducible rather than being a claim about a model
# nobody can run any more.
DEFAULT_EMBEDDING = "bge-m3"


def embedding_spec(key: str | None = None) -> EmbeddingSpec:
    name = (key or _env("EMBEDDING_MODEL_KEY", DEFAULT_EMBEDDING)).strip()
    if name not in EMBEDDING_MODELS:
        known = ", ".join(sorted(EMBEDDING_MODELS))
        raise KeyError(f"unknown embedding model {name!r}; known: {known}")
    return EMBEDDING_MODELS[name]


@dataclass(frozen=True)
class Settings:
    # --- database ---
    pg_user: str = field(default_factory=lambda: _env("POSTGRES_USER", "aml"))
    pg_password: str = field(default_factory=lambda: _env("POSTGRES_PASSWORD", "aml"))
    pg_db: str = field(default_factory=lambda: _env("POSTGRES_DB", "aml"))
    pg_host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "db"))
    pg_port: int = field(default_factory=lambda: _env_int("POSTGRES_PORT", 5432))

    # --- embeddings ---
    embedding_key: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL_KEY", DEFAULT_EMBEDDING)
    )
    embedding_device: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", "cuda"))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))

    # --- reranking ---
    # bge-reranker-v2-m3 is a cross-encoder: it scores a (query, passage) pair
    # in one forward pass rather than comparing independently-computed vectors.
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    )
    # First-stage candidates handed to the reranker. This is a hard ceiling on
    # the whole pipeline: a gold passage outside this pool can never be
    # recovered, however good the reranker is.
    rerank_candidates: int = field(default_factory=lambda: _env_int("RERANK_CANDIDATES", 50))
    rerank_batch_size: int = field(default_factory=lambda: _env_int("RERANK_BATCH_SIZE", 32))

    # --- generation ---
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-5"))
    # Required only for identity-linked API keys, which must name the
    # workspace a request acts in. Ordinary keys ignore this.
    anthropic_workspace_id: str = field(default_factory=lambda: _env("ANTHROPIC_WORKSPACE_ID", ""))

    # --- paths ---
    manifest_path: Path = REPO_ROOT / "corpus" / "manifest.yaml"
    raw_dir: Path = REPO_ROOT / "corpus" / "raw"
    results_dir: Path = REPO_ROOT / "results"
    traces_dir: Path = REPO_ROOT / "results" / "traces"
    questions_path: Path = REPO_ROOT / "eval" / "questions.yaml"

    @property
    def embedding(self) -> EmbeddingSpec:
        return embedding_spec(self.embedding_key)

    @property
    def embedding_model(self) -> str:
        return self.embedding.hf_id

    @property
    def embedding_dim(self) -> int:
        return self.embedding.dim

    @property
    def embedding_column(self) -> str:
        return self.embedding.column

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
            "embedding_key": self.embedding_key,
            "embedding_column": self.embedding_column,
            "reranker_model": self.reranker_model,
            "rerank_candidates": self.rerank_candidates,
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
