"""Embedding.

bge-base-en-v1.5 is asymmetric: queries are prefixed with a short instruction,
passages are not. Embedding both sides the same way is a common and quiet
mistake — it does not error, it just costs recall — so the two directions are
separate functions here rather than one function with a flag that is easy to
forget.

Vectors are L2-normalised, which makes cosine distance and inner product
equivalent and matches the `vector_cosine_ops` index in the migration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from ..config import settings

# Prefixes come from the active model's spec in config, because they differ
# per model: bge-base prefixes only queries, e5 prefixes both sides with
# different strings, and bge-m3 prefixes neither. Applying the wrong one does
# not error, it quietly costs recall.


@lru_cache(maxsize=2)
def get_model(model_name: str | None = None, device: str | None = None):
    """Load the embedding model once per process.

    Falls back to CPU rather than failing when CUDA is unavailable, because a
    stranger cloning this repository will not necessarily have a GPU and the
    corpus is small enough that CPU ingestion is slow but tolerable.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    name = model_name or settings.embedding_model
    requested = device or settings.embedding_device

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"  CUDA requested but not available; embedding on CPU instead")
        requested = "cpu"

    return SentenceTransformer(name, device=requested)


def embed_passages(texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)

    prefix = settings.embedding.passage_prefix
    model = get_model()
    vectors = model.encode(
        [prefix + t for t in texts] if prefix else list(texts),
        batch_size=batch_size or settings.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    _check_dim(vectors)
    return vectors.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    vector = model.encode(
        settings.embedding.query_prefix + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    _check_dim(vector.reshape(1, -1))
    return vector.astype(np.float32)


def check_no_truncation(texts: Sequence[str], sample: int = 200) -> None:
    """Fail loudly if the active model would truncate these chunks.

    Chunk boundaries come from a fixed reference tokenizer, so a model with a
    different tokenizer or a smaller window may not fit them. Truncation does
    not raise — it silently drops the tail of a passage, leaving text that BM25
    can find and dense retrieval cannot. That is exactly the kind of failure
    this project exists to avoid reporting as a result.
    """
    if not texts:
        return

    model = get_model()
    window = getattr(model, "max_seq_length", None)
    if not window:
        return

    tokenizer = model.tokenizer
    step = max(1, len(texts) // sample)
    worst = 0
    for text in texts[::step]:
        count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        worst = max(worst, count)

    if worst > window:
        raise ValueError(
            f"{settings.embedding_model} truncates at {window} tokens, but chunks "
            f"reach {worst} tokens under its tokenizer. Chunk boundaries are set by "
            f"the reference tokenizer, so this model needs a smaller chunk profile. "
            "Embedding anyway would silently drop the tail of long passages, making "
            "them findable by BM25 and invisible to dense retrieval."
        )


def _check_dim(vectors: np.ndarray) -> None:
    """Fail loudly on a dimension mismatch.

    The schema declares vector(768). Swapping EMBEDDING_MODEL for one with a
    different width would otherwise fail deep inside a bulk insert, with an
    error that says nothing about the cause.
    """
    width = vectors.shape[1]
    if width != settings.embedding_dim:
        raise ValueError(
            f"{settings.embedding_model} produces {width}-dimensional vectors, but "
            f"the {settings.embedding_key!r} spec declares {settings.embedding_dim} "
            f"and writes to chunks.{settings.embedding_column}. Fix the spec in "
            "config, and add a migration if no column of that width exists — a "
            "pgvector column has a fixed dimension."
        )
