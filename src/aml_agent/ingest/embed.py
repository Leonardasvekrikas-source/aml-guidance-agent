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

# Prefix specified by the model's authors for retrieval queries. Passages get
# no prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


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

    model = get_model()
    vectors = model.encode(
        list(texts),
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
        QUERY_INSTRUCTION + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    _check_dim(vector.reshape(1, -1))
    return vector.astype(np.float32)


def _check_dim(vectors: np.ndarray) -> None:
    """Fail loudly on a dimension mismatch.

    The schema declares vector(768). Swapping EMBEDDING_MODEL for one with a
    different width would otherwise fail deep inside a bulk insert, with an
    error that says nothing about the cause.
    """
    width = vectors.shape[1]
    if width != settings.embedding_dim:
        raise ValueError(
            f"{settings.embedding_model} produces {width}-dimensional vectors but "
            f"EMBEDDING_DIM is {settings.embedding_dim} and the chunks table "
            f"declares vector({settings.embedding_dim}). Change both, and write a "
            "migration — the existing index cannot hold a different width."
        )
