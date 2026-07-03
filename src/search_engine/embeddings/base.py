"""Embedder protocol — neutral interface across providers."""

from __future__ import annotations

from typing import Literal, Protocol

# Indexing path embeds *documents*; query path embeds *queries*.
# Vertex's gemini-embedding-001 uses this distinction to apply asymmetric
# encoding (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY), which materially
# improves retrieval quality. OpenAI's text-embedding-3 family has no
# equivalent and ignores the hint.
TaskType = Literal["document", "query"]


class Embedder(Protocol):
    """Provider-neutral embedding interface.

    Adapters batch as they see fit and return one vector per input,
    preserving order. Empty/whitespace inputs are sanitised to a
    single space *upstream* in `embeddings/__init__.py`, so adapters
    can assume non-empty strings.
    """

    @property
    def model_name(self) -> str:
        """Model identifier persisted in `RagChunk.embedding_model`."""
        ...

    @property
    def dimensions(self) -> int:
        """Vector length emitted by `embed()` — matches the OpenSearch
        index's `knn_vector.dimension` setting."""
        ...

    def embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]: ...
