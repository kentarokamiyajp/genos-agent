"""OpenAI embedding adapter.

Batches calls to the /v1/embeddings endpoint and retries transient
errors a couple of times with exponential backoff. Embeddings from the
text-embedding-3 family are pre-normalised, so OpenSearch can use
cosine similarity directly.

`task_type` is accepted to satisfy the `Embedder` protocol but ignored
— OpenAI's API has no equivalent to Vertex's RETRIEVAL_DOCUMENT /
RETRIEVAL_QUERY distinction.
"""

from __future__ import annotations

import logging
import time

from django.conf import settings
from openai import OpenAI

from origin.search_engine.embeddings.base import TaskType

logger = logging.getLogger(__name__)


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = settings.SEARCH_ENGINE["OPENAI_API_KEY"]
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in the environment "
            "before running indexing or query embedding."
        )
    _client = OpenAI(api_key=api_key)
    return _client


class OpenAIEmbedder:
    @property
    def model_name(self) -> str:
        return settings.SEARCH_ENGINE["OPENAI_EMBEDDING_MODEL"]

    @property
    def dimensions(self) -> int:
        return settings.SEARCH_ENGINE["OPENAI_EMBEDDING_DIMENSIONS"]

    def embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        del task_type  # OpenAI has no asymmetric encoding mode.
        if not texts:
            return []
        model = self.model_name
        batch_size = settings.SEARCH_ENGINE["EMBEDDING_BATCH_SIZE"]
        client = _get_client()

        results: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            results.extend(_embed_with_retry(client, batch, model))
        return results


def _embed_with_retry(client: OpenAI, batch: list[str], model: str, max_retries: int = 3):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=batch)
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa: BLE001 — rate-limit, transient net, etc.
            if attempt == max_retries - 1:
                raise
            logger.warning(
                "OpenAI embedding call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt + 1,
                max_retries,
                e,
                delay,
            )
            time.sleep(delay)
            delay *= 2
