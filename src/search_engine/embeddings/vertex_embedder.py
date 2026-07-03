"""Vertex AI embedding adapter.

Uses the same `google-genai` SDK and the same GCP auth env vars as the
Gemini LLM client (`llm/gemini_client.py`), specifically the Mode B
(Vertex AI service-account) branch documented in
`apis/settings.py:SEARCH_ENGINE`. We deliberately don't support the
AI Studio API-key mode here — embeddings are an index-time decision
that affects long-term storage, and "use Vertex" makes the billing /
IAM story unambiguous.

Default model is `gemini-embedding-001` truncated via Matryoshka to
the dim configured in `VERTEX_EMBEDDING_DIMENSIONS` (1536 by default,
matching the existing OpenSearch index) so a provider swap from
OpenAI doesn't require recreating the index.

`task_type` is mapped to the SDK's `RETRIEVAL_DOCUMENT` /
`RETRIEVAL_QUERY` so asymmetric retrieval pays off — index-time
embeddings and query-time embeddings land on opposite sides of the
encoder as the model was trained for.

Per-request batch size is read from `EMBEDDING_BATCH_SIZE` (shared
with the OpenAI path, default 100). Vertex enforces *server-side*
per-model limits that the SDK does not validate:

  - `text-embedding-005` / older Gecko models: up to 250 instances.
  - `gemini-embedding-001`: 1 instance per request as of late 2025.

If you see HTTP 400 with "instances must contain at most 1 element"
during reindex, set `SEARCH_EMBEDDING_BATCH_SIZE=1` in the env.
"""

from __future__ import annotations

import logging
import time

from django.conf import settings
from google import genai

from origin.search_engine.embeddings.base import TaskType

logger = logging.getLogger(__name__)


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client
    cfg = settings.SEARCH_ENGINE
    if not cfg.get("GEMINI_USE_VERTEX"):
        raise RuntimeError(
            "EMBEDDING_PROVIDER=vertex requires GEMINI_USE_VERTEX=true. "
            "Set GEMINI_USE_VERTEX=true, GEMINI_PROJECT=<gcp-project>, "
            "and either GEMINI_SERVICE_ACCOUNT_FILE or "
            "GOOGLE_APPLICATION_CREDENTIALS. See the SEARCH_ENGINE block "
            "in apis/settings.py for the full setup notes."
        )
    project = cfg.get("GEMINI_PROJECT") or ""
    location = cfg.get("GEMINI_LOCATION") or "us-central1"
    sa_file = cfg.get("GEMINI_SERVICE_ACCOUNT_FILE") or ""
    if not project:
        raise RuntimeError(
            "GEMINI_USE_VERTEX=true but GEMINI_PROJECT is not set. "
            "Set it to your GCP project id."
        )
    if sa_file:
        # Explicit service-account file → load credentials directly
        # rather than relying on the GOOGLE_APPLICATION_CREDENTIALS
        # convention, so the JSON doesn't have to live at a fixed path.
        from google.oauth2 import service_account  # noqa: PLC0415

        credentials = service_account.Credentials.from_service_account_file(
            sa_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        _client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )
    else:
        # Fall through to Application Default Credentials.
        _client = genai.Client(vertexai=True, project=project, location=location)
    return _client


_SDK_TASK_TYPE = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


class VertexEmbedder:
    @property
    def model_name(self) -> str:
        return settings.SEARCH_ENGINE["VERTEX_EMBEDDING_MODEL"]

    @property
    def dimensions(self) -> int:
        return settings.SEARCH_ENGINE["VERTEX_EMBEDDING_DIMENSIONS"]

    def embed(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        if not texts:
            return []
        from google.genai import types  # noqa: PLC0415

        model = self.model_name
        dims = self.dimensions
        batch_size = settings.SEARCH_ENGINE["EMBEDDING_BATCH_SIZE"]
        sdk_task_type = _SDK_TASK_TYPE[task_type]
        config = types.EmbedContentConfig(
            task_type=sdk_task_type,
            output_dimensionality=dims,
        )

        client = _get_client()
        results: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            results.extend(_embed_with_retry(client, batch, model, config))
        return results


def _embed_with_retry(
    client: genai.Client,
    batch: list[str],
    model: str,
    config,
    max_retries: int = 3,
) -> list[list[float]]:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.models.embed_content(
                model=model,
                contents=batch,
                config=config,
            )
            return [list(e.values) for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001 — rate-limit, transient net, etc.
            if attempt == max_retries - 1:
                raise
            logger.warning(
                "Vertex embedding call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt + 1,
                max_retries,
                e,
                delay,
            )
            time.sleep(delay)
            delay *= 2
