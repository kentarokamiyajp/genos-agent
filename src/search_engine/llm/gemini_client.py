"""Gemini adapter for the `ModelClient` interface.

Translates between provider-neutral types (`AgentMessage`,
`FunctionCall`, `ToolDeclaration`) and Google's `google-genai` SDK
wire types (`types.Content`, `types.Part`, `types.Tool`).

Supports two authentication modes (chosen via Django settings):

  Mode A — Gemini AI Studio API key (`GEMINI_USE_VERTEX=false`,
    default): set `GEMINI_API_KEY` from https://aistudio.google.com/apikey.

  Mode B — Vertex AI service account (`GEMINI_USE_VERTEX=true`):
    provide a GCP service-account JSON via either
    `GEMINI_SERVICE_ACCOUNT_FILE` or `GOOGLE_APPLICATION_CREDENTIALS`.
    Also requires `GEMINI_PROJECT` (and optionally `GEMINI_LOCATION`,
    default "us-central1"). The service account needs the
    `roles/aiplatform.user` role. `GEMINI_LLM_LOCATION` may override the
    region for this LLM client alone (falling back to `GEMINI_LOCATION`),
    leaving the Vertex embedder on its own region — e.g. point the LLM at
    `global` for a preview model while embeddings stay regional.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from django.conf import settings
from google import genai

from origin.search_engine.llm.types import AgentMessage, FunctionCall, ToolDeclaration

log = logging.getLogger(__name__)

_client: genai.Client | None = None


def _retry_http_options(cfg: Any):
    """HttpOptions enabling the SDK's transient-error retry, or None.

    google-genai wraps request INITIATION in a tenacity retry when
    `retry_options` is set — the streaming path included, but never
    mid-stream, so a retried call can't double-yield chunks the
    controller already consumed. Retryable by SDK default: HTTP
    408/429/500/502/503/504 plus httpx connect/timeout errors, with
    exponential backoff + jitter (initial 1s, base 2). We only pin
    `attempts` (GEMINI_RETRY_ATTEMPTS, total including the first call)
    and keep the SDK defaults for codes/delays.

    Why: a transient Vertex 429 RESOURCE_EXHAUSTED used to hard-fail
    the agent step (and poison the nightly tool_recall metric — issue
    #46); one retried call rides out quota blips instead.
    """
    from google.genai import types  # noqa: PLC0415

    attempts = int(cfg.get("GEMINI_RETRY_ATTEMPTS") or 0)
    if attempts <= 1:
        return None
    return types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=attempts))


def _build_client() -> genai.Client:
    """Construct the underlying Gemini SDK client from settings."""
    cfg = settings.SEARCH_ENGINE
    http_options = _retry_http_options(cfg)

    if cfg.get("GEMINI_USE_VERTEX"):
        project = cfg.get("GEMINI_PROJECT") or ""
        # Prefer the LLM-specific region when set, so the agent can reach a
        # model (e.g. a *-preview) that a shared regional endpoint doesn't
        # serve, WITHOUT moving the Vertex embedder (which reads
        # GEMINI_LOCATION) off the region its index was built in. Falls
        # back to GEMINI_LOCATION, then us-central1. See settings.py.
        location = cfg.get("GEMINI_LLM_LOCATION") or cfg.get("GEMINI_LOCATION") or "us-central1"
        sa_file = cfg.get("GEMINI_SERVICE_ACCOUNT_FILE") or ""
        if not project:
            raise RuntimeError(
                "GEMINI_USE_VERTEX=true but GEMINI_PROJECT is not set. "
                "Set it to your GCP project id."
            )
        if sa_file:
            # Explicit service-account file → load credentials directly
            # rather than relying on the GOOGLE_APPLICATION_CREDENTIALS
            # convention, so the JSON doesn't have to live at a fixed
            # path.
            from google.oauth2 import service_account  # noqa: PLC0415

            credentials = service_account.Credentials.from_service_account_file(
                sa_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=credentials,
                http_options=http_options,
            )
        # No explicit file → fall through to Application Default
        # Credentials (reads GOOGLE_APPLICATION_CREDENTIALS automatically).
        return genai.Client(
            vertexai=True, project=project, location=location, http_options=http_options
        )

    # Mode A: plain API key.
    api_key = cfg.get("GEMINI_API_KEY") or ""
    if not api_key:
        raise RuntimeError(
            "Neither GEMINI_API_KEY nor GEMINI_USE_VERTEX is configured. "
            "For Gemini AI Studio, set GEMINI_API_KEY. For Vertex AI, "
            "set GEMINI_USE_VERTEX=true plus GEMINI_PROJECT and "
            "GEMINI_SERVICE_ACCOUNT_FILE (or GOOGLE_APPLICATION_CREDENTIALS)."
        )
    return genai.Client(api_key=api_key, http_options=http_options)


def _get_client() -> genai.Client:
    """Singleton accessor — builds the SDK client on first call."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


class GeminiClient:
    """`ModelClient` adapter backed by Google's `google-genai` SDK."""

    def generate_step(
        self,
        messages: list[AgentMessage],
        tools: list[ToolDeclaration],
        system_instruction: str,
        *,
        model_override: str | None = None,
    ) -> Iterator[tuple[str | None, FunctionCall | None]]:
        """Stream one model turn against the given history.

        Yields `(text_chunk, None)` for incremental text and
        `(None, FunctionCall)` for each function call the model
        requests. The controller assembles these into the agent loop.
        """
        from google.genai import types  # noqa: PLC0415

        sdk_messages = [_message_to_sdk(m, types) for m in messages]
        sdk_tools = _tools_to_sdk(tools, types) if tools else None
        model = model_override or settings.SEARCH_ENGINE["GEMINI_MODEL"]

        config = types.GenerateContentConfig(
            tools=sdk_tools,
            system_instruction=system_instruction,
            temperature=0.2,
        )

        try:
            stream = _get_client().models.generate_content_stream(
                model=model,
                contents=sdk_messages,
                config=config,
            )
            # Parallel function calls — when Gemini 3 emits multiple
            # function_call parts in one response, only the FIRST part
            # carries a `thought_signature`; the rest come back with
            # `None`. Our controller splits those parallel calls into
            # separate assistant Content turns when echoing, so each
            # split-out call needs its OWN signature or Gemini 3 rejects
            # the request with:
            #   400 INVALID_ARGUMENT: Function call is missing a
            #   thought_signature in functionCall parts.
            # Replicating the shared signature to every split-out call
            # works (verified against the live API): the model treats
            # them as part of the same reasoning block. Track the last
            # signature we saw in THIS stream and back-fill missing ones
            # — scope is one `generate_step` call so we never leak a
            # signature from a prior turn.
            last_seen_signature: bytes | None = None
            # Capture the last `usage_metadata` we see in any chunk —
            # only the final chunk carries the meaningful totals, but
            # peeking at every chunk avoids assuming chunk order. Logged
            # below the stream so we can verify Gemini implicit caching
            # is actually firing (cached_content_token_count > 0).
            last_usage_metadata: Any = None
            for chunk in stream:
                usage = getattr(chunk, "usage_metadata", None)
                if usage is not None:
                    last_usage_metadata = usage
                # A streaming chunk's candidates carry content.parts —
                # each part is either a text fragment or a function
                # call. Yield them in order so the controller sees the
                # same interleaving the model emitted.
                candidates = getattr(chunk, "candidates", None) or []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    if content is None:
                        continue
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        # Update the carry-forward signature ANY time we
                        # see one, regardless of part kind — text parts
                        # in reasoning mode can carry signatures too.
                        part_sig = getattr(part, "thought_signature", None)
                        if part_sig is not None:
                            last_seen_signature = part_sig

                        fcall = getattr(part, "function_call", None)
                        if fcall is not None:
                            # Use this part's own signature when set;
                            # otherwise the most recent one we saw in
                            # this response (covers parallel calls 2+).
                            sig_for_call = part_sig or last_seen_signature
                            yield (None, _sdk_function_call_to_neutral(fcall, sig_for_call))
                            continue
                        text = getattr(part, "text", None)
                        if text:
                            yield (text, None)

            _log_usage(last_usage_metadata, model)
        except Exception:
            log.exception("Gemini generate_step failed")
            raise


# --------------------------------------------------------------------------- #
# Implicit-cache observability                                                #
# --------------------------------------------------------------------------- #


def _log_usage(usage: Any, model: str) -> None:
    """Log Gemini response usage_metadata so we can see implicit-cache
    hit rate. Implicit caching is on by default on Gemini 2.5+/3.x; this
    just makes the savings visible. `cached_content_token_count > 0`
    means the prompt prefix (system_instruction + tools + earlier turns)
    was served from cache and billed at the cached rate.

    Gated on `LLM_LOG_USAGE_METADATA` (default off) so production logs
    stay quiet unless an operator flips it on.
    """
    if usage is None:
        return
    if not settings.SEARCH_ENGINE.get("LLM_LOG_USAGE_METADATA", False):
        return
    prompt_n = getattr(usage, "prompt_token_count", None) or 0
    cached_n = getattr(usage, "cached_content_token_count", None) or 0
    cand_n = getattr(usage, "candidates_token_count", None) or 0
    thought_n = getattr(usage, "thoughts_token_count", None) or 0
    tool_n = getattr(usage, "tool_use_prompt_token_count", None) or 0
    total_n = getattr(usage, "total_token_count", None) or 0
    cache_pct = round((cached_n / prompt_n) * 100) if prompt_n else 0
    log.info(
        "gemini usage model=%s prompt=%d cached=%d (%d%%) candidates=%d "
        "thoughts=%d tool_prompt=%d total=%d",
        model,
        prompt_n,
        cached_n,
        cache_pct,
        cand_n,
        thought_n,
        tool_n,
        total_n,
    )


# --------------------------------------------------------------------------- #
# Translation helpers — neutral types <-> google-genai SDK types              #
# --------------------------------------------------------------------------- #


def _message_to_sdk(msg: AgentMessage, types: Any):
    """Translate one `AgentMessage` into a `types.Content` turn."""
    if msg.role == "user":
        return types.Content(
            role="user",
            parts=[types.Part(text=msg.text or "")],
        )
    if msg.role == "assistant":
        if msg.function_call is not None:
            # Build Part kwargs dynamically so we only set
            # `thought_signature` when we actually captured one — old
            # SDK versions whose Part type rejects unknown kwargs
            # stay happy, and we don't accidentally send a `None`
            # signature that the API might reject.
            part_kwargs: dict[str, Any] = {
                "function_call": types.FunctionCall(
                    name=msg.function_call.name,
                    args=dict(msg.function_call.args),
                ),
            }
            if msg.function_call.thought_signature is not None:
                part_kwargs["thought_signature"] = msg.function_call.thought_signature
            return types.Content(
                role="model",
                parts=[types.Part(**part_kwargs)],
            )
        return types.Content(
            role="model",
            parts=[types.Part(text=msg.text or "")],
        )
    if msg.role == "tool_response":
        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=msg.function_response_name or "",
                    response=msg.function_response or {},
                )
            ],
        )
    raise ValueError(f"Unknown AgentMessage role: {msg.role!r}")


def _tools_to_sdk(tools: list[ToolDeclaration], types: Any) -> list:
    """Translate neutral tool declarations into the SDK's `types.Tool`."""
    declarations = [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=t.parameters_schema,
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=declarations)]


def _sdk_function_call_to_neutral(
    fcall: Any,
    thought_signature: bytes | None = None,
) -> FunctionCall:
    """Convert a Gemini SDK function-call object to neutral `FunctionCall`.

    `thought_signature` comes from the surrounding `Part` (not the
    function-call object itself) and is required by Gemini 3+ when we
    echo this call back as part of the assistant turn — see
    `FunctionCall.thought_signature` in `types.py` for the rationale.
    """
    return FunctionCall(
        name=getattr(fcall, "name", "") or "",
        args=dict(getattr(fcall, "args", {}) or {}),
        thought_signature=thought_signature,
    )
