"""Provider-neutral types for the agent ↔ ModelClient boundary.

The agent controller (`agent/controller.py`) only knows about these
types — it never imports `google.genai` or `anthropic`. Each adapter
(`gemini_client.GeminiClient`, `claude_client.ClaudeClient`) is
responsible for translating these neutral shapes into its own SDK's
wire format on the way in, and translating the SDK's stream events
back into the (text, FunctionCall) yield pattern on the way out.

This is what makes the abstraction real: if a type leaks from a
specific SDK into the controller, the abstraction is fake. So the
controller's helpers (_user_turn, _assistant_function_call_turn,
_function_response_turn, _build_tool_declarations) build these
neutral shapes — and only these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class FunctionCall:
    """A function call the model wants to make.

    Yielded by `ModelClient.generate_step` and stored on
    `AgentMessage.function_call` when replaying assistant turns back
    to the model.
    """

    name: str
    args: dict[str, Any]
    # Gemini 3+ returns an opaque "thought signature" alongside each
    # functionCall part that captures the model's internal reasoning
    # state at the point of the call. The client MUST echo it back
    # when replaying the assistant turn (in the same /ask round-trip)
    # or Gemini 3 rejects the request with
    #   400 INVALID_ARGUMENT: Function call is missing a
    #   thought_signature in functionCall parts. ...
    # Older Gemini versions and Claude don't use it, so `None` is the
    # safe default. Stored as raw bytes; the adapter handles wire-
    # format encoding (Google's SDK serialises to base64 internally).
    # See https://ai.google.dev/gemini-api/docs/thought-signatures
    thought_signature: bytes | None = None


@dataclass
class AgentMessage:
    """One turn in the agent's conversation, provider-neutral.

    Exactly one of `text` / `function_call` / `function_response`
    is set, depending on `role`:

      * `role="user"`           → `text` set
      * `role="assistant"`      → `text` set OR `function_call` set
      * `role="tool_response"`  → `function_response_name`
                                  + `function_response` set

    `tool_response` is its own role rather than a flavor of `user`
    because Anthropic and Gemini both encode tool responses
    distinctly from a free-text user message, and adapters need a
    clean signal of which shape to emit.
    """

    role: Literal["user", "assistant", "tool_response"]
    text: str | None = None
    function_call: FunctionCall | None = None
    function_response_name: str | None = None
    function_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolDeclaration:
    """A tool the model may call.

    `parameters_schema` is JSON Schema (the same shape both Gemini and
    Anthropic accept — neither needs a separate type). Adapters
    translate this into their SDK's tool-declaration object.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
