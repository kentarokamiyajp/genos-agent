"""Plain-text extraction from BlockNote-style JSONField bodies.

Message bodies, comment bodies, task content, and note bodies in this
app are stored as BlockNote-style structured JSON. We need a plain
string for both keyword indexing and embedding generation.

Body shape: a list of blocks, where each block looks roughly like:

    {
      "type": "paragraph" | "heading" | "bulletListItem" | ...,
      "content": [
        {"type": "text", "text": "hello"},
        {"type": "mention", "props": {"userName": "alice"}},
        {"type": "link", "content": [{"text": "https://..."}]},
        ...
      ],
      "children": [<nested blocks>]  # optional
    }

Some entries may also be a single dict (not wrapped in a list) or a
plain string from older or simpler writers. We accept all of these.
"""

from typing import Any


def extract_text(body: Any) -> str:
    """Best-effort plain-text extraction from a BlockNote-style body.

    Never raises: unknown shapes degrade to an empty string rather
    than failing the entire indexing run.
    """
    if body is None:
        return ""
    if isinstance(body, str):
        return body.strip()
    if isinstance(body, list):
        return _join(_walk_blocks(body))
    if isinstance(body, dict):
        # Could be either a single block or a top-level wrapper.
        if "content" in body and isinstance(body.get("content"), list):
            return _join(_walk_block(body))
        # Some writers may nest the list under "blocks" or similar.
        for key in ("blocks", "body", "doc"):
            inner = body.get(key)
            if isinstance(inner, list):
                return _join(_walk_blocks(inner))
        return ""
    return ""


def extract_sections(body: Any) -> list[tuple[str, str]]:
    """Split a BlockNote body into (heading, section_text) sections.

    A new section begins at each `type: "heading"` block. Content
    before the first heading is returned as a section with an empty
    heading. Sections whose body AND heading are both empty are
    dropped — they index nothing useful.

    Returns a list of `(heading, body_text)` tuples in document
    order. For bodies without any headings, returns a single section
    `[("", full_text)]` (equivalent to `extract_text` flattened).

    Used by Phase 9 note chunking so each section becomes its own
    indexed chunk — improves retrieval precision when a note has
    multiple distinct topics under headings.
    """
    blocks = _top_level_blocks(body)
    if not blocks:
        text = extract_text(body)
        return [("", text)] if text else []

    sections: list[tuple[str, list[str]]] = [("", [])]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "heading":
            heading = _join(_walk_block(block))
            sections.append((heading, []))
            continue
        block_text = _join(_walk_block(block))
        if block_text:
            sections[-1][1].append(block_text)

    out: list[tuple[str, str]] = []
    for heading, parts in sections:
        body_text = "\n".join(parts).strip()
        if not heading and not body_text:
            continue
        out.append((heading, body_text))
    return out


def _top_level_blocks(body: Any) -> list:
    """Return the list of top-level blocks in `body`, or [] if shape is unknown."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        if "content" in body and isinstance(body.get("content"), list):
            return [body]
        for key in ("blocks", "body", "doc"):
            inner = body.get(key)
            if isinstance(inner, list):
                return inner
    return []


def _walk_blocks(blocks):
    parts = []
    for block in blocks:
        if isinstance(block, dict):
            parts.extend(_walk_block(block))
    return parts


def _walk_block(block):
    parts = []
    for inline in block.get("content", []) or []:
        parts.extend(_walk_inline(inline))
    # Recurse into nested children (e.g., list items with sub-lists).
    for child in block.get("children", []) or []:
        if isinstance(child, dict):
            parts.extend(_walk_block(child))
    return parts


def _walk_inline(inline):
    if not isinstance(inline, dict):
        return []
    t = inline.get("type")
    if t == "text":
        text = inline.get("text", "")
        return [str(text)] if text else []
    if t == "mention":
        props = inline.get("props") or {}
        name = props.get("userName") or props.get("name")
        return [f"@{name}"] if name else []
    if t == "link":
        # Links embed their own content array.
        nested = inline.get("content") or []
        return _join_inline(nested)
    # Unknown inline type — try to recover any "text" field.
    text = inline.get("text")
    return [str(text)] if text else []


def _join_inline(inlines):
    parts = []
    for inline in inlines:
        if isinstance(inline, dict):
            parts.extend(_walk_inline(inline))
    return parts


def _join(parts):
    return " ".join(p for p in parts if p).strip()
