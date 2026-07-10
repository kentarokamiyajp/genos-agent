"""Markdown → BlockNote block conversion for agent-written note bodies.

The agent's answers are markdown (headings, bullets, numbered lists,
**bold**, *italic*). When it saves one into a note, the body must be
BlockNote-shaped JSON so the note editor renders that structure natively
instead of showing one wall of plain text with literal `##`/`**` control
characters.

This is the backend counterpart of the frontend
`features/agentQA/markdownToBlocks.ts` — deliberately the same small
vocabulary and the same block/inline shapes, so a note the agent writes
is indistinguishable from one saved through the UI's "save answer as
note" flow. Citation tokens (`[prose](type:id)` and bare `[type:id]`)
are resolved into in-app `/workspace/...` links when the caller supplies
an `entity_link_resolver` (see `entity_links.resolve_note_entity_link`;
create_note / update_note pass one bound to the caller's team); without
a resolver, or on a miss, the target degrades to its visible prose
(never a dead link).

The output is consumed by two places, both of which this shape satisfies:
  * the BlockNote editor (paragraph/heading/bulletListItem/
    numberedListItem with the standard props), and
  * the reindex chunker (`text_extraction.extract_sections`), which walks
    `content`/`children` of every block type and splits on `heading`
    blocks — so structured output actually *improves* search precision
    (heading-bounded sections) over the old one-paragraph blob.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Maps a citation token ("task:12") to (href, fallback_label), or None
# when it can't be resolved (unknown id, foreign team, unsupported type).
EntityLinkResolver = Callable[[str], "tuple[str, str] | None"]

_BASE_PROPS = {
    "textColor": "default",
    "textAlignment": "left",
    "backgroundColor": "default",
}

# Inline markers. Bold is matched before italic so `**x**` isn't eaten by
# the single-`*` italic rule. Underscore italic/bold use word-boundary
# guards so `snake_case_name` isn't mangled into italics. Mirrors the
# frontend regex vocabulary.
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_FORMAT_RE = re.compile(
    r"\*\*(.+?)\*\*"  # **bold**
    r"|__(.+?)__"  # __bold__
    r"|(?<![A-Za-z0-9])\*([^*\s][^*]*?)\*(?![A-Za-z0-9])"  # *italic*
    r"|(?<![A-Za-z0-9])_([^_\s][^_]*?)_(?![A-Za-z0-9])"  # _italic_
)

# Accept 1–6 `#` (markdown allows h1–h6) but cap the rendered level at 3,
# which is what BlockNote's default theme supports — a `#### Sub` heading
# renders as level 3 rather than leaking literal `####` into a paragraph.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*•]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\d+\.\s+(.*)$")


def _text_node(text: str, styles: dict[str, bool] | None = None) -> dict[str, Any]:
    return {"type": "text", "text": text, "styles": styles or {}}


def _tokenize_formatting(text: str) -> list[dict[str, Any]]:
    """Split a link-free run into text nodes carrying bold/italic styles."""
    if not text:
        return []
    out: list[dict[str, Any]] = []
    cursor = 0
    for m in _FORMAT_RE.finditer(text):
        if m.start() > cursor:
            out.append(_text_node(text[cursor : m.start()]))
        if m.group(1) is not None or m.group(2) is not None:
            out.append(_text_node(m.group(1) or m.group(2) or "", {"bold": True}))
        else:
            out.append(_text_node(m.group(3) or m.group(4) or "", {"italic": True}))
        cursor = m.end()
    if cursor < len(text):
        out.append(_text_node(text[cursor:]))
    return out


# Bare citation token in prose — `[task:12]` — that is NOT the label of
# a markdown link (no following paren). Same vocabulary as the frontend's
# CITATION_PATTERN.
_BARE_TOKEN_RE = re.compile(
    r"\[((?:chat|task|note|project|todo|milestone):[^\]\s]+)\](?!\()"
)


def _link_node(href: str, label: str) -> dict[str, Any]:
    return {"type": "link", "href": href, "content": [_text_node(label)]}


def _tokenize_prose(text: str, resolver: EntityLinkResolver | None) -> list[dict[str, Any]]:
    """A link-free run: resolve bare `[type:id]` citation tokens into
    links when a resolver is supplied (label = the entity's
    display_id/name/title), otherwise/on a miss keep the literal text;
    everything between runs through bold/italic tokenisation."""
    if not text:
        return []
    if resolver is None:
        return _tokenize_formatting(text)
    out: list[dict[str, Any]] = []
    cursor = 0
    for m in _BARE_TOKEN_RE.finditer(text):
        resolved = resolver(m.group(1))
        if not resolved:
            continue  # leave the literal token inside the surrounding run
        if m.start() > cursor:
            out.extend(_tokenize_formatting(text[cursor : m.start()]))
        out.append(_link_node(resolved[0], resolved[1]))
        cursor = m.end()
    if cursor < len(text):
        out.extend(_tokenize_formatting(text[cursor:]))
    return out


def _tokenize_inline(
    text: str, resolver: EntityLinkResolver | None = None
) -> list[dict[str, Any]]:
    """Inline content for one line: markdown links become BlockNote link
    nodes — real URLs always, and `[prose](task:5)`-style citation
    targets when an entity-link resolver is supplied (create_note /
    update_note pass one bound to the caller's team, so saved notes
    carry working in-app links). An unresolvable citation target
    degrades to its prose so the note never carries a dead link."""
    if not text:
        return []
    out: list[dict[str, Any]] = []
    cursor = 0
    for m in _LINK_RE.finditer(text):
        if m.start() > cursor:
            out.extend(_tokenize_prose(text[cursor : m.start()], resolver))
        label, href = m.group(1), m.group(2)
        if href.startswith(("http://", "https://", "mailto:")):
            out.append(_link_node(href, label))
        else:
            resolved = resolver(href) if resolver else None
            if resolved:
                # Citation target → in-app link, keeping the model's prose.
                out.append(_link_node(resolved[0], label))
            else:
                # Non-URL target we can't resolve — keep the prose.
                out.extend(_tokenize_formatting(label))
        cursor = m.end()
    if cursor < len(text):
        out.extend(_tokenize_prose(text[cursor:], resolver))
    return out


def _block(
    block_type: str,
    text: str,
    *,
    level: int | None = None,
    resolver: EntityLinkResolver | None = None,
) -> dict[str, Any]:
    props = dict(_BASE_PROPS)
    if level is not None:
        props["level"] = level
    return {
        "type": block_type,
        "props": props,
        "content": _tokenize_inline(text, resolver),
        "children": [],
    }


def markdown_to_blocks(
    markdown: str, *, entity_link_resolver: EntityLinkResolver | None = None
) -> list[dict[str, Any]]:
    """Parse the agent's markdown into BlockNote `PartialBlock`-shaped dicts.

    Handles headings (# / ## / ###), bullet lists (-, *, •), numbered
    lists (1. …), paragraphs, and inline **bold** / *italic* / markdown
    links. Anything unrecognised degrades to a plain paragraph rather than
    being dropped.

    Returns `[]` for empty input (a deliberately title-only note), matching
    the previous `_wrap_blocknote` contract.
    """
    text = (markdown or "").strip()
    if not text:
        return []

    blocks: list[dict[str, Any]] = []
    lines = text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()

        # Blank line — block boundary.
        if not stripped:
            i += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = min(len(heading.group(1)), 3)
            blocks.append(_block("heading", heading.group(2), level=level, resolver=entity_link_resolver))
            i += 1
            continue

        if _BULLET_RE.match(stripped):
            while i < n:
                mm = _BULLET_RE.match(lines[i].strip())
                if not mm:
                    break
                blocks.append(_block("bulletListItem", mm.group(1), resolver=entity_link_resolver))
                i += 1
            continue

        if _NUMBERED_RE.match(stripped):
            while i < n:
                mm = _NUMBERED_RE.match(lines[i].strip())
                if not mm:
                    break
                blocks.append(_block("numberedListItem", mm.group(1), resolver=entity_link_resolver))
                i += 1
            continue

        # Paragraph: gather until a blank line or a block-starting line.
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if not nxt:
                break
            if _HEADING_RE.match(nxt) or _BULLET_RE.match(nxt) or _NUMBERED_RE.match(nxt):
                break
            buf.append(nxt.rstrip("\\"))
            i += 1
        blocks.append(_block("paragraph", " ".join(buf), resolver=entity_link_resolver))

    return blocks
