"""Pinned tail-steering reminder — the engine's true tail-reminder slot.

Hermes builds ONE cached, byte-stable head ``system`` message per session and
replays it verbatim every turn (prompt caching is sacred). Volatile per-turn
guidance must therefore ride *after* the cached prefix, near the model's
generation point — the analog of Cursor's ``<system_reminder>`` pattern. This
module owns the canonical steer text plus the cache-correct injector.

``DEFAULT_TAIL_STEER`` is the single source of truth shared with the generated
Cursor contract (parity spec §D.3): the contract embeds the identical line so
the IDE and the engine pin the same reminder. Keep it byte-identical with the
line in ``.cursor/rules/hermes-contract.generated.mdc``.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Canonical, verbatim steer line. The generated Cursor contract embeds this
# exact string — do not reword without updating the contract generator too.
DEFAULT_TAIL_STEER = (
    "Re-read your system prompt and current objective before taking your next action."
)

# Fence mirroring Cursor's tail-reminder marker, so the same guidance reads the
# same way on both interfaces.
TAIL_STEER_OPEN = "<system_reminder>"
TAIL_STEER_CLOSE = "</system_reminder>"


def format_tail_steer(text: str) -> str:
    """Render steer ``text`` inside the tail-reminder fence."""
    return f"{TAIL_STEER_OPEN}\n{text}\n{TAIL_STEER_CLOSE}"


def inject_tail_steer(
    api_messages: List[Dict[str, Any]],
    steer_text: str = DEFAULT_TAIL_STEER,
) -> List[Dict[str, Any]]:
    """Fold a tail-pinned steering reminder onto the LAST conversational
    (``user``/``tool``) message, in place, so it sits *after* the cached head
    prefix and right before the model generates.

    Why fold rather than append a trailing ``{"role": "system"}`` message:
    Hermes treats the head ``system`` message as singular and byte-stable, and
    native-Anthropic / Bedrock adapters extract system messages into a
    top-level ``system`` param keeping only the *last* one
    (``convert_messages_to_anthropic``). A second trailing system message would
    silently replace the real head prompt there and break the cache prefix.
    Folding onto the last message is provider-agnostic, role-alternation-safe,
    cache-correct (the head is untouched; the last message is already volatile
    per turn, so the steer costs nothing the multi-turn cache wasn't already
    re-sending), and re-salient — the same mechanism ``/steer`` and the
    memory-prefetch block already use.

    Operates on ``api_messages`` (the per-call copy); each element is itself a
    shallow copy of the stored history message, so list content is replaced
    with a fresh list (never mutated in place) to avoid leaking the reminder
    into persisted history.

    No-op when ``steer_text`` is empty/whitespace or there is no ``user``/
    ``tool`` message to fold onto.
    """
    if not steer_text or not steer_text.strip() or not api_messages:
        return api_messages

    fence = format_tail_steer(steer_text.strip())

    for i in range(len(api_messages) - 1, -1, -1):
        msg = api_messages[i]
        if msg.get("role") not in ("user", "tool"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            base = content.rstrip()
            msg["content"] = f"{base}\n\n{fence}" if base else fence
        elif isinstance(content, list):
            blocks = list(content)
            blocks.append({"type": "text", "text": fence})
            msg["content"] = blocks
        else:
            msg["content"] = fence
        return api_messages

    return api_messages
