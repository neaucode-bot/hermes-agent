#!/usr/bin/env python3
"""
Session Title Tool — get or set the current conversation's display title.

Mirrors the /title slash command and TUI gateway session.title RPC. Titles
live in the SQLite session DB and appear in session lists, /resume, and sidebars.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _ensure_session_row(db, session_id: str) -> None:
    """Create the session row if it is missing (early title set before first turn)."""
    if db.get_session(session_id):
        return
    source = os.environ.get("HERMES_SESSION_SOURCE", "cli")
    try:
        db.create_session(session_id=session_id, source=source)
    except Exception as exc:
        logger.debug("session_title create_session skipped for %s: %s", session_id, exc)


def session_title(
    title: Optional[str] = None,
    *,
    db=None,
    current_session_id: Optional[str] = None,
) -> str:
    """Get or set the display title for the current session."""
    if not db:
        from hermes_state import format_session_db_unavailable

        return json.dumps({"success": False, "error": format_session_db_unavailable()})

    session_id = (current_session_id or "").strip()
    if not session_id:
        return json.dumps(
            {"success": False, "error": "No active session id — cannot read or set a title."}
        )

    set_requested = title is not None
    title_arg = title.strip() if isinstance(title, str) else None

    try:
        if set_requested:
            _ensure_session_row(db, session_id)
            sanitized = db.sanitize_title(title_arg or "")
            if not sanitized:
                return json.dumps(
                    {
                        "success": False,
                        "error": "Title is empty after sanitization.",
                        "session_id": session_id,
                    }
                )
            if not db.set_session_title(session_id, sanitized):
                return json.dumps(
                    {
                        "success": False,
                        "error": "Session not found — could not set title.",
                        "session_id": session_id,
                    }
                )
            return json.dumps(
                {
                    "success": True,
                    "action": "set",
                    "session_id": session_id,
                    "title": sanitized,
                }
            )

        resolved = db.get_session_title(session_id)
        return json.dumps(
            {
                "success": True,
                "action": "get",
                "session_id": session_id,
                "title": resolved or None,
            }
        )
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc), "session_id": session_id})
    except Exception as exc:
        logger.debug("session_title failed for %s", session_id, exc_info=True)
        return json.dumps({"success": False, "error": str(exc), "session_id": session_id})


def check_session_title_requirements() -> bool:
    """Requires the SQLite state database."""
    try:
        from hermes_state import DEFAULT_DB_PATH

        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_TITLE_SCHEMA: Dict[str, Any] = {
    "name": "session_title",
    "description": (
        "Get or set the display title for the current conversation session. "
        "Titles appear in session lists, /resume, and the TUI sidebar.\n\n"
        "CALLING SHAPES\n\n"
        "  1) SET — pass `title`:\n"
        "     session_title(title=\"Hermes proxy tool routing\")\n"
        "     Use when the user asks to rename the chat, update the title, or when "
        "the topic has shifted enough that the auto-generated title is misleading.\n\n"
        "  2) GET — omit `title`:\n"
        "     session_title()\n"
        "     Returns the current title (null when unset).\n\n"
        "NOT auto-title — Hermes may auto-generate a title after the first exchange "
        "via auxiliary title_generation. Use this tool for explicit user requests or "
        "orchestrator-initiated renames. Keep titles short (3–8 words), descriptive, "
        "no quotes or trailing punctuation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": (
                    "New session title (set shape). Sanitized to 3–80 printable chars. "
                    "Omit to read the current title."
                ),
            },
        },
        "required": [],
    },
}


from tools.registry import registry

registry.register(
    name="session_title",
    toolset="session_title",
    schema=SESSION_TITLE_SCHEMA,
    handler=lambda args, **kw: session_title(
        title=args.get("title"),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_session_title_requirements,
    emoji="✏️",
)
