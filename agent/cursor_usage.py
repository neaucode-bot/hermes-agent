from __future__ import annotations

import base64
import json
import logging
import math
import os
import sqlite3
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

CURSOR_USAGE_SUMMARY_URL = "https://cursor.com/api/usage-summary"
CURSOR_STATE_DB_KEY = "cursorAuth/accessToken"


def _cursor_state_db_path() -> str:
    return os.path.expanduser(
        "~/Library/Application Support/Cursor/User/globalStorage/state.vscdb"
    )


def _decode_jwt_payload(token: str) -> Optional[dict[str, Any]]:
    parts = str(token or "").strip().split(".")
    if len(parts) < 2:
        return None
    segment = parts[1]
    padding = "=" * (-len(segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(segment + padding)
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def extract_cursor_session_token() -> Optional[str]:
    """Build WorkosCursorSessionToken from Cursor IDE local auth (macOS).

    Reads ``cursorAuth/accessToken`` from Cursor's VS Code state DB and
    composes ``<user_id>::<jwt>``. Returns None when Cursor is not installed,
    not signed in, or the token cannot be parsed. Never logs the token.
    """
    db_path = _cursor_state_db_path()
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT value FROM ItemTable WHERE key=?", (CURSOR_STATE_DB_KEY,))
            row = cur.fetchone()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return None
    if not row:
        return None
    token = row[0].decode() if isinstance(row[0], bytes) else str(row[0])
    token = token.strip()
    if not token:
        return None
    payload = _decode_jwt_payload(token)
    if not payload:
        return None
    sub = str(payload.get("sub") or "").strip()
    if "|" not in sub:
        return None
    user_id = sub.split("|", 1)[1].strip()
    if not user_id:
        return None
    return f"{user_id}::{token}"


def _is_finite_percent(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fetch_cursor_usage_summary(*, timeout: float = 10.0) -> Optional[dict[str, Any]]:
    """Fetch Cursor dashboard usage percentages. Fail-open → None."""
    session_token = extract_cursor_session_token()
    if not session_token:
        return None
    headers = {
        "Cookie": f"WorkosCursorSessionToken={session_token}",
        "Origin": "https://cursor.com",
        "Referer": "https://cursor.com/dashboard/usage",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(CURSOR_USAGE_SUMMARY_URL, headers=headers)
            response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        logger.debug("cursor ▸ usage-summary fetch failed (fail-open)", exc_info=True)
        return None


def _round_percent(value: Any) -> Optional[int]:
    if not _is_finite_percent(value):
        return None
    return max(0, min(100, round(float(value))))


def cursor_usage_lines(*, markdown: bool = False, timeout: float = 10.0) -> list[str]:
    """Return rendered Cursor included-usage /usage lines, or [] when unavailable.

    Agent-independent: reads local Cursor auth and fetches the dashboard API.
    Shared by CLI, gateway, and TUI /usage surfaces. Fail-open on any error.
    """
    payload = fetch_cursor_usage_summary(timeout=timeout)
    if not payload:
        return []

    plan = (payload.get("individualUsage") or {}).get("plan") or {}
    rows: list[tuple[str, str]] = []
    for label, key in (
        ("Total", "totalPercentUsed"),
        ("API", "apiPercentUsed"),
        ("Auto", "autoPercentUsed"),
    ):
        pct = _round_percent(plan.get(key))
        if pct is not None:
            rows.append((label, f"{pct}% used"))

    if not rows:
        return []

    header = f"📊 {'**' if markdown else ''}Cursor included usage{'**' if markdown else ''}"
    lines = [header]
    for label, text in rows:
        lines.append(f"{label}: {text}")
    return lines
