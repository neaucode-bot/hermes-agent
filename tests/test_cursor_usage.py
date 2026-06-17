"""Tests for agent/cursor_usage.py — Cursor dashboard included-usage overlay."""

from __future__ import annotations

import base64
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from agent.cursor_usage import (
    cursor_usage_lines,
    extract_cursor_session_token,
    fetch_cursor_usage_summary,
)


def _make_jwt(*, sub: str = "auth0|user_abc123") -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, *, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, url: str, headers: dict) -> _Response:
        return _Response(self._payload)


def test_extract_cursor_session_token_from_db(tmp_path):
    db_path = tmp_path / "state.vscdb"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value BLOB)")
        token = _make_jwt()
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            ("cursorAuth/accessToken", token.encode()),
        )
        conn.commit()
    finally:
        conn.close()

    with patch("agent.cursor_usage._cursor_state_db_path", return_value=str(db_path)):
        session = extract_cursor_session_token()

    assert session == f"user_abc123::{token}"


def test_extract_cursor_session_token_missing_db(tmp_path):
    with patch("agent.cursor_usage._cursor_state_db_path", return_value=str(tmp_path / "missing.vscdb")):
        assert extract_cursor_session_token() is None


def test_fetch_cursor_usage_summary_success():
    payload = {
        "individualUsage": {
            "plan": {
                "totalPercentUsed": 42.6,
                "apiPercentUsed": 10.2,
                "autoPercentUsed": 32.4,
            }
        }
    }

    with patch("agent.cursor_usage.extract_cursor_session_token", return_value="user::token"), patch(
        "agent.cursor_usage.httpx.Client", return_value=_Client(payload=payload)
    ):
        result = fetch_cursor_usage_summary()

    assert result == payload


def test_fetch_cursor_usage_summary_no_token():
    with patch("agent.cursor_usage.extract_cursor_session_token", return_value=None):
        assert fetch_cursor_usage_summary() is None


def test_cursor_usage_lines_renders_rounded_percentages():
    payload = {
        "individualUsage": {
            "plan": {
                "totalPercentUsed": 42.6,
                "apiPercentUsed": 10.2,
                "autoPercentUsed": 32.4,
            }
        }
    }

    with patch("agent.cursor_usage.fetch_cursor_usage_summary", return_value=payload):
        lines = cursor_usage_lines()

    assert lines[0] == "📊 Cursor included usage"
    assert "Total: 43% used" in lines
    assert "API: 10% used" in lines
    assert "Auto: 32% used" in lines


def test_cursor_usage_lines_markdown_bold_header():
    payload = {
        "individualUsage": {
            "plan": {"totalPercentUsed": 5.0},
        }
    }

    with patch("agent.cursor_usage.fetch_cursor_usage_summary", return_value=payload):
        lines = cursor_usage_lines(markdown=True)

    assert lines[0] == "📊 **Cursor included usage**"


def test_cursor_usage_lines_fail_open():
    with patch("agent.cursor_usage.fetch_cursor_usage_summary", return_value=None):
        assert cursor_usage_lines() == []


@pytest.mark.integration
def test_cursor_usage_lines_live_fetch():
    """Optional live check when Cursor is installed and signed in on macOS."""
    lines = cursor_usage_lines()
    if not lines:
        pytest.skip("Cursor auth unavailable on this host")
    assert lines[0].startswith("📊 Cursor included usage")
    assert any("% used" in line for line in lines)
