"""Tests for the session_title tool (get/set current session display title)."""

import json

import pytest

from hermes_state import SessionDB
from tools.session_title_tool import SESSION_TITLE_SCHEMA, session_title


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


class TestSchema:
    def test_name_and_optional_title(self):
        assert SESSION_TITLE_SCHEMA["name"] == "session_title"
        assert SESSION_TITLE_SCHEMA["parameters"]["required"] == []


class TestSessionTitle:
    def test_get_when_unset(self, db):
        db.create_session("s1", source="cli")
        out = json.loads(session_title(db=db, current_session_id="s1"))
        assert out["success"] is True
        assert out["action"] == "get"
        assert out["title"] is None

    def test_set_and_get(self, db):
        db.create_session("s1", source="cli")
        set_out = json.loads(
            session_title(title="Proxy tool routing", db=db, current_session_id="s1")
        )
        assert set_out["success"] is True
        assert set_out["action"] == "set"
        assert set_out["title"] == "Proxy tool routing"

        get_out = json.loads(session_title(db=db, current_session_id="s1"))
        assert get_out["title"] == "Proxy tool routing"

    def test_set_creates_missing_session_row(self, db):
        out = json.loads(
            session_title(title="Early title", db=db, current_session_id="new-session")
        )
        assert out["success"] is True
        assert db.get_session_title("new-session") == "Early title"

    def test_rejects_empty_title(self, db):
        db.create_session("s1", source="cli")
        out = json.loads(session_title(title="   ", db=db, current_session_id="s1"))
        assert out["success"] is False

    def test_requires_session_id(self, db):
        out = json.loads(session_title(title="Nope", db=db, current_session_id=None))
        assert out["success"] is False

    def test_duplicate_title_conflict(self, db):
        db.create_session("s1", source="cli")
        db.create_session("s2", source="cli")
        db.set_session_title("s1", "Taken name")
        out = json.loads(
            session_title(title="Taken name", db=db, current_session_id="s2")
        )
        assert out["success"] is False
        assert "already in use" in out["error"]
