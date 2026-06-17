#!/usr/bin/env python3
"""Tests for subagent handoff parse/strip helpers."""

import json
import unittest
from types import SimpleNamespace

from tools.subagent_handoff import (
    consume_delegate_handoff,
    extract_cursor_meta_from_response,
    extract_handoff_block,
    parse_handoff,
    strip_handoff_fence,
)


def _fence(payload):
    return "```handoff\n" + json.dumps(payload, indent=2) + "\n```"


_MINIMAL = {
    "schema_version": "1.0",
    "status": "done",
    "summary": "Did the thing.",
    "artifacts": [],
}


class TestSubagentHandoff(unittest.TestCase):
    def test_strip_removes_fence_preserves_prose(self):
        text = "Narrative here.\n\n" + _fence(_MINIMAL)
        self.assertEqual(strip_handoff_fence(text), "Narrative here.")

    def test_extract_last_fence_wins(self):
        first = _fence({**_MINIMAL, "summary": "draft"})
        second = _fence({**_MINIMAL, "summary": "final"})
        body = extract_handoff_block(first + "\n\n" + second)
        self.assertEqual(json.loads(body)["summary"], "final")

    def test_parse_minimal_done(self):
        result = parse_handoff(_fence(_MINIMAL))
        self.assertTrue(result["ok"])
        self.assertEqual(result["report"]["status"], "done")

    def test_parse_no_fence_degrades(self):
        result = parse_handoff("prose only")
        self.assertFalse(result["ok"])
        self.assertEqual(result["report"]["status"], "partial")
        self.assertTrue(result["report"].get("_degraded"))

    def test_consume_prefers_cursor_meta(self):
        raw = "Worker narrative.\n\n" + _fence(_MINIMAL)
        cursor_meta = {
            "handoff": {
                "schema_version": "1.0",
                "status": "partial",
                "summary": "From cursor meta.",
                "artifacts": [{"id": "a", "kind": "file", "handle": "/tmp/x", "mutated": True}],
            }
        }
        clean, payload = consume_delegate_handoff(raw, cursor_meta)
        self.assertEqual(clean, "Worker narrative.")
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["summary"], "From cursor meta.")
        self.assertEqual(len(payload["artifacts"]), 1)
        self.assertNotIn("```handoff", clean)

    def test_extract_cursor_meta_from_response(self):
        response = SimpleNamespace(
            cursor={"handoff": {"status": "done", "summary": "x", "artifacts": []}}
        )
        meta = extract_cursor_meta_from_response(response)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["handoff"]["status"], "done")


if __name__ == "__main__":
    unittest.main()
