#!/usr/bin/env python3
"""Tests for subagent handoff parse/strip helpers."""

import json
import unittest
from types import SimpleNamespace

from tools.subagent_handoff import (
    PROSE_ONLY_NOTE,
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

    def test_parse_prose_only_is_done_not_failure(self):
        # A leaf that answered in prose but skipped the fence should read as a
        # completed answer carrying that prose, not an alarming partial/failure.
        result = parse_handoff("Here are my instructions: be helpful.")
        self.assertTrue(result["ok"])
        self.assertEqual(result["report"]["status"], "done")
        self.assertEqual(
            result["report"]["summary"], "Here are my instructions: be helpful."
        )
        # Low-severity signal is preserved, but no alarming unresolved item.
        self.assertTrue(result["report"].get("_degraded"))
        self.assertIn(PROSE_ONLY_NOTE, result["report"].get("_warnings", []))
        self.assertNotIn("unresolved", result["report"])

    def test_parse_empty_output_degrades(self):
        # Genuinely empty output keeps the partial/failure framing.
        result = parse_handoff("")
        self.assertFalse(result["ok"])
        self.assertEqual(result["report"]["status"], "partial")
        self.assertTrue(result["report"].get("_degraded"))
        self.assertTrue(result["report"].get("unresolved"))

    def test_consume_prose_only_presents_clean_summary(self):
        raw = "Worker answered in prose with no fence."
        clean, payload = consume_delegate_handoff(raw, None)
        self.assertEqual(clean, raw)
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["summary"], raw)
        self.assertTrue(payload.get("degraded"))
        self.assertIn(PROSE_ONLY_NOTE, payload.get("warnings", []))
        self.assertNotIn("unresolved", payload)

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
