"""Tests for the pinned tail-steering reminder (agent/steering.py).

The tail steer is Hermes's true tail-reminder slot: a short reminder re-emitted
every turn AFTER the conversation history, past the cached head prefix, so the
byte-stable head ``system`` message is never mutated (prompt caching stays
warm). It is folded onto the last user/tool message rather than appended as a
trailing ``system`` message, which native-Anthropic adapters would fold back
into the cached head param.
"""

from __future__ import annotations

import copy

from agent.steering import (
    DEFAULT_TAIL_STEER,
    TAIL_STEER_CLOSE,
    TAIL_STEER_OPEN,
    format_tail_steer,
    inject_tail_steer,
)


class TestCanonicalText:
    def test_default_steer_is_verbatim_contract_line(self):
        # Worker A embeds this identical line in the generated contract — it
        # MUST stay byte-for-byte identical or the IDE and engine pin
        # different reminders.
        assert DEFAULT_TAIL_STEER == (
            "Re-read your system prompt and current objective "
            "before taking your next action."
        )

    def test_format_wraps_in_reminder_fence(self):
        out = format_tail_steer("hello")
        assert out == f"{TAIL_STEER_OPEN}\nhello\n{TAIL_STEER_CLOSE}"


class TestInjection:
    def test_folds_onto_last_user_message(self):
        msgs = [
            {"role": "system", "content": "HEAD PROMPT"},
            {"role": "user", "content": "do the thing"},
        ]
        out = inject_tail_steer(msgs)
        # Head system message is untouched (cache prefix stays byte-stable).
        assert out[0] == {"role": "system", "content": "HEAD PROMPT"}
        # Reminder folded onto the last user message.
        assert "do the thing" in out[-1]["content"]
        assert DEFAULT_TAIL_STEER in out[-1]["content"]
        assert TAIL_STEER_OPEN in out[-1]["content"]
        assert out[-1]["role"] == "user"

    def test_folds_onto_last_tool_message_on_tool_iterations(self):
        msgs = [
            {"role": "system", "content": "HEAD"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "tool output", "tool_call_id": "a"},
        ]
        out = inject_tail_steer(msgs, "custom steer")
        assert out[-1]["role"] == "tool"
        assert "tool output" in out[-1]["content"]
        assert "custom steer" in out[-1]["content"]

    def test_never_appends_a_trailing_system_message(self):
        # A trailing system message would be folded back into the cached head
        # param by native-Anthropic adapters, dropping the real head prompt.
        msgs = [
            {"role": "system", "content": "HEAD"},
            {"role": "user", "content": "q"},
        ]
        out = inject_tail_steer(msgs)
        system_msgs = [m for m in out if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "HEAD"

    def test_multimodal_list_content_gets_text_block_appended(self):
        original = [{"type": "text", "text": "existing"}]
        msgs = [{"role": "user", "content": copy.deepcopy(original)}]
        out = inject_tail_steer(msgs, "note")
        content = out[-1]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "existing"}
        assert content[-1]["type"] == "text"
        assert "note" in content[-1]["text"]

    def test_does_not_mutate_shared_list_content(self):
        # api_messages elements are shallow copies of history; list content is
        # shared by reference, so the injector must NOT mutate it in place.
        shared = [{"type": "text", "text": "existing"}]
        msgs = [{"role": "user", "content": shared}]
        inject_tail_steer(msgs, "note")
        assert shared == [{"type": "text", "text": "existing"}]

    def test_empty_steer_is_noop(self):
        msgs = [{"role": "user", "content": "q"}]
        before = copy.deepcopy(msgs)
        assert inject_tail_steer(msgs, "") == before
        assert inject_tail_steer(msgs, "   \n ") == before

    def test_no_user_or_tool_message_is_noop(self):
        msgs = [{"role": "system", "content": "HEAD"}]
        before = copy.deepcopy(msgs)
        out = inject_tail_steer(msgs)
        assert out == before

    def test_empty_string_content_user_message(self):
        msgs = [{"role": "user", "content": ""}]
        out = inject_tail_steer(msgs, "steer")
        assert out[-1]["content"] == format_tail_steer("steer")

    def test_folds_onto_last_when_assistant_is_tail(self):
        # If the tail is an assistant message (rare at the API-build point),
        # fold onto the most recent user/tool message instead.
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "thinking"},
        ]
        out = inject_tail_steer(msgs, "steer")
        assert out[1] == {"role": "assistant", "content": "thinking"}
        assert "steer" in out[0]["content"]
