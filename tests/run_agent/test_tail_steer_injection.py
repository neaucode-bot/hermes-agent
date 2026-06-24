"""Integration tests: the tail-pinned steering reminder in the live loop.

Exercises ``AIAgent.run_conversation`` with a mocked client and asserts that
every API call carries the steering reminder folded onto the tail message,
while the head ``system`` message stays byte-stable across turns (the sacred
prompt-cache invariant).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.steering import DEFAULT_TAIL_STEER, TAIL_STEER_OPEN
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(*, content="", finish_reason="stop", tool_calls=None):
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    choice.message.reasoning = None
    choice.message.reasoning_content = None
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None
    resp.model = "test/model"
    return resp


def _mock_tool_call(name, arguments, call_id):
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _sent_messages(agent, call_index=0):
    return agent.client.chat.completions.create.call_args_list[call_index].kwargs["messages"]


def test_tail_steer_folded_onto_user_message_every_turn(agent):
    agent.client.chat.completions.create.return_value = _mock_response(
        content="Final answer", finish_reason="stop"
    )
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    sent = _sent_messages(agent)

    # Head system message is present, first, and byte-stable.
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == "You are helpful."
    # Exactly one system message — no trailing system slot appended.
    assert sum(1 for m in sent if m.get("role") == "system") == 1
    # The reminder rides the tail (last) message, not the head.
    assert sent[-1]["role"] == "user"
    assert TAIL_STEER_OPEN in sent[-1]["content"]
    assert DEFAULT_TAIL_STEER in sent[-1]["content"]
    assert "hello" in sent[-1]["content"]
    # Head must not carry the reminder.
    assert DEFAULT_TAIL_STEER not in sent[0]["content"]


def test_head_system_byte_stable_while_tail_reminder_reemitted(agent):
    """Across a tool turn + final turn, the head system message is identical
    every call (cache stays warm) and the reminder is re-emitted each turn on
    the tail message."""
    tool_turn = _mock_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[_mock_tool_call("web_search", "{}", "c1")],
    )
    final = _mock_response(content="done", finish_reason="stop")
    agent.client.chat.completions.create.side_effect = [tool_turn, final]

    with (
        patch("run_agent.handle_function_call", return_value="search result"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert result["api_calls"] == 2

    first = _sent_messages(agent, 0)
    second = _sent_messages(agent, 1)

    # Sacred invariant: head system content identical across turns.
    assert first[0]["role"] == "system" and second[0]["role"] == "system"
    assert first[0]["content"] == second[0]["content"] == "You are helpful."

    # Turn 1: reminder folded onto the user message.
    assert TAIL_STEER_OPEN in first[-1]["content"]
    # Turn 2: tail is the tool result; reminder re-emitted there.
    assert second[-1]["role"] == "tool"
    assert TAIL_STEER_OPEN in second[-1]["content"]
    assert "search result" in second[-1]["content"]


def test_agent_tail_steer_override_replaces_default(agent):
    agent.tail_steer = "Custom per-turn steer."
    agent.client.chat.completions.create.return_value = _mock_response(
        content="ok", finish_reason="stop"
    )
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("hello")

    sent = _sent_messages(agent)
    assert "Custom per-turn steer." in sent[-1]["content"]
    assert DEFAULT_TAIL_STEER not in sent[-1]["content"]
