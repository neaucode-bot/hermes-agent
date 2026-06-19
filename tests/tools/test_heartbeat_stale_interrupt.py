"""Tests that the delegate heartbeat staleness monitor interrupts an idle,
wedged subagent at the idle stale limit.

Previously the monitor only logged and stopped touching the parent at the idle
threshold, relying on the gateway inactivity timeout to fire — which never
happens for a live (non-drain) session, leaving a wedged child with no
recovery. Now an *idle* stale child is interrupted via the same path the hard
timeout branch uses. A child sitting inside a long-running tool hit the much
more generous in-tool threshold instead and is NOT interrupted (it may still be
doing legitimate work).
"""

import threading

from unittest.mock import MagicMock


class _IdleStaleChild:
    """Subagent stub that hangs idle (no tool, no iteration progress) until
    the staleness monitor interrupts it."""

    def __init__(self, *, current_tool=None):
        self._subagent_id = None  # skip the live-agent registry
        self.model = "test/model"
        self.max_iterations = 30
        self._current_tool = current_tool
        self._hang = threading.Event()
        self.interrupt_called = threading.Event()

    def get_activity_summary(self):
        # api_call_count and current_tool never advance → counts as stale.
        return {
            "api_call_count": 0,
            "max_iterations": self.max_iterations,
            "current_tool": self._current_tool,
            "last_activity_desc": "",
        }

    def run_conversation(self, user_message, task_id=None, stream_callback=None):
        # Block until interrupted (or a generous safety timeout) so the test
        # finishes promptly when the monitor interrupts.
        self._hang.wait(10.0)
        return {"final_response": "", "completed": False, "api_calls": 0}

    def interrupt(self):
        self.interrupt_called.set()
        self._hang.set()


def _run(child, monkeypatch):
    from tools import delegate_tool

    # Tighten the heartbeat so the stale window elapses in milliseconds, and
    # leave the hard child timeout off so only the staleness monitor can act.
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IDLE", 2)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IN_TOOL", 2)
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: None)

    parent = MagicMock()
    parent._touch_activity = MagicMock()
    parent._current_task_id = None
    return delegate_tool._run_single_child(
        task_index=0,
        goal="test goal",
        child=child,
        parent_agent=parent,
    )


class TestHeartbeatStaleInterrupt:

    def test_idle_stale_child_is_interrupted(self, monkeypatch):
        child = _IdleStaleChild(current_tool=None)
        _run(child, monkeypatch)
        assert child.interrupt_called.is_set()

    def test_in_tool_stale_child_not_interrupted(self, monkeypatch):
        """A child stuck inside a tool hit the in-tool threshold; the monitor
        stops touching the parent but must NOT interrupt mid-tool. We release
        the child ourselves so the test can complete."""
        child = _IdleStaleChild(current_tool="terminal")

        def _release_soon():
            # Give the heartbeat loop time to reach the in-tool stale limit and
            # break without interrupting, then unblock the child.
            threading.Event().wait(0.2)
            child._hang.set()

        releaser = threading.Thread(target=_release_soon, daemon=True)
        releaser.start()
        _run(child, monkeypatch)
        releaser.join(timeout=2.0)
        assert not child.interrupt_called.is_set()
