"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.

WS-C redaction tests verify that send_message on the direct-dispatch path
has its output scrubbed before it returns to the calling agent.
"""

from __future__ import annotations

from typing import Any




class TestModuleSurface:
    def test_module_imports_clean(self):
        from agent.transports import hermes_tools_mcp_server as m
        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_exposed_tools_are_safe_subset(self):
        """We MUST NOT expose tools codex already has, because codex'
        own builtins are better-integrated with its sandbox + approvals.
        Specifically: no terminal/shell, no read_file/write_file, no
        patch — those are codex's built-in tools."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked, (
            f"these tools must NOT be exposed via the codex callback "
            f"because codex has built-in equivalents: {leaked}"
        )

    def test_expected_hermes_specific_tools_listed(self):
        """The Hermes-specific tools should be present so users on the
        codex runtime keep access to them."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for required in (
            "web_search",
            "web_extract",
            "browser_navigate",
            "vision_analyze",
            "image_generate",
            "skill_view",
        ):
            assert required in EXPOSED_TOOLS, f"missing {required!r}"

    def test_agent_loop_tools_not_exposed(self):
        """delegate_task / memory / session_search / todo require the
        running AIAgent context to dispatch, so a stateless MCP callback
        can't drive them. They must NOT be in EXPOSED_TOOLS."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for agent_loop_tool in ("delegate_task", "memory", "session_search", "todo"):
            assert agent_loop_tool not in EXPOSED_TOOLS, (
                f"{agent_loop_tool!r} requires the agent loop context "
                "and can't be reached through a stateless MCP callback"
            )

    def test_kanban_worker_tools_exposed(self):
        """Kanban workers run as `hermes chat -q` subprocesses; if they
        come up on the codex_app_server runtime, the worker can do the
        actual work via codex's shell but needs the kanban tools through
        the MCP callback to report back to the kernel. Without these
        tools available, the worker would hang at completion time."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        # Worker handoff tools — every dispatched worker uses at least
        # one of {complete, block, comment} to close out its task.
        for worker_tool in (
            "kanban_complete",
            "kanban_block",
            "kanban_comment",
            "kanban_heartbeat",
        ):
            assert worker_tool in EXPOSED_TOOLS, (
                f"{worker_tool!r} missing from codex callback — kanban "
                "workers on codex_app_server runtime would hang"
            )

    def test_kanban_orchestrator_tools_exposed(self):
        """Orchestrator agents need to dispatch new tasks, query the
        board, and unblock/link tasks. Exposed so an orchestrator on
        codex_app_server can do its job."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for orch_tool in (
            "kanban_create",
            "kanban_show",
            "kanban_list",
            "kanban_unblock",
            "kanban_link",
        ):
            assert orch_tool in EXPOSED_TOOLS, (
                f"{orch_tool!r} missing from codex callback"
            )


class TestRedactToolOutput:
    """_redact_tool_output scrubs secrets before any result leaves the process."""

    def test_redacts_sk_key_in_output(self):
        import os
        os.environ.setdefault("HERMES_REDACT_SECRETS", "true")
        from agent.transports.hermes_tools_mcp_server import _redact_tool_output
        planted = "Message sent. Echo: my key is sk-abcdef1234567890XXXX"
        result = _redact_tool_output(planted)
        assert "sk-abcdef1234567890XXXX" not in result, (
            "sk- API key must be redacted from tool output"
        )

    def test_passthrough_on_non_string(self):
        from agent.transports.hermes_tools_mcp_server import _redact_tool_output
        assert _redact_tool_output(42) == 42
        assert _redact_tool_output(None) is None

    def test_clean_text_unchanged(self):
        from agent.transports.hermes_tools_mcp_server import _redact_tool_output
        clean = "Message delivered to #general."
        assert _redact_tool_output(clean) == clean


class TestGuardedDirectDispatch:
    """send_message on the direct-dispatch path goes through output redaction."""

    def test_send_message_output_redacted(self, monkeypatch):
        """Plant a sk- secret in send_message's return value; verify it is
        scrubbed before the result is returned to the calling agent."""
        import os
        os.environ["HERMES_REDACT_SECRETS"] = "true"

        planted_secret = "sk-abcdef1234567890PLANTED"

        import agent.transports.hermes_tools_mcp_server as m

        # Stub _guarded_direct_dispatch to a simple pass-through so this test
        # only exercises the _register redaction wrapper, not middleware import.
        def _fake_guarded(tool_name, handler):
            return handler

        monkeypatch.setattr(m, "_guarded_direct_dispatch", _fake_guarded)

        # Stub out _direct_tools to return a handler that embeds the secret.
        def _fake_direct_tools():
            return {
                "send_message": (
                    "Send a message",
                    lambda args: f"Sent. Your key: {planted_secret}",
                ),
            }

        monkeypatch.setattr(m, "_direct_tools", _fake_direct_tools)

        # Build a minimal fake FastMCP that records registered tools.
        registered: dict[str, Any] = {}

        class _FakeMCP:
            def add_tool(self, fn, *, name, description, **kw):
                registered[name] = fn

            def tool(self, *, name, description):
                def _dec(fn):
                    registered[name] = fn
                    return fn
                return _dec

        import sys
        fake_mcp_mod = type(sys)("mcp.server.fastmcp")
        fake_mcp_mod.FastMCP = lambda *a, **kw: _FakeMCP()

        # Stub model_tools so _build_server doesn't need the full Hermes init.
        fake_model_tools = type(sys)("model_tools")
        fake_model_tools.get_tool_definitions = lambda **kw: []
        fake_model_tools.handle_function_call = lambda name, args: ""

        with monkeypatch.context() as ctx:
            ctx.setitem(sys.modules, "mcp.server.fastmcp", fake_mcp_mod)
            ctx.setitem(sys.modules, "mcp", type(sys)("mcp"))
            ctx.setitem(sys.modules, "model_tools", fake_model_tools)
            m._build_server()

        assert "send_message" in registered, "send_message must be registered"
        result = registered["send_message"]()
        assert planted_secret not in result, (
            f"sk- secret must be redacted from send_message output; got: {result!r}"
        )

    def test_guarded_dispatch_is_used_for_send_message(self, monkeypatch):
        """Verify _guarded_direct_dispatch is called for send_message (not bare
        handler), ensuring middleware + approval gate are wired in."""
        import sys
        import agent.transports.hermes_tools_mcp_server as m

        guarded_names: list[str] = []

        original_guarded = m._guarded_direct_dispatch

        def _spy_guarded(tool_name, handler):
            guarded_names.append(tool_name)
            return original_guarded(tool_name, handler)

        monkeypatch.setattr(m, "_guarded_direct_dispatch", _spy_guarded)

        # Stub _direct_tools to avoid deep imports that fail on older Python.
        monkeypatch.setattr(m, "_direct_tools", lambda: {
            "send_message": ("Send a message", lambda args: "ok"),
            "memory": ("Memory", lambda args: "ok"),
            "session_search": ("Search sessions", lambda args: "ok"),
        })

        class _FakeMCP:
            def add_tool(self, fn, *, name, description, **kw): pass
            def tool(self, *, name, description):
                def _dec(fn): return fn
                return _dec

        fake_mcp_mod = type(sys)("mcp.server.fastmcp")
        fake_mcp_mod.FastMCP = lambda *a, **kw: _FakeMCP()

        fake_model_tools = type(sys)("model_tools")
        fake_model_tools.get_tool_definitions = lambda **kw: []
        fake_model_tools.handle_function_call = lambda name, args: ""

        with monkeypatch.context() as ctx:
            ctx.setitem(sys.modules, "mcp.server.fastmcp", fake_mcp_mod)
            ctx.setitem(sys.modules, "mcp", type(sys)("mcp"))
            ctx.setitem(sys.modules, "model_tools", fake_model_tools)
            m._build_server()

        assert "send_message" in guarded_names, (
            "send_message must be routed through _guarded_direct_dispatch"
        )


class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        """When the mcp package isn't installed, main() should exit
        cleanly with code 2 and an install hint, not crash."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1
