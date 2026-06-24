"""Hermes-tools-as-MCP server for the codex_app_server runtime.

When the user runs `openai/*` turns through the codex app-server, codex
owns the loop and builds its own tool list. By default, that means
Hermes' richer tool surface — web search, browser automation,
delegate_task subagents, vision analysis, persistent memory, skills,
cross-session search, image generation, TTS — is unreachable.

This module exposes a curated subset of those Hermes tools to the
spawned codex subprocess via stdio MCP. Codex registers it as a normal
MCP server (per `~/.codex/config.toml [mcp_servers.hermes-tools]`) and
the user gets full Hermes capability inside a Codex turn.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list, skill_manage — Hermes' skill library
    (skill_manage create/edit/patch/delete is stateless: validates +
    writes ~/.hermes/skills/ on disk, not `_AGENT_LOOP_TOOLS`-gated)
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

Orchestration tools (DIRECT_TOOLS): send_message, memory, session_search.
These are either unregistered (send_message) or `_AGENT_LOOP_TOOLS`-gated
(memory, session_search) in Hermes, so the stateless
`handle_function_call()` path refuses them. But each one CAN run
statelessly when given a freshly-built context, so we dispatch them
directly to their underlying tool functions (see DIRECT_TOOLS below). This
is what lets a Cursor-native agent act as the Jarvis orchestrator —
read/scroll prior sessions, persist durable memory, and message users.

What we DO NOT expose:
  - terminal / shell                     — codex's own shell tool
  - read_file / write_file / patch       — codex's apply_patch + shell
  - search_files / process               — codex's shell
  - clarify                              — codex's own UX
  - delegate_task                        — Cursor-native agents already have
                                           a native task/delegation tool;
                                           Hermes delegate_task also needs the
                                           running AIAgent loop to spawn and
                                           track subagents, so exposing it
                                           here would be redundant/conflicting.
  - todo                                 — `_AGENT_LOOP_TOOLS`; the agent's
                                           todo list is loop-resident state
                                           with no stateless backing store.
                                           Cursor provides a native todo tool.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: CodexAppServerSession.ensure_started() when the runtime is
            active and config opts in.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Registry-dispatched tools. Each name MUST match a registered Hermes tool
# that `model_tools.handle_function_call()` can dispatch statelessly.
#
# What we deliberately DO NOT expose here:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — the host agent's built-ins cover these and approval routes
#     through the host's own UI.
#   - delegate_task / todo — delegate_task is redundant with Cursor's native
#     task tool and needs the live AIAgent loop to spawn/track subagents;
#     todo is `_AGENT_LOOP_TOOLS` loop-resident state with no stateless
#     backing store (Cursor has a native todo tool). See module docstring.
#
# Orchestration tools that ARE reachable statelessly (send_message, memory,
# session_search) are wired separately via DIRECT_TOOLS below, because the
# stateless `handle_function_call()` path refuses them.
#
# HERMES_CURSOR_MODE=1: set by .cursor/mcp.json when launched from the
# Cursor IDE. Suppresses browser_* tools (Hermes CloakBrowser stack) because
# Cursor already provides cursor-ide-browser for standard automation; the
# dual stack adds ~2,000 tok/turn with no benefit for most IDE tasks.
# CloakBrowser (stealth auth/signup) is still reachable from the Hermes
# TUI/gateway path where this env var is not set.
_CURSOR_MODE: bool = os.environ.get("HERMES_CURSOR_MODE", "") == "1"

# The exposed tool surface is DERIVED from the canonical role-tagged manifest
# in ``agent.hermes_tool_surface`` (the single source of truth shared with the
# proxy/orchestrator side). The lists below are byte-equivalent to the previous
# hand-written tuples — see that module for the per-tool rationale, the
# intentional exclusions (delegate_task / todo / terminal / file / search), and
# the role tags.
#
# ``EXPOSED_TOOLS``  = registry-dispatched IDE tools (filters role ⊇ {ide},
#                      dispatch == "registry"). browser_* tools are suppressed
#                      when HERMES_CURSOR_MODE=1, exactly as before.
# ``_DIRECT_TOOL_NAMES`` = direct-dispatch IDE orchestration tools
#                      (send_message, memory, session_search).
from agent.hermes_tool_surface import (
    ide_direct_tools as _surface_ide_direct_tools,
    ide_registry_tools as _surface_ide_registry_tools,
)

# Hermes browser_* tools — exposed in gateway/TUI mode, suppressed in Cursor
# IDE mode (cursor-ide-browser covers that lane). Derived from the manifest's
# ``cursor_suppressed`` flag; kept as a module constant for back-compat.
_HERMES_BROWSER_TOOLS: tuple[str, ...] = tuple(
    name
    for name in _surface_ide_registry_tools(cursor_mode=False)
    if name not in _surface_ide_registry_tools(cursor_mode=True)
)

EXPOSED_TOOLS: tuple[str, ...] = _surface_ide_registry_tools(cursor_mode=_CURSOR_MODE)

# Direct-dispatch orchestration tool names (handlers built in _direct_tools()).
_DIRECT_TOOL_NAMES: tuple[str, ...] = _surface_ide_direct_tools()


def _direct_tools() -> dict[str, tuple[str, Any]]:
    """Orchestration tools dispatched directly to their underlying functions,
    bypassing the stateless ``handle_function_call()`` guard.

    Returned as ``{name: (description, handler)}`` where ``handler`` takes the
    arguments dict and returns the tool-result string. Each handler builds the
    minimal context the tool needs so it runs without a live AIAgent loop:
      - send_message  — not registered as a model tool by design; we call the
                        shared transport directly (same as mcp_serve.py).
      - memory        — `_AGENT_LOOP_TOOLS`-gated; the live loop binds a
                        MemoryStore. We build a fresh one from disk; writes are
                        atomic + file-locked, so they're safe alongside a live
                        agent (its frozen system-prompt snapshot refreshes next
                        session, matching memory's documented behavior).
      - session_search — `_AGENT_LOOP_TOOLS`-gated; it already falls back to a
                        default SessionDB when no db is injected, so it reads
                        session history/index fine statelessly.
    """
    from tools.memory_tool import MEMORY_SCHEMA, MemoryStore, memory_tool
    from tools.send_message_tool import SEND_MESSAGE_SCHEMA, send_message_tool
    from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search

    def _dispatch_send_message(args: dict[str, Any]) -> str:
        return send_message_tool(args or {})

    def _dispatch_memory(args: dict[str, Any]) -> str:
        args = args or {}
        store = MemoryStore()
        store.load_from_disk()
        return memory_tool(
            action=args.get("action"),
            target=args.get("target", "memory"),
            content=args.get("content"),
            old_text=args.get("old_text"),
            operations=args.get("operations"),
            store=store,
        )

    def _dispatch_session_search(args: dict[str, Any]) -> str:
        args = args or {}
        return session_search(
            query=args.get("query") or "",
            role_filter=args.get("role_filter"),
            limit=args.get("limit", 3),
            session_id=args.get("session_id"),
            around_message_id=args.get("around_message_id"),
            window=args.get("window", 5),
            sort=args.get("sort"),
            profile=args.get("profile"),
        )

    # Membership + order are driven by the canonical manifest
    # (``_DIRECT_TOOL_NAMES``); the bespoke handlers below stay here because
    # each builds the minimal stateless context its tool needs.
    handlers: dict[str, tuple[str, Any]] = {
        "send_message": (SEND_MESSAGE_SCHEMA["description"], _dispatch_send_message),
        "memory": (MEMORY_SCHEMA["description"], _dispatch_memory),
        "session_search": (SESSION_SEARCH_SCHEMA["description"], _dispatch_session_search),
    }
    return {name: handlers[name] for name in _DIRECT_TOOL_NAMES if name in handlers}


def _redact_tool_output(result: Any) -> Any:
    """Scrub secrets from a tool-result string before it leaves this process.

    The normal Hermes agent loop runs every tool result through
    ``redact_sensitive_text`` before the model (or anything downstream) sees
    it. The MCP server returns results straight over the wire to the calling
    Cursor/Codex agent, so we apply the same scrubber here to preserve that
    guarantee. ``redact_sensitive_text`` honours ``HERMES_REDACT_SECRETS``
    (defaulted to ``true`` in ``main()``) and is a cheap no-op on non-matching
    text, so this is safe for every exposed tool. Fail-open if the redactor
    can't be imported — never block a tool result on the scrubber.
    """
    if not isinstance(result, str):
        return result
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(result)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("output redaction unavailable: %s", exc)
        return result


def _guarded_direct_dispatch(tool_name: str, terminal_fn: Any) -> Any:
    """Wrap a stateless direct-dispatch terminal call so it runs through the
    same protective middleware the normal ``model_tools.handle_function_call()``
    path applies — tool_request middleware, the pre_tool_call block hook, ACP
    edit approval, tool_execution middleware, the post_tool_call observer hook,
    and the transform_tool_result seam — WITHOUT reintroducing the
    ``_AGENT_LOOP_TOOLS`` guard that makes these tools uncallable statelessly.

    This matters most for ``send_message``, which has an external side-effect:
    it actually delivers a message to the user over Telegram/Slack/etc. The
    bare direct-dispatch path skipped all of these protections. Output
    redaction is applied once, centrally, by the shared ``_register`` dispatch
    wrapper, so it is intentionally not repeated here.

    Every protective step fails open (logs + continues) so a missing optional
    subsystem can never make an otherwise-valid orchestration call unusable —
    the terminal tool still runs.
    """

    def _run(args: dict[str, Any]) -> str:
        args = args or {}
        original_args = dict(args)
        middleware_trace: list[dict[str, Any]] = []

        # 1. tool_request middleware — may rewrite args before anything sees them.
        try:
            from hermes_cli.middleware import apply_tool_request_middleware
            mw = apply_tool_request_middleware(tool_name, args)
            if isinstance(mw.payload, dict):
                args = mw.payload
            original_args = mw.original_payload
            middleware_trace = mw.trace
        except Exception as exc:
            logger.debug("tool_request middleware error for %s: %s", tool_name, exc)

        # 2. pre_tool_call block hook — a plugin may veto the call.
        block_message: Optional[str] = None
        try:
            from hermes_cli.plugins import get_pre_tool_call_block_message
            block_message = get_pre_tool_call_block_message(
                tool_name, args, middleware_trace=list(middleware_trace),
            )
        except Exception as exc:
            logger.debug("pre_tool_call hook error for %s: %s", tool_name, exc)
        if block_message is not None:
            result = json.dumps({"error": block_message}, ensure_ascii=False)
            try:
                from model_tools import _emit_post_tool_call_hook
                _emit_post_tool_call_hook(
                    function_name=tool_name,
                    function_args=args,
                    result=result,
                    status="blocked",
                    error_type="plugin_block",
                    error_message=block_message,
                    middleware_trace=list(middleware_trace),
                )
            except Exception as exc:
                logger.debug("post_tool_call hook error for %s: %s", tool_name, exc)
            return result

        # 3. ACP/Zed edit approval — a no-op for these non-file tools, but kept
        #    so the path mirrors handle_function_call() exactly.
        try:
            from acp_adapter.edit_approval import maybe_require_edit_approval
            edit_block_message = maybe_require_edit_approval(tool_name, args)
            if edit_block_message is not None:
                return edit_block_message
        except Exception as exc:
            logger.debug("edit approval guard error for %s: %s", tool_name, exc)

        # 4. tool_execution middleware wrapping the terminal direct call.
        dispatch_start = time.monotonic()
        try:
            from hermes_cli.middleware import run_tool_execution_middleware
            result = run_tool_execution_middleware(
                tool_name,
                args,
                lambda next_args: terminal_fn(
                    next_args if isinstance(next_args, dict) else args
                ),
                original_args=original_args,
            )
        except Exception as exc:
            # If the execution-middleware machinery itself is unavailable, fall
            # back to the bare terminal call so the tool still works.
            logger.debug("tool_execution middleware error for %s: %s", tool_name, exc)
            result = terminal_fn(args)
        duration_ms = int((time.monotonic() - dispatch_start) * 1000)

        # 5. post_tool_call observer hook.
        try:
            from model_tools import _emit_post_tool_call_hook
            _emit_post_tool_call_hook(
                function_name=tool_name,
                function_args=args,
                result=result,
                duration_ms=duration_ms,
                middleware_trace=list(middleware_trace),
            )
        except Exception as exc:
            logger.debug("post_tool_call hook error for %s: %s", tool_name, exc)

        # 6. transform_tool_result seam — plugins may canonicalize the result.
        try:
            from hermes_cli.plugins import has_hook, invoke_hook
            if has_hook("transform_tool_result"):
                hook_results = invoke_hook(
                    "transform_tool_result",
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    duration_ms=duration_ms,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        result = hook_result
                        break
        except Exception as exc:
            logger.debug("transform_tool_result hook error for %s: %s", tool_name, exc)

        return result

    return _run


def _build_server() -> Any:
    """Create the FastMCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            f"hermes-tools MCP server requires the 'mcp' package: {exc}"
        ) from exc

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    mcp = FastMCP(
        "hermes-tools",
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "subagent delegation, vision, image generation, persistent "
            "memory, skills, and cross-session search."
        ),
    )

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    # Build a closure that takes the arguments dict, runs ``dispatch_fn``, and
    # returns the result string. We register via add_tool()/tool() with a
    # **kwargs signature so FastMCP accepts arbitrary tool arguments (the rich
    # parameter docs live in the description text).
    def _register(tool_name: str, description: str, dispatch_fn: Any) -> None:
        def _dispatch(**kwargs: Any) -> str:
            try:
                args = kwargs or {}
                # FastMCP introspects this **kwargs signature into a single
                # ``kwargs`` object property, so MCP clients deliver the real
                # arguments nested under that key. Unwrap it so the underlying
                # tool sees its actual parameters.
                if set(args) == {"kwargs"} and isinstance(args["kwargs"], dict):
                    args = args["kwargs"]
                # Scrub secrets from every tool result before it leaves the
                # process, matching the agent loop's redaction guarantee.
                return _redact_tool_output(dispatch_fn(args))
            except Exception as exc:
                logger.exception("tool %s raised", tool_name)
                return json.dumps({"error": str(exc), "tool": tool_name})
        _dispatch.__name__ = tool_name
        _dispatch.__doc__ = description
        try:
            mcp.add_tool(_dispatch, name=tool_name, description=description)
        except TypeError:
            # Older mcp SDK signature — fall back to decorator-style.
            mcp.tool(name=tool_name, description=description)(_dispatch)

    exposed_count = 0

    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"

        def _make_registry_dispatch(tool_name: str):
            def _via_registry(args: dict[str, Any]) -> str:
                return handle_function_call(tool_name, args)
            return _via_registry

        _register(name, description, _make_registry_dispatch(name))
        exposed_count += 1

    # Direct-dispatch orchestration tools (send_message, memory,
    # session_search). These bypass the stateless handle_function_call()
    # registry guard that would otherwise refuse them (see _direct_tools()),
    # but we still route each one through _guarded_direct_dispatch() so it gets
    # the same protective middleware the normal path applies — tool_request
    # middleware, pre/post plugin hooks, edit approval, and the
    # transform_tool_result seam. Output redaction is added by _register's
    # _dispatch wrapper. This is essential for send_message, which produces an
    # external side-effect.
    direct_count = 0
    for name, (description, handler) in _direct_tools().items():
        _register(name, description, _guarded_direct_dispatch(name, handler))
        direct_count += 1

    logger.info(
        "hermes-tools MCP server registered %d/%d registry tools + %d direct tools",
        exposed_count,
        len(EXPOSED_TOOLS),
        direct_count,
    )
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet mode: keep Hermes' own banners off stdout (which is the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2

    # FastMCP runs with stdio transport by default when launched as a
    # subprocess.
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
