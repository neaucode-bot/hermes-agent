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
  - skill_view, skills_list              — Hermes' skill library
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
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_extract",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
    "browser_get_images",
    "browser_console",
    "browser_vision",
    "vision_analyze",
    "image_generate",
    "skill_view",
    "skills_list",
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Without these
    # in the callback, a worker spawned with openai_runtime=codex_app_server
    # could do the work but couldn't report completion back to the kernel,
    # making it hang until timeout. Stateless dispatch — they just read
    # the env var and write to ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running on the codex
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)


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

    return {
        "send_message": (SEND_MESSAGE_SCHEMA["description"], _dispatch_send_message),
        "memory": (MEMORY_SCHEMA["description"], _dispatch_memory),
        "session_search": (SESSION_SEARCH_SCHEMA["description"], _dispatch_session_search),
    }


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
                return dispatch_fn(args)
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
    # session_search) — these bypass the stateless handle_function_call()
    # guard that would otherwise refuse them. See _direct_tools().
    direct_count = 0
    for name, (description, handler) in _direct_tools().items():
        _register(name, description, handler)
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
