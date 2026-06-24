"""Canonical Hermes-brain tool surface — one role-tagged manifest.

This is the single source of truth for *which Hermes tools exist* and *which
interfaces (roles) expose them*. Today the same information is duplicated and
drifts across two hand-maintained lists:

  * the MCP server ``EXPOSED_TOOLS`` / direct-dispatch tools
    (``agent.transports.hermes_tools_mcp_server``) — the IDE / Codex surface;
  * the orchestrator ``customTools`` set on the proxy side
    (``cursor-openai-api/.../prompt.ts``) — the engine-orchestrator surface.

The manifest below lets each consumer *derive* its list from one place.

Roles
-----
* ``"ide"``          — exposed through the ``hermes-tools`` MCP server to a
                       Cursor IDE / Codex session.
* ``"orchestrator"`` — exposed to the engine orchestrator via the proxy's
                       ``customTools`` bridge.

Dispatch (how the IDE MCP server invokes the tool)
--------------------------------------------------
* ``"registry"`` — dispatched statelessly via
                   ``model_tools.handle_function_call()`` (the normal path,
                   full middleware stack).
* ``"direct"``   — orchestration tools that are unregistered or
                   ``_AGENT_LOOP_TOOLS``-gated and must be dispatched to their
                   underlying function with a freshly-built context (see
                   ``hermes_tools_mcp_server._direct_tools``).
* ``"loop"``     — needs the live ``AIAgent`` loop to run; not reachable
                   statelessly (e.g. ``delegate_task``). Never an IDE tool.

Scope of THIS module (shared-foundation step)
---------------------------------------------
The IDE projection (``ide_registry_tools`` + ``ide_direct_tools``) reproduces
today's MCP surface byte-for-byte; the MCP server is refactored to derive its
lists from here with **no behavior change**.

The ``orchestrator`` role tags encode the *canonical target* surface (per the
parity overhaul §C.3): the orchestrator should self-serve skills + research +
session search the same way the IDE can. The proxy ``customTools`` set is NOT
changed here — Worker B reconciles the TS side against this manifest. Until
then the live orchestrator surface remains the smaller historical set
(``delegate_task``, ``memory``, ``send_message``, ``cronjob``, ``clarify`` —
see ``ORCHESTRATOR_WIRED_TODAY``); the additional ``orchestrator``-tagged
tools below are the deltas Worker B should wire.

Intentional exclusions are expressed as data, not scattered comments:
``delegate_task`` is orchestrator-only (needs the live loop); terminal / file
/ search / todo are Cursor-native on both sides and therefore absent from this
manifest entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict


@dataclass(frozen=True)
class ToolSpec:
    """One tool's place in the Hermes tool surface."""

    name: str
    roles: frozenset[str]
    dispatch: str  # "registry" | "direct" | "loop"
    # Browser_* tools are suppressed in the IDE when ``HERMES_CURSOR_MODE=1``
    # (cursor-ide-browser covers that lane). True only for those tools.
    cursor_suppressed: bool = False
    note: str = ""


# Valid role / dispatch vocabularies (kept small + explicit).
ROLE_IDE = "ide"
ROLE_ORCHESTRATOR = "orchestrator"

DISPATCH_REGISTRY = "registry"
DISPATCH_DIRECT = "direct"
DISPATCH_LOOP = "loop"


def _spec(
    name: str,
    roles: "tuple[str, ...]",
    dispatch: str,
    *,
    cursor_suppressed: bool = False,
    note: str = "",
) -> ToolSpec:
    return ToolSpec(
        name=name,
        roles=frozenset(roles),
        dispatch=dispatch,
        cursor_suppressed=cursor_suppressed,
        note=note,
    )


# The manifest. Order is significant: the IDE registry/direct projections
# preserve insertion order so the derived lists match today's hand-written
# tuples exactly. Keep IDE registry tools first (in EXPOSED_TOOLS order), then
# IDE direct-dispatch tools, then orchestrator-only tools.
HERMES_TOOL_SURFACE: "OrderedDict[str, ToolSpec]" = OrderedDict(
    (s.name, s)
    for s in (
        # ── IDE registry tools (EXPOSED_TOOLS), in the exact current order ──
        _spec("web_search", (ROLE_IDE, ROLE_ORCHESTRATOR), DISPATCH_REGISTRY,
              note="orchestrator role is a §C.3 target add — not yet wired on TS side"),
        _spec("web_extract", (ROLE_IDE,), DISPATCH_REGISTRY),
        # Hermes CloakBrowser stack — suppressed in Cursor IDE mode.
        _spec("browser_navigate", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_click", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_type", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_press", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_snapshot", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_scroll", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_back", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_get_images", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_console", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("browser_vision", (ROLE_IDE,), DISPATCH_REGISTRY, cursor_suppressed=True),
        _spec("vision_analyze", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("image_generate", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("skill_view", (ROLE_IDE, ROLE_ORCHESTRATOR), DISPATCH_REGISTRY,
              note="orchestrator role is a §C.3 target add — not yet wired on TS side"),
        _spec("skills_list", (ROLE_IDE, ROLE_ORCHESTRATOR), DISPATCH_REGISTRY,
              note="orchestrator role is a §C.3 target add — not yet wired on TS side"),
        _spec("skill_manage", (ROLE_IDE, ROLE_ORCHESTRATOR), DISPATCH_REGISTRY,
              note="orchestrator role is a §C.3 target add — not yet wired on TS side"),
        _spec("text_to_speech", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_complete", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_block", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_comment", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_heartbeat", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_show", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_list", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_create", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_unblock", (ROLE_IDE,), DISPATCH_REGISTRY),
        _spec("kanban_link", (ROLE_IDE,), DISPATCH_REGISTRY),
        # ── IDE direct-dispatch orchestration tools (DIRECT_TOOLS order) ──
        _spec("send_message", (ROLE_IDE, ROLE_ORCHESTRATOR), DISPATCH_DIRECT),
        _spec("memory", (ROLE_IDE, ROLE_ORCHESTRATOR), DISPATCH_DIRECT),
        _spec("session_search", (ROLE_IDE, ROLE_ORCHESTRATOR), DISPATCH_DIRECT,
              note="orchestrator role is a §C.3 target add — not yet wired on TS side"),
        # ── Orchestrator-only tools (not exposed via the IDE MCP) ──
        _spec("delegate_task", (ROLE_ORCHESTRATOR,), DISPATCH_LOOP,
              note="needs the live AIAgent loop to spawn/track subagents; IDE has a native task tool"),
        _spec("cronjob", (ROLE_ORCHESTRATOR,), DISPATCH_REGISTRY,
              note="§C.3 leaves IDE scheduling out for now (document if exposed)"),
        _spec("clarify", (ROLE_ORCHESTRATOR,), DISPATCH_LOOP,
              note="Cursor provides its own clarify UX; orchestrator-only on the engine side"),
    )
)


# The orchestrator tools the proxy customTools bridge wires up TODAY. Worker B
# reconciles the live TS set toward ``orchestrator_tools()`` (the §C.3 target);
# this constant records the historical baseline so the delta is explicit.
ORCHESTRATOR_WIRED_TODAY: "tuple[str, ...]" = (
    "delegate_task",
    "memory",
    "send_message",
    "cronjob",
    "clarify",
)


def tools_for_role(
    role: str,
    *,
    dispatch: "str | None" = None,
    cursor_mode: bool = False,
) -> "tuple[str, ...]":
    """Return the manifest tool names for ``role`` in manifest order.

    ``dispatch`` optionally narrows to a single dispatch kind. ``cursor_mode``
    drops ``cursor_suppressed`` tools (browser_*) — mirrors the IDE MCP server's
    ``HERMES_CURSOR_MODE=1`` suppression.
    """
    names: list[str] = []
    for spec in HERMES_TOOL_SURFACE.values():
        if role not in spec.roles:
            continue
        if dispatch is not None and spec.dispatch != dispatch:
            continue
        if cursor_mode and spec.cursor_suppressed:
            continue
        names.append(spec.name)
    return tuple(names)


def ide_registry_tools(cursor_mode: bool = False) -> "tuple[str, ...]":
    """Registry-dispatched IDE tools — equals the MCP server ``EXPOSED_TOOLS``.

    With ``cursor_mode=True`` the browser_* tools are suppressed, exactly as the
    MCP server does when ``HERMES_CURSOR_MODE=1``.
    """
    return tools_for_role(ROLE_IDE, dispatch=DISPATCH_REGISTRY, cursor_mode=cursor_mode)


def ide_direct_tools() -> "tuple[str, ...]":
    """Direct-dispatch IDE orchestration tools (send_message, memory, session_search)."""
    return tools_for_role(ROLE_IDE, dispatch=DISPATCH_DIRECT)


def ide_tools(cursor_mode: bool = False) -> "tuple[str, ...]":
    """All IDE-exposed tools (registry + direct)."""
    return tools_for_role(ROLE_IDE, cursor_mode=cursor_mode)


def orchestrator_tools() -> "tuple[str, ...]":
    """Canonical (target) orchestrator tool surface — for the proxy customTools.

    NOTE: this is the §C.3 *target*. The currently-wired set is
    ``ORCHESTRATOR_WIRED_TODAY``; Worker B reconciles the TS side toward this.
    """
    return tools_for_role(ROLE_ORCHESTRATOR)
