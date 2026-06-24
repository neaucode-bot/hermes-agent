"""Cheap-model skill pre-sort — select the relevant skill subset for a task.

Doctrine (cursor↔hermes parity overhaul, §E.3): skills are a tree of
descriptions + memories, **pre-sorted each turn by a CHEAP model** so the
orchestrator's context stays clean. Given the full skills index
(``agent.skills_index.render_skills_index``) + relevant memories + the task
context, a cheap/fast model returns an ordered, capped shortlist of skill names
plus a one-line "why" each.

The headline use case: **the orchestrator filters skills for delegates** so each
delegate gets a small, focused, relevant set instead of the whole index.

This module is reusable and side-effect free apart from the one auxiliary LLM
call it makes:

  - The DELEGATE path (``tools/delegate_tool.py``) calls :func:`select_skills`
    to filter skills per delegate and packs the focused set into the delegate's
    task ``context`` (Tier-2 pointer — the worker Reads the canonical SKILL.md).
  - The same module can later feed an **orchestrator-self tail slot**
    (``agent/conversation_loop.py``). That path is intentionally **not** wired
    here — it is owned separately.

Model tier: cheap/fast. Per ``jarvis-orchestrator-routing`` the right call is the
"simple/mechanical, wide fan-out" tier (e.g. ``composer-2.5`` at ``max``) — sorting
a name+description list against a task is mechanical, fully-specified work, NOT
abstract intent, so the Opus floor does not apply.

Config (``config.yaml``):

  skills.presort.enabled            master gate (default False — conservative)
  skills.presort.delegate_enabled   per-delegate packing gate (default True)
  skills.presort.max_skills         cap on the shortlist (default 5)
  skills.presort.model              cheap-model override (default: aux task cfg)
  skills.presort.timeout            per-call timeout seconds (default 20)
  auxiliary.skill_presort.*         provider/model/base_url for the side-LLM call
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AUX_TASK = "skill_presort"
_DEFAULT_MAX_SKILLS = 5
_DEFAULT_TIMEOUT = 20.0
_DEFAULT_MAX_TOKENS = 600

# Matches a rendered ``<available_skills>`` skill line: "    - <name>: desc" or
# "    - <name>". Skill names are lowercased slugs (see the validator's
# ^[a-z0-9][a-z0-9._-]*$ rule).
_SKILL_LINE_RE = re.compile(r"^\s+-\s+([a-z0-9][a-z0-9._-]*)\b")
# Matches a "[names only]" demoted category line: "  cat [names only]: a, b, c".
_NAMES_ONLY_RE = re.compile(r"^\s+\S.*\[names only\]:\s*(.+)$")


# ── config ───────────────────────────────────────────────────────────────────


def _load_full_config() -> Dict[str, Any]:
    """Best-effort load of the merged config (CLI runtime first, then disk)."""
    try:
        from cli import CLI_CONFIG  # type: ignore

        if isinstance(CLI_CONFIG, dict) and CLI_CONFIG:
            return CLI_CONFIG
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config

        return load_config() or {}
    except Exception:
        return {}


def _presort_config() -> Dict[str, Any]:
    cfg = _load_full_config()
    skills_cfg = cfg.get("skills") if isinstance(cfg, dict) else None
    if not isinstance(skills_cfg, dict):
        return {}
    presort = skills_cfg.get("presort")
    return presort if isinstance(presort, dict) else {}


def presort_enabled() -> bool:
    """Master gate — whether the cheap-model skill pre-sort runs at all."""
    return bool(_presort_config().get("enabled", False))


def delegate_presort_enabled() -> bool:
    """Whether the delegate path packs an auto-selected skill set per worker.

    Requires the master gate. ``delegate_enabled`` defaults to True so that
    enabling presort at all turns on the headline (orchestrator-filters-for-
    delegates) use case without a second flag.
    """
    if not presort_enabled():
        return False
    return bool(_presort_config().get("delegate_enabled", True))


def _max_skills() -> int:
    try:
        n = int(_presort_config().get("max_skills", _DEFAULT_MAX_SKILLS))
        return n if n > 0 else _DEFAULT_MAX_SKILLS
    except Exception:
        return _DEFAULT_MAX_SKILLS


def _timeout() -> float:
    try:
        return float(_presort_config().get("timeout", _DEFAULT_TIMEOUT))
    except Exception:
        return _DEFAULT_TIMEOUT


def _model_override() -> Optional[str]:
    m = str(_presort_config().get("model", "") or "").strip()
    return m or None


# ── result type ──────────────────────────────────────────────────────────────


@dataclass
class SkillPick:
    name: str
    why: str = ""


@dataclass
class SkillPresortResult:
    """Outcome of one pre-sort call.

    ``picks`` is the ordered, capped, name-validated shortlist. ``paths`` maps
    each picked name to its canonical ``SKILL.md`` so a native worker (which has
    no ``skill_view``) can Read the body. ``error`` is set on a graceful failure
    (the caller should simply skip packing — never raise).
    """

    picks: List[SkillPick] = field(default_factory=list)
    paths: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    raw: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.picks)

    def as_context_block(self, *, with_skill_view: bool = False) -> str:
        """Render a Tier-2 "possibly relevant Hermes skills" pointer block.

        ``with_skill_view=False`` (delegate default): point at the canonical
        ``SKILL.md`` path so a native worker can Read it. ``with_skill_view=True``:
        for a surface that has the ``hermes-tools`` MCP (e.g. the orchestrator
        self path) — load via ``skill_view(name)``.
        """
        if not self.picks:
            return ""
        if with_skill_view:
            header = (
                "Possibly relevant Hermes skills (auto-selected — load with "
                "skill_view(name) before relying on them):"
            )
        else:
            header = (
                "Possibly relevant Hermes skills (auto-selected — Read the "
                "SKILL.md before relying on it):"
            )
        lines = [header]
        for p in self.picks:
            why = f" — {p.why}" if p.why else ""
            if with_skill_view:
                lines.append(f"- {p.name}{why}")
            else:
                path = self.paths.get(p.name)
                loc = f": {path}" if path else ""
                lines.append(f"- {p.name}{loc}{why}")
        return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────────


def _index_skill_names(index_text: str) -> set[str]:
    """Extract the set of valid skill names from a rendered skills index.

    Used to reject hallucinated names — only skills actually shown to the model
    are eligible.
    """
    names: set[str] = set()
    for line in index_text.splitlines():
        m = _SKILL_LINE_RE.match(line)
        if m:
            names.add(m.group(1))
            continue
        m = _NAMES_ONLY_RE.match(line)
        if m:
            for part in m.group(1).split(","):
                slug = part.strip()
                if slug and re.fullmatch(r"[a-z0-9][a-z0-9._-]*", slug):
                    names.add(slug)
    return names


def _resolve_skill_paths(names: set[str]) -> Dict[str, str]:
    """Map skill names to their canonical SKILL.md path (local + external dirs)."""
    mapping: Dict[str, str] = {}
    if not names:
        return mapping
    try:
        from agent.skill_utils import (
            get_all_skills_dirs,
            iter_skill_index_files,
            parse_frontmatter,
        )
    except Exception:
        return mapping
    remaining = set(names)
    for skills_dir in get_all_skills_dirs():
        try:
            if not skills_dir.exists():
                continue
        except Exception:
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                fm, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
            except Exception:
                fm = {}
            name = None
            if isinstance(fm, dict):
                name = str(fm.get("name") or "").strip() or None
            name = name or skill_file.parent.name
            if name in remaining:
                mapping.setdefault(name, str(skill_file))
                remaining.discard(name)
        if not remaining:
            break
    return mapping


def _extract_json_obj(text: str) -> Optional[Any]:
    """Pull the first JSON object/array out of a model response."""
    if not text:
        return None
    text = text.strip()
    # Strip code fences if present.
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Fast path: whole string is JSON.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find the first balanced {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None


def _parse_picks(payload: Any, valid_names: set[str], max_skills: int) -> List[SkillPick]:
    """Coerce model JSON into a validated, capped, ordered list of picks."""
    if isinstance(payload, dict):
        items = payload.get("skills")
        if items is None:
            # Tolerate {"<name>": "<why>", ...} shape too.
            items = [{"name": k, "why": v} for k, v in payload.items()]
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    picks: List[SkillPick] = []
    seen: set[str] = set()
    for item in items or []:
        name = ""
        why = ""
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("skill") or "").strip()
            why = str(item.get("why") or item.get("reason") or "").strip()
        if not name or name in seen:
            continue
        if name not in valid_names:
            continue
        seen.add(name)
        picks.append(SkillPick(name=name, why=why))
        if len(picks) >= max_skills:
            break
    return picks


def _build_messages(
    task_text: str,
    index_text: str,
    *,
    context: Optional[str],
    memories: Optional[str],
    max_skills: int,
) -> List[Dict[str, str]]:
    system = (
        "You are a fast skill selector for an AI orchestrator. Given a task and a "
        "catalog of available skills (name + description), pick ONLY the skills "
        "genuinely relevant to completing the task — the ones whose conventions, "
        "commands, or workflows the worker would actually need. "
        f"Return AT MOST {max_skills} skills, most relevant first. If none are "
        "genuinely relevant, return an empty list. Be strict: do not include "
        "skills that are only tangentially related. "
        "Output STRICT JSON only, no prose, in this exact shape:\n"
        '{"skills": [{"name": "<exact-skill-name-from-catalog>", "why": "<= 12 words"}]}\n'
        "Use exact skill names copied from the catalog. Never invent a skill name."
    )
    user_parts = [f"TASK:\n{task_text.strip()}"]
    if context and context.strip():
        user_parts.append(f"\nCONTEXT:\n{context.strip()}")
    if memories and memories.strip():
        user_parts.append(f"\nRELEVANT MEMORIES:\n{memories.strip()}")
    user_parts.append(f"\nAVAILABLE SKILLS:\n{index_text.strip()}")
    user_parts.append("\nReturn the JSON now.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "".join(user_parts)},
    ]


# ── public API ───────────────────────────────────────────────────────────────


def select_skills(
    task_text: str,
    *,
    context: Optional[str] = None,
    memories: Optional[str] = None,
    max_skills: Optional[int] = None,
    available_tools: "set[str] | None" = None,
    available_toolsets: "set[str] | None" = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> SkillPresortResult:
    """Select the relevant skill subset for a task using a cheap model.

    Inputs (per §E.3): the full rendered skills index (names + descriptions),
    optional relevant memories, and the task text. Output: an ordered, capped,
    name-validated shortlist with a one-line "why" each, plus canonical SKILL.md
    paths for the picks.

    Never raises — on any failure returns an empty result with ``.error`` set so
    the caller can simply skip packing.
    """
    if not task_text or not task_text.strip():
        return SkillPresortResult(error="empty task text")

    cap = max_skills if (max_skills and max_skills > 0) else _max_skills()

    # 1. Render the full index (names + descriptions only — cheap input).
    try:
        from agent.skills_index import render_skills_index

        index_text = render_skills_index(
            available_tools=available_tools,
            available_toolsets=available_toolsets,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("skill_presort: could not render skills index: %s", exc)
        return SkillPresortResult(error=f"index render failed: {exc}")

    if not index_text or not index_text.strip():
        return SkillPresortResult(error="empty skills index")

    valid_names = _index_skill_names(index_text)
    if not valid_names:
        return SkillPresortResult(error="no skill names in index")

    # 2. Cheap-model call.
    messages = _build_messages(
        task_text,
        index_text,
        context=context,
        memories=memories,
        max_skills=cap,
    )
    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task=_AUX_TASK,
            model=model or _model_override(),
            messages=messages,
            temperature=0,
            max_tokens=_DEFAULT_MAX_TOKENS,
            timeout=timeout if timeout is not None else _timeout(),
            main_runtime=main_runtime,
        )
    except Exception as exc:
        logger.info("skill_presort: cheap-model call failed: %s", exc)
        return SkillPresortResult(error=f"llm call failed: {exc}")

    try:
        content = response.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - defensive
        return SkillPresortResult(error=f"bad llm response: {exc}")

    # 3. Parse + validate.
    payload = _extract_json_obj(content)
    if payload is None:
        logger.debug("skill_presort: no JSON in response: %r", content[:200])
        return SkillPresortResult(error="no JSON in response", raw=content)

    picks = _parse_picks(payload, valid_names, cap)
    if not picks:
        return SkillPresortResult(raw=content)

    paths = _resolve_skill_paths({p.name for p in picks})
    return SkillPresortResult(picks=picks, paths=paths, raw=content)
