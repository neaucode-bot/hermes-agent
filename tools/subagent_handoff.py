"""Subagent handoff contract (v1) — parse/strip for delegate_task consumers.

Mirrors cursor-openai-api/src/client-tools/handoff.ts (extract/parse/strip).
No I/O; never raises.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

HANDOFF_SCHEMA_VERSION = "1.0"
HANDOFF_STATUSES = frozenset({"done", "partial", "blocked", "failed"})

_HANDOFF_FENCE_RE = re.compile(
    r"```handoff[ \t]*\r?\n([\s\S]*?)(?:```|\Z)",
)
_HANDOFF_FENCE_STRIP_RE = re.compile(
    r"```handoff[ \t]*\r?\n[\s\S]*?(?:```|\Z)",
)


class HandoffParseResult(TypedDict):
    ok: bool
    report: Dict[str, Any]
    warnings: List[str]
    degraded: bool
    reason: Optional[str]


def extract_handoff_block(final_text: str) -> Optional[str]:
    last: Optional[str] = None
    for match in _HANDOFF_FENCE_RE.finditer(final_text or ""):
        body = match.group(1) or ""
        if body.strip():
            last = body
    return last


def strip_handoff_fence(final_text: str) -> str:
    return _HANDOFF_FENCE_STRIP_RE.sub("", final_text or "").rstrip()


def _major_version(version: str) -> Optional[int]:
    match = re.match(r"^\s*(\d+)", version or "")
    return int(match.group(1)) if match else None


def _synthesize_degraded(final_text: str, reason: str) -> HandoffParseResult:
    trimmed = (final_text or "").strip()
    report = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "status": "partial",
        "summary": trimmed or "Leaf returned no usable output.",
        "artifacts": [],
        "unresolved": [
            {
                "what": "structured handoff missing/invalid",
                "why": "leaf returned prose only",
                "severity": "warn",
            }
        ],
        "_degraded": True,
    }
    return {
        "ok": False,
        "report": report,
        "warnings": [],
        "degraded": True,
        "reason": reason,
    }


def parse_handoff(final_text: str) -> HandoffParseResult:
    block = extract_handoff_block(final_text or "")
    if block is None:
        return _synthesize_degraded(final_text, "no handoff block found in final text")

    try:
        raw = json.loads(block)
    except json.JSONDecodeError:
        return _synthesize_degraded(final_text, "handoff block is not valid JSON")

    if not isinstance(raw, dict):
        return _synthesize_degraded(final_text, "handoff block is not a JSON object")

    version = raw.get("schema_version")
    if not isinstance(version, str) or _major_version(version) != 1:
        return _synthesize_degraded(
            final_text, "schema_version missing or incompatible major version"
        )

    status = raw.get("status")
    if not isinstance(status, str) or status not in HANDOFF_STATUSES:
        return _synthesize_degraded(final_text, "status missing or not a known value")

    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return _synthesize_degraded(final_text, "summary missing or empty")

    warnings: List[str] = []
    artifacts, dropped = _collect_artifacts(raw.get("artifacts"), warnings)
    if status == "done" and dropped:
        status = "partial"
        warnings.append(
            "status downgraded done→partial: one or more artifacts dropped"
        )

    report: Dict[str, Any] = {
        "schema_version": version,
        "status": status,
        "summary": summary,
        "artifacts": artifacts,
    }
    if raw.get("task_id") is None or isinstance(raw.get("task_id"), str):
        report["task_id"] = raw.get("task_id")
    if isinstance(raw.get("confidence"), (int, float)):
        report["confidence"] = raw["confidence"]
    if isinstance(raw.get("truncated"), bool):
        report["truncated"] = raw["truncated"]

    unresolved = _collect_unresolved(raw.get("unresolved"), warnings)
    if unresolved:
        report["unresolved"] = unresolved

    recommended_next = _collect_recommendations(raw.get("recommended_next"), warnings)
    if recommended_next:
        report["recommended_next"] = recommended_next

    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        report["metrics"] = metrics

    degraded = bool(warnings)
    if warnings:
        report["_warnings"] = warnings
        report["_degraded"] = True

    return {
        "ok": True,
        "report": report,
        "warnings": warnings,
        "degraded": degraded,
        "reason": None,
    }


def _collect_artifacts(
    raw: Any, warnings: List[str]
) -> Tuple[List[Dict[str, Any]], bool]:
    if raw is None:
        return [], False
    if not isinstance(raw, list):
        warnings.append("artifacts is not an array; treated as empty")
        return [], True

    artifacts: List[Dict[str, Any]] = []
    seen: set[str] = set()
    dropped = False
    for index, element in enumerate(raw):
        if not isinstance(element, dict):
            warnings.append(f"artifacts[{index}] dropped: missing/invalid required field")
            dropped = True
            continue
        required = ("id", "kind", "handle", "mutated")
        if any(k not in element for k in required):
            warnings.append(f"artifacts[{index}] dropped: missing/invalid required field")
            dropped = True
            continue
        artifact_id = element["id"]
        if artifact_id in seen:
            warnings.append(
                f"artifacts[{index}] dropped: duplicate id \"{artifact_id}\""
            )
            dropped = True
            continue
        kind = element["kind"]
        handle = element["handle"]
        if kind in ("file", "dir") and not (
            isinstance(handle, str)
            and (handle.startswith("/") or handle.startswith("~"))
        ):
            warnings.append(
                f"artifacts[{index}] dropped: {kind} handle \"{handle}\" "
                "is not an absolute path"
            )
            dropped = True
            continue
        seen.add(str(artifact_id))
        artifacts.append(dict(element))
    return artifacts, dropped


def _collect_unresolved(raw: Any, warnings: List[str]) -> Optional[List[Dict[str, Any]]]:
    if raw is None:
        return None
    if not isinstance(raw, list):
        warnings.append("unresolved is not an array; dropped")
        return None
    out: List[Dict[str, Any]] = []
    for index, element in enumerate(raw):
        if (
            not isinstance(element, dict)
            or not all(k in element for k in ("what", "why", "severity"))
        ):
            warnings.append(
                f"unresolved[{index}] dropped: missing/invalid required field"
            )
            continue
        out.append(dict(element))
    return out or None


def _collect_recommendations(
    raw: Any, warnings: List[str]
) -> Optional[List[Dict[str, Any]]]:
    if raw is None:
        return None
    if not isinstance(raw, list):
        warnings.append("recommended_next is not an array; dropped")
        return None
    out: List[Dict[str, Any]] = []
    for index, element in enumerate(raw):
        if not isinstance(element, dict) or "id" not in element or "goal" not in element:
            warnings.append(
                f"recommended_next[{index}] dropped: missing/invalid required field"
            )
            continue
        out.append(dict(element))
    return out or None


def extract_cursor_meta_from_response(response: Any) -> Optional[Dict[str, Any]]:
    """Read the proxy's top-level ``cursor`` object from a chat completion response."""
    if response is None:
        return None

    cursor = getattr(response, "cursor", None)
    if cursor is None and hasattr(response, "model_extra"):
        extra = getattr(response, "model_extra", None) or {}
        if isinstance(extra, dict):
            cursor = extra.get("cursor")

    if cursor is None:
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                if isinstance(dumped, dict):
                    cursor = dumped.get("cursor")
            except Exception:
                pass

    if isinstance(cursor, dict):
        return cursor
    return None


def capture_cursor_meta(agent: Any, response: Any) -> None:
    """Store the latest ``cursor`` meta on the agent for delegate_task consumption."""
    cursor = extract_cursor_meta_from_response(response)
    if cursor is not None:
        agent._last_cursor_meta = cursor


def handoff_report_to_delegate_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Shape exposed to the orchestrator via delegate_task tool results."""
    payload: Dict[str, Any] = {
        "status": report.get("status"),
        "summary": report.get("summary"),
        "artifacts": report.get("artifacts") or [],
    }
    if report.get("recommended_next"):
        payload["recommended_next"] = report["recommended_next"]
    if report.get("unresolved"):
        payload["unresolved"] = report["unresolved"]
    if report.get("_degraded"):
        payload["degraded"] = True
    if report.get("_warnings"):
        payload["warnings"] = report["_warnings"]
    return payload


def consume_delegate_handoff(
    raw_summary: str,
    cursor_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return (clean prose, structured handoff payload for tool result)."""
    parsed = parse_handoff(raw_summary or "")
    report = parsed["report"]

    structured = None
    if cursor_meta and isinstance(cursor_meta.get("handoff"), dict):
        structured = cursor_meta["handoff"]

    handoff_payload = (
        handoff_report_to_delegate_payload(structured)
        if structured
        else handoff_report_to_delegate_payload(report)
    )

    clean = strip_handoff_fence(raw_summary or "").strip()
    return clean, handoff_payload
