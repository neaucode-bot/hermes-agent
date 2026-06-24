#!/usr/bin/env python3
"""Tests for the cheap-model skill pre-sort module (agent/skill_presort.py).

Covers the pure parsing/validation helpers, the result rendering, and a fully
mocked select_skills() path (no network). Run with:

    scripts/run_tests.sh tests/agent/test_skill_presort.py -q
"""

from unittest.mock import MagicMock, patch

from agent import skill_presort
from agent.skill_presort import (
    SkillPick,
    SkillPresortResult,
    _extract_json_obj,
    _index_skill_names,
    _parse_picks,
    select_skills,
)

SAMPLE_INDEX = """## Skills (mandatory)
Before replying, scan the skills below.

<available_skills>
  autonomous-ai-agents:
    - jarvis-orchestrator-routing: Orchestrator routing policy.
    - agent-swarm-orchestration: Worker swarm patterns.
  finance [names only]: stocks, sndk-premarket
  general:
    - jarvis-diary: Maintain the build diary.
</available_skills>

Only proceed without loading a skill if genuinely none are relevant.
"""


def test_index_skill_names_extracts_described_and_names_only():
    names = _index_skill_names(SAMPLE_INDEX)
    assert "jarvis-orchestrator-routing" in names
    assert "agent-swarm-orchestration" in names
    assert "jarvis-diary" in names
    # names-only category line members are still eligible
    assert "stocks" in names
    assert "sndk-premarket" in names


def test_extract_json_obj_handles_fenced_and_bare():
    assert _extract_json_obj('{"skills": []}') == {"skills": []}
    fenced = "```json\n{\"skills\": [{\"name\": \"x\"}]}\n```"
    assert _extract_json_obj(fenced) == {"skills": [{"name": "x"}]}
    noisy = 'Here you go: {"skills": [{"name": "x", "why": "y"}]} done'
    assert _extract_json_obj(noisy) == {"skills": [{"name": "x", "why": "y"}]}
    assert _extract_json_obj("not json at all") is None


def test_parse_picks_validates_names_and_caps():
    valid = {"a", "b", "c", "d"}
    payload = {
        "skills": [
            {"name": "a", "why": "first"},
            {"name": "hallucinated", "why": "nope"},  # dropped: not in valid set
            {"name": "b"},
            {"name": "a"},  # dropped: duplicate
            {"name": "c"},
            {"name": "d"},
        ]
    }
    picks = _parse_picks(payload, valid, max_skills=3)
    assert [p.name for p in picks] == ["a", "b", "c"]
    assert picks[0].why == "first"


def test_parse_picks_accepts_list_and_string_shapes():
    valid = {"a", "b"}
    assert [p.name for p in _parse_picks(["a", "b"], valid, 5)] == ["a", "b"]
    assert _parse_picks("garbage", valid, 5) == []


def test_result_context_block_path_vs_skill_view():
    res = SkillPresortResult(
        picks=[SkillPick("jarvis-diary", "diary format")],
        paths={"jarvis-diary": "/x/jarvis-diary/SKILL.md"},
    )
    # delegate default: include the SKILL.md path for a worker to Read
    block = res.as_context_block(with_skill_view=False)
    assert "/x/jarvis-diary/SKILL.md" in block
    assert "jarvis-diary" in block
    assert "diary format" in block
    # skill_view variant: no path, mentions skill_view
    sv = res.as_context_block(with_skill_view=True)
    assert "skill_view" in sv
    assert "/x/jarvis-diary/SKILL.md" not in sv


def test_empty_result_is_falsy_and_blocks_empty():
    assert not SkillPresortResult()
    assert SkillPresortResult().as_context_block() == ""


def test_select_skills_empty_task():
    res = select_skills("   ")
    assert not res
    assert res.error


def test_select_skills_happy_path_mocked():
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = (
        '{"skills": [{"name": "jarvis-diary", "why": "writing the diary"}]}'
    )

    with patch("agent.skills_index.render_skills_index", return_value=SAMPLE_INDEX), \
         patch("agent.auxiliary_client.call_llm", return_value=fake_resp) as mock_llm, \
         patch.object(skill_presort, "_resolve_skill_paths",
                      return_value={"jarvis-diary": "/x/jarvis-diary/SKILL.md"}):
        res = select_skills("Help me write a diary entry", context="some context")

    assert mock_llm.called
    assert res
    assert [p.name for p in res.picks] == ["jarvis-diary"]
    block = res.as_context_block()
    assert "jarvis-diary" in block and "/x/jarvis-diary/SKILL.md" in block


def test_select_skills_rejects_hallucinated_names():
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = '{"skills": [{"name": "totally-made-up"}]}'

    with patch("agent.skills_index.render_skills_index", return_value=SAMPLE_INDEX), \
         patch("agent.auxiliary_client.call_llm", return_value=fake_resp):
        res = select_skills("do something")

    assert not res  # hallucinated name filtered out -> empty


def test_select_skills_swallows_llm_failure():
    with patch("agent.skills_index.render_skills_index", return_value=SAMPLE_INDEX), \
         patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")):
        res = select_skills("do something")
    assert not res
    assert "boom" in (res.error or "")


# ── delegate-path packing ────────────────────────────────────────────────────


def test_delegate_packing_disabled_is_noop():
    from tools.delegate_tool import _maybe_pack_presort_skills

    tasks = [{"goal": "do work", "context": "orig"}]
    with patch("agent.skill_presort.delegate_presort_enabled", return_value=False):
        _maybe_pack_presort_skills(tasks)
    assert tasks[0]["context"] == "orig"


def test_delegate_packing_appends_block_when_enabled():
    from tools.delegate_tool import _maybe_pack_presort_skills

    result = SkillPresortResult(
        picks=[SkillPick("jarvis-diary", "diary format")],
        paths={"jarvis-diary": "/x/jarvis-diary/SKILL.md"},
    )
    tasks = [{"goal": "log a diary entry", "context": "orig context"}]
    with patch("agent.skill_presort.delegate_presort_enabled", return_value=True), \
         patch("agent.skill_presort.select_skills", return_value=result):
        _maybe_pack_presort_skills(tasks)
    assert tasks[0]["context"].startswith("orig context")
    assert "Possibly relevant Hermes skills (auto-selected" in tasks[0]["context"]
    assert "/x/jarvis-diary/SKILL.md" in tasks[0]["context"]


def test_delegate_packing_is_idempotent():
    from tools.delegate_tool import _maybe_pack_presort_skills

    result = SkillPresortResult(picks=[SkillPick("jarvis-diary", "x")])
    pre_packed = "goal stuff\n\nPossibly relevant Hermes skills (auto-selected — Read..."
    tasks = [{"goal": "g", "context": pre_packed}]
    with patch("agent.skill_presort.delegate_presort_enabled", return_value=True), \
         patch("agent.skill_presort.select_skills", return_value=result) as sel:
        _maybe_pack_presort_skills(tasks)
    assert not sel.called  # already packed -> skipped
    assert tasks[0]["context"] == pre_packed
