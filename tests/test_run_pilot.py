"""End-to-end test of the pilot orchestrator with MockJudgeClient.

Builds a tiny 2-conversation, 2-system, 3-tier compression set, registers
mock judge responses for faithfulness Stage 1 + Stage 2, and verifies the
orchestrator produces bootstrap CIs, paired McNemar/Wilcoxon vs ours, and
tier curve stats — all with zero API calls.
"""
from __future__ import annotations

import json

import pytest

from compressor.eval.conversations import Conversation, Turn, write_jsonl
from compressor.eval.faithfulness import (
    CoverageDecision,
    CoverageReportV1,
    CriticalItem,
    ExtractedItemsV1,
)
from compressor.eval.judge_cache import JudgeCache
from compressor.eval.llm_client import MockJudgeClient
from compressor.eval.prompts import load_prompt
from compressor.eval.run_pilot import (
    CompressionRow,
    aggregate_faithfulness,
    aggregate_tier,
    run_faithfulness,
)


def _conv(cid: str) -> Conversation:
    return Conversation(
        id=cid,
        scenario_type="test",
        turns=(
            Turn("user", f"{cid} u1 asking about redis and port 5432"),
            Turn("assistant", f"{cid} a1 use redis on port 5432"),
            Turn("user", f"{cid} u2 follow up"),
            Turn("assistant", f"{cid} a2 final answer"),
        ),
    )


def _register_faithfulness(
    judge: MockJudgeClient,
    conv: Conversation,
    compressions: dict[tuple[str, str], str],
    coverage: dict[tuple[str, str], list[str]],
) -> None:
    """Register Stage 1 (per conv) + Stage 2 (per compression) mock responses.

    `compressions` maps (system, tier) -> compressed text.
    `coverage` maps (system, tier) -> list of 'present'/'partial'/'false'
    for the two items.
    """
    # Stage 1: two items
    items = [
        CriticalItem(id=1, type="entity", summary="redis", verbatim_indicator="redis"),
        CriticalItem(id=2, type="number", summary="5432", verbatim_indicator="5432"),
    ]
    p1 = load_prompt("faithfulness_stage1_v1")
    _, u1 = p1.render(source=conv.flatten())
    judge.register_response("faithfulness_stage1_v1", u1, ExtractedItemsV1(items=items))

    # Stage 2: per compression
    p2 = load_prompt("faithfulness_stage2_v1")
    items_json = json.dumps([it.model_dump() for it in items], ensure_ascii=False)
    for (system, tier), comp in compressions.items():
        _, u2 = p2.render(items_json=items_json, compression=comp)
        calls = coverage[(system, tier)]
        decisions = []
        for i, call in enumerate(calls, start=1):
            # provide valid evidence for present/partial
            ev = "redis" if i == 1 else "5432"
            decisions.append(
                CoverageDecision(id=i, present=call, evidence=ev if call != "false" else "")
            )
        judge.register_response(
            "faithfulness_stage2_v1", u2, CoverageReportV1(decisions=decisions)
        )


def test_pilot_faithfulness_end_to_end(tmp_path):
    convs = [_conv("c1"), _conv("c2")]
    conversations = {c.id: c for c in convs}

    # Two systems x 3 tiers x 2 convs.
    # "tfix375" (ours) preserves both items at recent, degrades by tier.
    # "llmlingua2" preserves fewer, flat across tiers (tier-blind).
    tiers = ["recent", "mid", "old"]

    # Build compressions: text must contain the evidence tokens where 'present'
    def comp_text(present_flags):
        parts = []
        if present_flags[0]:
            parts.append("redis is the cache")
        if present_flags[1]:
            parts.append("port 5432")
        return ". ".join(parts) or "nothing"

    # ours: recent both, mid redis only, old neither -> steep curve
    ours_cov = {"recent": ["present", "present"], "mid": ["present", "false"], "old": ["false", "false"]}
    # llmlingua: redis only at every tier -> flat curve
    ling_cov = {"recent": ["present", "false"], "mid": ["present", "false"], "old": ["present", "false"]}

    comp_rows = []
    judge_a = MockJudgeClient(name="gpt-primary")
    judge_b = MockJudgeClient(name="claude-secondary")

    for conv in convs:
        compressions = {}
        coverage = {}
        for tier in tiers:
            oc = ours_cov[tier]
            lc = ling_cov[tier]
            ours_text = comp_text([oc[0] == "present", oc[1] == "present"])
            ling_text = comp_text([lc[0] == "present", lc[1] == "present"])
            # make the two systems' texts distinct so hashes differ
            ours_text = f"[ours] {ours_text}"
            ling_text = f"[ling] {ling_text}"
            compressions[("tfix375", tier)] = ours_text
            compressions[("llmlingua2", tier)] = ling_text
            coverage[("tfix375", tier)] = oc
            coverage[("llmlingua2", tier)] = lc
            comp_rows.append(CompressionRow(conv.id, "tfix375", tier, 3, ours_text, 100, 40))
            comp_rows.append(CompressionRow(conv.id, "llmlingua2", tier, 3, ling_text, 100, 40))
        for judge in (judge_a, judge_b):
            _register_faithfulness(judge, conv, compressions, coverage)

    cache = JudgeCache(tmp_path / "cache.jsonl")
    scores = run_faithfulness(comp_rows, conversations, [judge_a, judge_b], cache)

    # Sanity: ours at recent should score 1.0 (both items present)
    assert scores[("c1", "tfix375", "recent")]["gpt-primary"] == pytest.approx(1.0)
    # ours at old should score 0.0
    assert scores[("c1", "tfix375", "old")]["gpt-primary"] == pytest.approx(0.0)
    # llmlingua flat at 0.5 (1 of 2 items) every tier
    assert scores[("c1", "llmlingua2", "recent")]["gpt-primary"] == pytest.approx(0.5)
    assert scores[("c1", "llmlingua2", "old")]["gpt-primary"] == pytest.approx(0.5)

    # Faithfulness aggregate: CIs + paired tests
    faith = aggregate_faithfulness(scores, our_system="tfix375", n_boot=500, seed=1)
    assert "tfix375|recent" in faith["per_system_tier"]
    assert "llmlingua2" in faith["paired_vs_ours"]
    # ours wins at recent (1.0 vs 0.5) and mid (0.5 vs 0.5 tie) loses at old (0 vs 0.5)
    wl = faith["paired_vs_ours"]["llmlingua2"]["win_loss"]
    # per (conv,tier): recent ours>ling (win x2), mid tie x2, old ours<ling (loss x2)
    assert wl["a_wins"] == 2
    assert wl["b_wins"] == 2
    assert wl["ties"] == 2

    # Tier curve: ours should be monotonic decreasing, llmlingua flat
    tier_agg = aggregate_tier(scores, our_system="tfix375")
    assert tier_agg["tfix375"]["monotonicity_rate"] == pytest.approx(1.0)
    assert tier_agg["tfix375"]["mean_delta_recent_old"] == pytest.approx(1.0)
    # llmlingua flat -> delta 0, still monotonic (flat within eps)
    assert tier_agg["llmlingua2"]["mean_delta_recent_old"] == pytest.approx(0.0)


def test_pilot_cache_resumes(tmp_path):
    """Second run with the same cache fires zero new judge calls."""
    conv = _conv("c1")
    conversations = {"c1": conv}
    tiers = ["recent", "mid", "old"]
    comp_rows = []
    judge = MockJudgeClient(name="gpt-primary")
    compressions, coverage = {}, {}
    for tier in tiers:
        text = f"[ours] redis is the cache. port 5432 ({tier})"
        compressions[("tfix375", tier)] = text
        coverage[("tfix375", tier)] = ["present", "present"]
        comp_rows.append(CompressionRow("c1", "tfix375", tier, 3, text, 100, 40))
    _register_faithfulness(judge, conv, compressions, coverage)

    cache_path = tmp_path / "cache.jsonl"
    cache1 = JudgeCache(cache_path)
    run_faithfulness(comp_rows, conversations, [judge], cache1)
    calls_after_first = len(judge.calls)

    # New judge instance (no registered responses) + reloaded cache: should
    # hit cache for everything and make ZERO new calls.
    judge2 = MockJudgeClient(name="gpt-primary")
    cache2 = JudgeCache(cache_path)
    run_faithfulness(comp_rows, conversations, [judge2], cache2)
    assert len(judge2.calls) == 0  # fully served from cache
    assert calls_after_first > 0
