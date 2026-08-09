"""Pilot / full bake-off orchestrator: drive judging + stats over compression outputs.

Inputs
------
  * a "compressions" JSONL: one row per (conversation, system, tier) produced
    by the compression runners (infer.py / run_downstream_bakeoff.py / the API
    baselines). Each row has at least: conversation_id, source (system name),
    turn_age, target_ratio, compressed, input_chars, output_chars, and — for
    downstream rows — a `downstream` block with last_user_turn + held_assistant_turn.
  * the source conversations (structured-turn JSONL) for M_faithfulness Stage 1.

What it does
------------
  1. M_faithfulness: for each conversation, extract critical items ONCE per
     (source_conversation, judge) — shared across all systems/tiers of that
     conversation — then check coverage per compression. Ensemble over judges.
  2. M_downstream: generate a continuation per compression (fixed generator),
     score with the judge ensemble.
  3. M_tier_appropriate: derive curve stats from the per-tier faithfulness scores.
  4. Sanity metrics: deterministic, per compression.
  5. Stats: clustered bootstrap CIs, McNemar + Wilcoxon vs each competitor,
     inter-judge agreement (kappa on coverage labels, ICC on scores).

Compute placement
-----------------
GPU baselines run on the pod; this orchestrator runs LOCALLY (judges are just
API calls). Point it at the pulled-down compression JSONLs.

Caching
-------
Every judge call result is cached to a JSONL keyed by the eval.llm_client
cache_key (prompt+schema+snapshot+rubric+seed+inputs). Re-runs skip completed
work, so an interrupted pilot resumes cheaply.

Cost safety
-----------
`--dry-run` prints the planned call counts + a rough token estimate and fires
NO API calls. Always dry-run first.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .conversations import Conversation, load_jsonl as load_conversations
from . import downstream as m_downstream
from . import faithfulness as m_faith
from . import sanity_metrics
from . import stats as m_stats
from . import tier_metrics
from .ensemble import ensemble_scalar
from .judge_cache import JudgeCache
from .llm_client import JudgeClient

logger = logging.getLogger("run_pilot")


# ---------------------------------------------------------------------------
# Compression-row loading
# ---------------------------------------------------------------------------


@dataclass
class CompressionRow:
    conversation_id: str
    source: str  # system name
    turn_age: str
    target_ratio: int
    compressed: str
    input_chars: int
    output_chars: int
    # downstream-only
    last_user_turn: str | None = None
    held_assistant_turn: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CompressionRow":
        ds = d.get("downstream") or {}
        return cls(
            conversation_id=d["conversation_id"],
            source=d["source"],
            turn_age=d["turn_age"],
            target_ratio=int(d["target_ratio"]),
            compressed=d["compressed"],
            input_chars=int(d.get("input_chars", len(d["compressed"]))),
            output_chars=int(d.get("output_chars", len(d["compressed"]))),
            last_user_turn=ds.get("last_user_turn"),
            held_assistant_turn=ds.get("held_assistant_turn"),
            raw=d,
        )


def load_compression_rows(path: Path) -> list[CompressionRow]:
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        if line.strip():
            rows.append(CompressionRow.from_dict(json.loads(line)))
    return rows


# ---------------------------------------------------------------------------
# Faithfulness over the whole compression set (ensemble, shared Stage 1)
# ---------------------------------------------------------------------------


def run_faithfulness(
    comp_rows: Sequence[CompressionRow],
    conversations: dict[str, Conversation],
    judges: Sequence[JudgeClient],
    cache: JudgeCache,
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Returns {(conversation_id, system, tier): {judge_name: score}}.

    Stage 1 (item extraction) is computed once per (conversation, judge) and
    reused across every system/tier compression of that conversation.
    """
    # 1. Stage 1 per (conversation, judge)
    stage1: dict[tuple[str, str], Any] = {}  # (conv_id, judge) -> (items, result)
    conv_ids = sorted({r.conversation_id for r in comp_rows})
    for cid in conv_ids:
        conv = conversations[cid]
        source_text = conv.flatten()
        for judge in judges:
            key = (cid, judge.name)
            cached = cache.get_faithfulness_stage1(cid, judge.name)
            if cached is not None:
                stage1[key] = cached
                continue
            items, result = m_faith.extract_critical_items(source_text, judge)
            cache.put_faithfulness_stage1(cid, judge.name, items, result)
            stage1[key] = (items, result)

    # 2. Stage 2 per (compression row, judge)
    scores: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in comp_rows:
        source_text = conversations[row.conversation_id].flatten()
        for judge in judges:
            items, s1 = stage1[(row.conversation_id, judge.name)]
            cached_score = cache.get_faithfulness_score(
                row.conversation_id, row.source, row.turn_age, judge.name
            )
            if cached_score is not None:
                scores[(row.conversation_id, row.source, row.turn_age)][
                    judge.name
                ] = cached_score
                continue
            ev = m_faith.evaluate(
                source_text,
                row.compressed,
                judge,
                items=items,
                stage1_result=s1,
            )
            score = ev.score.score
            cache.put_faithfulness_score(
                row.conversation_id, row.source, row.turn_age, judge.name, ev
            )
            scores[(row.conversation_id, row.source, row.turn_age)][judge.name] = score
    return scores


# ---------------------------------------------------------------------------
# Aggregation into per-(system, tier) tables with CIs + paired tests
# ---------------------------------------------------------------------------


def aggregate_faithfulness(
    scores: dict[tuple[str, str, str], dict[str, float]],
    *,
    our_system: str,
    seed: int = 42,
    n_boot: int = 10000,
) -> dict:
    """Ensemble-mean per row, then per-(system, tier) bootstrap CIs and
    paired McNemar + Wilcoxon of `our_system` vs every other system."""
    # Ensemble mean per (conv, system, tier)
    ens: dict[tuple[str, str, str], float] = {}
    for key, per_judge in scores.items():
        ens[key] = ensemble_scalar(per_judge).mean

    systems = sorted({k[1] for k in ens})
    tiers = sorted({k[2] for k in ens})

    out: dict[str, Any] = {"per_system_tier": {}, "paired_vs_ours": {}}

    # Per-(system, tier) CI
    for system in systems:
        for tier in tiers:
            vals, clusters = [], []
            for (cid, s, t), v in ens.items():
                if s == system and t == tier:
                    vals.append(v)
                    clusters.append(cid)
            if not vals:
                continue
            ci = m_stats.clustered_bootstrap_ci(
                vals, clusters, n_boot=n_boot, seed=seed
            )
            out["per_system_tier"][f"{system}|{tier}"] = ci.to_dict()

    # Paired ours vs each competitor (pooled across tiers, keyed by conv+tier)
    for competitor in systems:
        if competitor == our_system:
            continue
        ours_scores, comp_scores = {}, {}
        for (cid, s, t), v in ens.items():
            pair_key = f"{cid}|{t}"
            if s == our_system:
                ours_scores[pair_key] = v
            elif s == competitor:
                comp_scores[pair_key] = v
        if not (ours_scores and comp_scores):
            continue
        wl = m_stats.paired_win_loss(ours_scores, comp_scores)
        mcn = m_stats.mcnemar_test(wl)
        wil = m_stats.wilcoxon_paired(ours_scores, comp_scores)
        out["paired_vs_ours"][competitor] = {
            "win_loss": {"a_wins": wl.a_wins, "b_wins": wl.b_wins, "ties": wl.ties},
            "mcnemar": mcn.to_dict(),
            "wilcoxon": wil.to_dict(),
        }
    return out


def aggregate_tier(
    scores: dict[tuple[str, str, str], dict[str, float]],
    *,
    our_system: str,
) -> dict:
    """Curve stats per (conversation, system) from ensemble-mean faithfulness."""
    ens: dict[tuple[str, str, str], float] = {
        k: ensemble_scalar(v).mean for k, v in scores.items()
    }
    # Group by (system, conv) -> {tier: score}
    by_sys_conv: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (cid, system, tier), v in ens.items():
        by_sys_conv[(system, cid)][tier] = v

    out: dict[str, Any] = {}
    per_system_curves: dict[str, list] = defaultdict(list)
    for (system, cid), tier_scores in by_sys_conv.items():
        if len(tier_scores) < 3:
            continue  # need all three tiers for a curve
        try:
            curve = tier_metrics.compute_curve(tier_scores)
        except KeyError:
            continue
        per_system_curves[system].append(curve)

    for system, curves in per_system_curves.items():
        agg = tier_metrics.aggregate_curves(curves)
        out[system] = agg.to_dict()
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_judges(spec: str) -> list[JudgeClient]:
    """spec is a comma-separated list of judge specs 'backend:model:name'.
    e.g. 'openai:gpt-5.5:gpt-primary,anthropic:claude-sonnet-4-6:claude-secondary'"""
    from .llm_client import AnthropicJudgeClient, MockJudgeClient, OpenAIJudgeClient

    judges: list[JudgeClient] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        backend, model, name = part.split(":")
        if backend == "openai":
            judges.append(OpenAIJudgeClient(name=name, model=model, snapshot_id=model))
        elif backend == "anthropic":
            judges.append(
                AnthropicJudgeClient(name=name, model=model, snapshot_id=model)
            )
        elif backend == "mock":
            judges.append(MockJudgeClient(name=name, model=model, snapshot_id=model))
        else:
            raise ValueError(f"unknown judge backend: {backend}")
    return judges


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--compressions", type=Path, required=True,
                   help="JSONL of compression outputs (any set of systems).")
    p.add_argument("--conversations", type=Path, required=True,
                   help="Structured-turn source conversations JSONL.")
    p.add_argument("--our-system", default="tfix375",
                   help="System name treated as 'ours' in paired tests.")
    p.add_argument("--judges",
                   default="openai:gpt-5.5:gpt-primary,anthropic:claude-sonnet-4-6:claude-secondary",
                   help="Comma-separated backend:model:name judge specs.")
    p.add_argument("--cache", type=Path, default=Path("data/pilot_judge_cache.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/pilot_results.json"))
    p.add_argument("--metrics", default="faithfulness,tier",
                   help="Comma list: faithfulness, downstream, tier, sanity.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    comp_rows = load_compression_rows(args.compressions)
    conv_list = load_conversations(args.conversations)
    conversations = {c.id: c for c in conv_list}
    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}

    systems = sorted({r.source for r in comp_rows})
    tiers = sorted({r.turn_age for r in comp_rows})
    logger.info(
        "Loaded %d compression rows: %d systems %s, %d tiers %s, %d conversations",
        len(comp_rows), len(systems), systems, len(tiers), tiers,
        len(conversations),
    )

    if args.dry_run:
        judges = args.judges.split(",")
        n_conv = len({r.conversation_id for r in comp_rows})
        n_stage1 = n_conv * len(judges)
        n_stage2 = len(comp_rows) * len(judges)
        n_down = len([r for r in comp_rows if r.last_user_turn]) * (len(judges) + 1)
        logger.info("[dry-run] planned judge calls:")
        logger.info("  faithfulness Stage 1 (per conv x judge): %d", n_stage1)
        logger.info("  faithfulness Stage 2 (per row x judge):  %d", n_stage2)
        if "downstream" in metrics:
            logger.info("  downstream (gen + score, per row):       ~%d", n_down)
        logger.info("  judges: %s", judges)
        logger.info("[dry-run] no API calls fired.")
        return

    judges = _build_judges(args.judges)
    cache = JudgeCache(args.cache)
    results: dict[str, Any] = {"systems": systems, "tiers": tiers,
                               "n_conversations": len(conversations)}

    if "faithfulness" in metrics or "tier" in metrics:
        logger.info("Running M_faithfulness ...")
        faith_scores = run_faithfulness(comp_rows, conversations, judges, cache)
        if "faithfulness" in metrics:
            results["faithfulness"] = aggregate_faithfulness(
                faith_scores, our_system=args.our_system,
                seed=args.seed, n_boot=args.n_boot,
            )
        if "tier" in metrics:
            results["tier_appropriate"] = aggregate_tier(
                faith_scores, our_system=args.our_system
            )

    if "sanity" in metrics:
        logger.info("Running sanity metrics ...")
        sanity: dict[str, Any] = {}
        for row in comp_rows:
            key = f"{row.conversation_id}|{row.source}|{row.turn_age}"
            sanity[key] = {
                name: r.value
                for name, r in sanity_metrics.all_sanity_metrics(row.compressed).items()
            }
        results["sanity"] = sanity

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote pilot results to %s", args.out)


if __name__ == "__main__":
    main()
