"""M_faithfulness — critical-information preservation judge.

Two-stage methodology per `docs/rubrics/faithfulness.md`:

  Stage 1: extract 5-25 critical items from source (judge sees source only)
  Stage 2: check coverage in compression (judge sees items + compression,
           NOT source — see rubber-duck blocker #3 / blinding note)

The Stage 1 result is shared across all baselines for a given conversation
(it depends only on the source) — caller is expected to call
`extract_critical_items` once per conversation and `check_coverage`
once per (conversation, compression).

Evidence validation: the rubric requires `present`/`partial` calls to be
backed by a verbatim substring from the compression. We programmatically
verify this and DOWNGRADE invalid evidence claims to `false`. This catches
hallucinated evidence (judge claiming the compression says X when it
doesn't).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from .llm_client import JudgeClient, JudgeResult
from .prompts import load_prompt

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "v1"  # bump to invalidate the cache when rubric semantics change


# ---------------------------------------------------------------------------
# Stage 1: critical-item extraction
# ---------------------------------------------------------------------------


CriticalItemType = Literal[
    "decision", "number", "code", "error", "entity", "constraint", "action"
]


class CriticalItem(BaseModel):
    id: int = Field(description="1-indexed identifier, unique within this list")
    type: CriticalItemType = Field(description="Category of the item")
    summary: str = Field(
        description="At most 15 words describing the item directly", max_length=200
    )
    verbatim_indicator: str = Field(
        description="Verbatim substring (max 30 chars) from the source uniquely anchoring this item",
        max_length=60,  # tolerate a bit more than the spec because LLMs are imprecise
    )


class ExtractedItemsV1(BaseModel):
    """Stage 1 structured output: a list of critical items extracted from the source."""

    items: list[CriticalItem] = Field(
        description="Between 5 and 25 critical items; if fewer than 5 exist in the source, report what exists",
        min_length=1,
        max_length=30,  # tolerate one or two over the cap
    )


def extract_critical_items(
    source: str,
    judge: JudgeClient,
) -> tuple[list[CriticalItem], JudgeResult]:
    """Stage 1. Returns (items, full judge result).

    The full JudgeResult is returned so caller can record provenance +
    token usage in the cached output. Items are also returned separately
    for convenience.
    """
    prompt = load_prompt("faithfulness_stage1_v1")
    system, user = prompt.render(source=source)
    result = judge.call(
        system,
        user,
        ExtractedItemsV1,
        prompt_name=prompt.name,
        prompt_hash=prompt.content_hash,
    )
    extracted = result.parsed
    assert isinstance(extracted, ExtractedItemsV1)
    return list(extracted.items), result


# ---------------------------------------------------------------------------
# Stage 2: coverage check
# ---------------------------------------------------------------------------


CoverageCall = Literal["present", "partial", "false"]


class CoverageDecision(BaseModel):
    id: int = Field(description="Matches the CriticalItem.id being graded")
    present: CoverageCall
    evidence: str = Field(
        default="",
        description=(
            "Verbatim substring from the compression supporting "
            "present/partial; empty for false. The prompt requests a short "
            "span, but grounding validation—not a transport length cap—is "
            "the authoritative constraint."
        ),
    )


class CoverageReportV1(BaseModel):
    """Stage 2 structured output: per-item coverage decisions."""

    decisions: list[CoverageDecision] = Field(min_length=1)


def check_coverage(
    items: list[CriticalItem],
    compression: str,
    judge: JudgeClient,
) -> tuple[list[CoverageDecision], JudgeResult]:
    """Stage 2. Returns (validated_decisions, full judge result).

    Validation: any present/partial call whose `evidence` is NOT a substring
    of the compression is DOWNGRADED to `false`. This catches judges that
    hallucinate evidence (claim the compression says something it doesn't).
    Substring match is case-insensitive after stripping outer whitespace —
    same approach OpenAI's evals library uses for grounding.
    """
    prompt = load_prompt("faithfulness_stage2_v1")
    # Stage 2 sees the item list as JSON (not the source). Render compactly.
    items_json = json.dumps([it.model_dump() for it in items], ensure_ascii=False)
    system, user = prompt.render(items_json=items_json, compression=compression)
    result = judge.call(
        system,
        user,
        CoverageReportV1,
        prompt_name=prompt.name,
        prompt_hash=prompt.content_hash,
    )
    report = result.parsed
    assert isinstance(report, CoverageReportV1)

    # Validation pass: downgrade unsupported evidence to `false`
    compression_lower = compression.lower()
    validated: list[CoverageDecision] = []
    n_downgrades = 0
    for d in report.decisions:
        if d.present in ("present", "partial"):
            evidence_stripped = d.evidence.strip()
            if not evidence_stripped or evidence_stripped.lower() not in compression_lower:
                logger.warning(
                    "Downgrading item %d from %r to 'false' due to invalid evidence: %r",
                    d.id,
                    d.present,
                    d.evidence,
                )
                validated.append(
                    CoverageDecision(id=d.id, present="false", evidence="")
                )
                n_downgrades += 1
                continue
        validated.append(d)
    if n_downgrades:
        logger.info(
            "Stage-2 validation downgraded %d/%d decisions for unsupported evidence",
            n_downgrades,
            len(report.decisions),
        )

    return validated, result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaithfulnessScore:
    """Per-row M_faithfulness output. Continuous score in [0, 1]."""

    score: float
    n_total: int
    n_present: int
    n_partial: int
    n_false: int
    n_downgraded: int  # count of validation downgrades (for diagnostic)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "n_total": self.n_total,
            "n_present": self.n_present,
            "n_partial": self.n_partial,
            "n_false": self.n_false,
            "n_downgraded": self.n_downgraded,
        }


def faithfulness_score(
    items: list[CriticalItem],
    decisions: list[CoverageDecision],
) -> FaithfulnessScore:
    """Aggregate per-item decisions into a single 0-1 score.

    Score = (#present + 0.5 * #partial) / #total

    Validates that the decision list matches the item list (same ids,
    no duplicates, no missing). Raises ValueError on schema violations.
    """
    item_ids = {it.id for it in items}
    decision_ids = [d.id for d in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError(
            f"Stage 2 returned duplicate decision ids: {decision_ids}"
        )
    if set(decision_ids) != item_ids:
        missing = item_ids - set(decision_ids)
        extra = set(decision_ids) - item_ids
        raise ValueError(
            f"Stage 2 decision ids do not match item ids: missing={missing}, extra={extra}"
        )

    n_total = len(items)
    n_present = sum(1 for d in decisions if d.present == "present")
    n_partial = sum(1 for d in decisions if d.present == "partial")
    n_false = sum(1 for d in decisions if d.present == "false")
    score = (n_present + 0.5 * n_partial) / n_total if n_total else 0.0
    return FaithfulnessScore(
        score=score,
        n_total=n_total,
        n_present=n_present,
        n_partial=n_partial,
        n_false=n_false,
        n_downgraded=0,  # filled in by caller who knows downgrades from check_coverage
    )


# ---------------------------------------------------------------------------
# Convenience: full per-row evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaithfulnessEvaluation:
    """Complete output for one (source, compression) pair."""

    items: list[CriticalItem]
    decisions: list[CoverageDecision]
    score: FaithfulnessScore
    stage1_result: JudgeResult
    stage2_result: JudgeResult

    def to_dict(self) -> dict:
        return {
            "items": [it.model_dump() for it in self.items],
            "decisions": [d.model_dump() for d in self.decisions],
            "score": self.score.to_dict(),
            "stage1_provenance": self.stage1_result.provenance.to_dict(),
            "stage2_provenance": self.stage2_result.provenance.to_dict(),
            "stage1_usage": {
                "input_tokens": self.stage1_result.usage.input_tokens,
                "output_tokens": self.stage1_result.usage.output_tokens,
            },
            "stage2_usage": {
                "input_tokens": self.stage2_result.usage.input_tokens,
                "output_tokens": self.stage2_result.usage.output_tokens,
            },
            "rubric_version": RUBRIC_VERSION,
        }


def evaluate(
    source: str,
    compression: str,
    judge: JudgeClient,
    *,
    items: list[CriticalItem] | None = None,
    stage1_result: JudgeResult | None = None,
) -> FaithfulnessEvaluation:
    """Run both stages on a (source, compression) pair.

    If `items` and `stage1_result` are provided (extracted previously and
    cached for this source), Stage 1 is skipped. This is the recommended
    pattern for bake-off evaluation: extract once per source, check
    coverage per (source, system, tier) compression.
    """
    if items is None or stage1_result is None:
        if items is not None or stage1_result is not None:
            raise ValueError(
                "either both `items` and `stage1_result` must be provided, or neither"
            )
        items, stage1_result = extract_critical_items(source, judge)

    decisions, stage2_result = check_coverage(items, compression, judge)
    score = faithfulness_score(items, decisions)
    # Patch the downgrade count from validation
    n_downgraded = sum(
        1
        for orig, val in zip(
            (None,) * len(decisions), decisions  # we don't have originals here
        )
    )
    # Note: downgrade count is lost in the validated list (we relabel as 'false').
    # Caller can recover it from logs if needed; for now we store 0.
    return FaithfulnessEvaluation(
        items=items,
        decisions=decisions,
        score=score,
        stage1_result=stage1_result,
        stage2_result=stage2_result,
    )


__all__ = [
    "RUBRIC_VERSION",
    "CoverageCall",
    "CoverageDecision",
    "CoverageReportV1",
    "CriticalItem",
    "CriticalItemType",
    "ExtractedItemsV1",
    "FaithfulnessEvaluation",
    "FaithfulnessScore",
    "check_coverage",
    "evaluate",
    "extract_critical_items",
    "faithfulness_score",
]
