"""M_downstream — conversation-continuation judge (ensemble).

Pipeline per (source_conversation, system, tier) compression:

  1. generate_continuation(): a fixed continuation LLM (temp=0, same for
     ALL baselines — baseline parity) answers the held-out last user turn
     using ONLY the compression as memory.
  2. score_continuation(): each judge in the ensemble scores the generated
     continuation against the held-out ground-truth assistant turn on 3
     axes (A substance, B code/numeric fidelity [nullable], C coherence),
     each 1-5. Judge does NOT see the compression or the source prior turns.
  3. Aggregate: per-judge per-row score = mean of applicable axes; ensemble
     headline = mean across judges.

Holdout inputs (last_user_turn, ground_truth) come from
Conversation.with_holdout() (Phase B.0) and are carried in the downstream
bake-off rows produced by run_downstream_bakeoff (Phase B.0.5).

Baseline parity: the continuation generator is identical across baselines
so the metric isolates compression quality, not generator quality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from pydantic import BaseModel, Field

from .ensemble import EnsembleScore, ensemble_scalar
from .llm_client import JudgeClient, JudgeResult
from .prompts import load_prompt

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "v1"


# ---------------------------------------------------------------------------
# Continuation generation
# ---------------------------------------------------------------------------


def generate_continuation(
    compression: str,
    last_user_turn: str,
    generator: JudgeClient,
    *,
    max_tokens: int = 1024,
) -> tuple[str, JudgeResult]:
    """Generate the assistant continuation from [compression + last user turn].

    Uses the JudgeClient interface for uniform provenance/retry, but this is
    a GENERATION call, not a judging call. The same generator (temp=0) must
    be used across all baselines for parity. Returns (text, full result).

    The generator must be configured with a plain free-text response model
    (see ContinuationText below) so we get a string back through the same
    structured path.
    """
    prompt = load_prompt("downstream_continue_v1")
    system, user = prompt.render(compression=compression, last_user_turn=last_user_turn)
    result = generator.call(
        system,
        user,
        ContinuationText,
        prompt_name=prompt.name,
        prompt_hash=prompt.content_hash,
        max_tokens=max_tokens,
    )
    parsed = result.parsed
    assert isinstance(parsed, ContinuationText)
    return parsed.text, result


class ContinuationText(BaseModel):
    """Structured wrapper so continuation generation flows through the same
    structured-output path as judging. The model returns its full assistant
    reply in `text`."""

    text: str = Field(description="The assistant's full reply to the user")


# ---------------------------------------------------------------------------
# Scoring (per judge)
# ---------------------------------------------------------------------------


class AxisScores(BaseModel):
    """One judge's 3-axis scoring of a continuation vs ground truth."""

    substance: int = Field(ge=1, le=5, description="Axis A: substantive correctness")
    substance_rationale: str = Field(default="", max_length=400)
    fidelity: int | None = Field(
        default=None,
        description="Axis B: code/numeric fidelity, 1-5, or null if ground truth has no code/numbers/entities",
    )
    fidelity_rationale: str = Field(default="", max_length=400)
    coherence: int = Field(ge=1, le=5, description="Axis C: conversation coherence")
    coherence_rationale: str = Field(default="", max_length=400)

    def per_row_score(self) -> float:
        """Mean of applicable axes (B excluded when null)."""
        axes = [float(self.substance), float(self.coherence)]
        if self.fidelity is not None:
            axes.append(float(self.fidelity))
        return sum(axes) / len(axes)


def score_continuation(
    last_user_turn: str,
    continuation: str,
    ground_truth: str,
    judge: JudgeClient,
) -> tuple[AxisScores, JudgeResult]:
    """One judge scores one continuation. Returns (axis scores, full result)."""
    prompt = load_prompt("downstream_score_v1")
    system, user = prompt.render(
        last_user_turn=last_user_turn,
        continuation=continuation,
        ground_truth=ground_truth,
    )
    result = judge.call(
        system,
        user,
        AxisScores,
        prompt_name=prompt.name,
        prompt_hash=prompt.content_hash,
    )
    parsed = result.parsed
    assert isinstance(parsed, AxisScores)
    return parsed, result


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownstreamEvaluation:
    """Complete M_downstream output for one (source, system, tier) row."""

    continuation: str
    per_judge_axes: Mapping[str, AxisScores]  # judge_name -> axis scores
    per_judge_score: Mapping[str, float]  # judge_name -> mean-of-axes 1-5
    ensemble: EnsembleScore  # headline = ensemble.mean
    # Per-axis ensembles for the paper's axis-decomposition table
    ensemble_substance: EnsembleScore
    ensemble_coherence: EnsembleScore
    ensemble_fidelity: EnsembleScore | None  # None if all judges said N/A
    generation_result: JudgeResult
    judge_results: Mapping[str, JudgeResult]

    def to_dict(self) -> dict:
        return {
            "continuation": self.continuation,
            "per_judge_axes": {k: v.model_dump() for k, v in self.per_judge_axes.items()},
            "per_judge_score": dict(self.per_judge_score),
            "ensemble": self.ensemble.to_dict(),
            "ensemble_substance": self.ensemble_substance.to_dict(),
            "ensemble_coherence": self.ensemble_coherence.to_dict(),
            "ensemble_fidelity": self.ensemble_fidelity.to_dict()
            if self.ensemble_fidelity
            else None,
            "generation_provenance": self.generation_result.provenance.to_dict(),
            "judge_provenance": {
                k: v.provenance.to_dict() for k, v in self.judge_results.items()
            },
            "rubric_version": RUBRIC_VERSION,
        }


def evaluate(
    last_user_turn: str,
    ground_truth: str,
    compression: str,
    generator: JudgeClient,
    judges: Sequence[JudgeClient],
    *,
    continuation: str | None = None,
    generation_result: JudgeResult | None = None,
) -> DownstreamEvaluation:
    """Full M_downstream evaluation for one compression.

    If `continuation` + `generation_result` are provided, generation is
    skipped (caller cached it). Otherwise a continuation is generated with
    `generator`. Each judge in `judges` then scores the continuation; the
    ensemble mean is the headline.

    Baseline parity requires the SAME `generator` across all baselines.
    """
    if not judges:
        raise ValueError("evaluate requires at least one judge")

    if continuation is None or generation_result is None:
        if continuation is not None or generation_result is not None:
            raise ValueError(
                "either both `continuation` and `generation_result` or neither"
            )
        continuation, generation_result = generate_continuation(
            compression, last_user_turn, generator
        )

    per_judge_axes: dict[str, AxisScores] = {}
    per_judge_score: dict[str, float] = {}
    judge_results: dict[str, JudgeResult] = {}
    subs: dict[str, float] = {}
    cohs: dict[str, float] = {}
    fids: dict[str, float] = {}
    for judge in judges:
        axes, result = score_continuation(
            last_user_turn, continuation, ground_truth, judge
        )
        name = judge.name
        per_judge_axes[name] = axes
        per_judge_score[name] = axes.per_row_score()
        judge_results[name] = result
        subs[name] = float(axes.substance)
        cohs[name] = float(axes.coherence)
        if axes.fidelity is not None:
            fids[name] = float(axes.fidelity)

    ensemble = ensemble_scalar(per_judge_score)
    ensemble_substance = ensemble_scalar(subs)
    ensemble_coherence = ensemble_scalar(cohs)
    ensemble_fidelity = ensemble_scalar(fids) if fids else None

    return DownstreamEvaluation(
        continuation=continuation,
        per_judge_axes=per_judge_axes,
        per_judge_score=per_judge_score,
        ensemble=ensemble,
        ensemble_substance=ensemble_substance,
        ensemble_coherence=ensemble_coherence,
        ensemble_fidelity=ensemble_fidelity,
        generation_result=generation_result,
        judge_results=judge_results,
    )


__all__ = [
    "RUBRIC_VERSION",
    "AxisScores",
    "ContinuationText",
    "DownstreamEvaluation",
    "evaluate",
    "generate_continuation",
    "score_continuation",
]
