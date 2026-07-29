"""Tests for compressor.eval.downstream — M_downstream ensemble judge.

Uses MockJudgeClient. No real API calls. Covers:
  * continuation generation via the structured path
  * per-judge 3-axis scoring + per-row mean (B excluded when null)
  * ensemble aggregation across 2 judges (mean, spread, per-judge)
  * axis-decomposition ensembles
  * cached-continuation path (skip generation)
  * error paths
"""
from __future__ import annotations

import pytest

from compressor.eval.downstream import (
    AxisScores,
    ContinuationText,
    DownstreamEvaluation,
    evaluate,
    generate_continuation,
    score_continuation,
)
from compressor.eval.ensemble import ensemble_ordinal, ensemble_scalar
from compressor.eval.llm_client import MockJudgeClient
from compressor.eval.prompts import load_prompt


# ---------------------------------------------------------------------------
# AxisScores.per_row_score
# ---------------------------------------------------------------------------


def test_per_row_score_includes_fidelity_when_present():
    a = AxisScores(substance=5, fidelity=3, coherence=4)
    # (5 + 3 + 4) / 3
    assert a.per_row_score() == pytest.approx(4.0)


def test_per_row_score_excludes_fidelity_when_null():
    a = AxisScores(substance=5, fidelity=None, coherence=3)
    # (5 + 3) / 2
    assert a.per_row_score() == pytest.approx(4.0)


def test_axis_bounds_enforced():
    with pytest.raises(ValueError):
        AxisScores(substance=6, coherence=3)
    with pytest.raises(ValueError):
        AxisScores(substance=0, coherence=3)


# ---------------------------------------------------------------------------
# Ensemble helper
# ---------------------------------------------------------------------------


def test_ensemble_scalar_two_judges():
    e = ensemble_scalar({"gpt": 4.0, "claude": 5.0})
    assert e.mean == pytest.approx(4.5)
    assert e.n_judges == 2
    assert e.spread == pytest.approx(1.0)
    assert e.stdev == pytest.approx(0.7071, abs=1e-3)


def test_ensemble_scalar_single_judge_zero_spread():
    e = ensemble_scalar({"gpt": 3.0})
    assert e.mean == 3.0
    assert e.stdev == 0.0
    assert e.spread == 0.0


def test_ensemble_scalar_rejects_empty():
    with pytest.raises(ValueError):
        ensemble_scalar({})


def test_ensemble_ordinal_rounds():
    e = ensemble_ordinal({"a": 4.0, "b": 5.0})
    assert e.mean == pytest.approx(4.5)
    assert e.rounded_mean == 4  # round(4.5) -> 4 (banker's rounding)


# ---------------------------------------------------------------------------
# Continuation generation
# ---------------------------------------------------------------------------


def test_generate_continuation_returns_text():
    gen = MockJudgeClient(name="gpt-5.5-gen")
    prompt = load_prompt("downstream_continue_v1")
    _, user = prompt.render(compression="summary here", last_user_turn="what next?")
    gen.register_response(
        "downstream_continue_v1", user, ContinuationText(text="Here is my answer.")
    )
    text, result = generate_continuation("summary here", "what next?", gen)
    assert text == "Here is my answer."
    assert result.provenance.prompt_name == "downstream_continue_v1"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_continuation_single_judge():
    judge = MockJudgeClient(name="gpt-5.5")
    prompt = load_prompt("downstream_score_v1")
    _, user = prompt.render(
        last_user_turn="q", continuation="gen", ground_truth="gt"
    )
    judge.register_response(
        "downstream_score_v1",
        user,
        AxisScores(substance=4, fidelity=5, coherence=4),
    )
    axes, _ = score_continuation("q", "gen", "gt", judge)
    assert axes.substance == 4
    assert axes.per_row_score() == pytest.approx((4 + 5 + 4) / 3)


# ---------------------------------------------------------------------------
# End-to-end ensemble evaluate()
# ---------------------------------------------------------------------------


def _register_score(judge: MockJudgeClient, axes: AxisScores) -> None:
    prompt = load_prompt("downstream_score_v1")
    _, user = prompt.render(last_user_turn="q", continuation="GEN", ground_truth="gt")
    judge.register_response("downstream_score_v1", user, axes)


def test_evaluate_ensemble_two_judges_cached_continuation():
    from compressor.eval.llm_client import (
        JudgeProvenance,
        JudgeResult,
        TokenUsage,
    )

    judge_a = MockJudgeClient(name="gpt-5.5")
    judge_b = MockJudgeClient(name="claude-sonnet-4-6")
    _register_score(judge_a, AxisScores(substance=5, fidelity=4, coherence=5))
    _register_score(judge_b, AxisScores(substance=4, fidelity=4, coherence=3))

    # Fabricate a generation result so we skip generation
    fake_gen = JudgeResult(
        parsed=ContinuationText(text="GEN"),
        raw="GEN",
        provenance=JudgeProvenance(
            judge_name="gen", model="m", snapshot_id="m", backend="mock",
            prompt_hash="h", prompt_name="downstream_continue_v1",
            schema_hash="s", schema_name="ContinuationText",
            temperature=0.0, seed=42, seed_supported=True,
        ),
        usage=TokenUsage(),
        wall_seconds=0.0,
    )

    ev = evaluate(
        last_user_turn="q",
        ground_truth="gt",
        compression="cmp",
        generator=judge_a,  # unused since continuation cached
        judges=[judge_a, judge_b],
        continuation="GEN",
        generation_result=fake_gen,
    )
    # judge_a per-row = (5+4+5)/3 = 4.667 ; judge_b = (4+4+3)/3 = 3.667
    assert ev.per_judge_score["gpt-5.5"] == pytest.approx((5 + 4 + 5) / 3)
    assert ev.per_judge_score["claude-sonnet-4-6"] == pytest.approx((4 + 4 + 3) / 3)
    # ensemble mean = mean of the two per-row scores
    assert ev.ensemble.mean == pytest.approx(
        ((5 + 4 + 5) / 3 + (4 + 4 + 3) / 3) / 2
    )
    assert ev.ensemble.n_judges == 2
    # axis-decomposition
    assert ev.ensemble_substance.mean == pytest.approx(4.5)
    assert ev.ensemble_coherence.mean == pytest.approx(4.0)
    assert ev.ensemble_fidelity is not None
    assert ev.ensemble_fidelity.mean == pytest.approx(4.0)


def test_evaluate_fidelity_none_when_all_judges_na():
    from compressor.eval.llm_client import (
        JudgeProvenance,
        JudgeResult,
        TokenUsage,
    )

    judge_a = MockJudgeClient(name="gpt-5.5")
    judge_b = MockJudgeClient(name="claude-sonnet-4-6")
    _register_score(judge_a, AxisScores(substance=5, fidelity=None, coherence=5))
    _register_score(judge_b, AxisScores(substance=4, fidelity=None, coherence=4))

    fake_gen = JudgeResult(
        parsed=ContinuationText(text="GEN"),
        raw="GEN",
        provenance=JudgeProvenance(
            judge_name="gen", model="m", snapshot_id="m", backend="mock",
            prompt_hash="h", prompt_name="downstream_continue_v1",
            schema_hash="s", schema_name="ContinuationText",
            temperature=0.0, seed=42, seed_supported=True,
        ),
        usage=TokenUsage(),
        wall_seconds=0.0,
    )

    ev = evaluate(
        last_user_turn="q",
        ground_truth="gt",
        compression="cmp",
        generator=judge_a,
        judges=[judge_a, judge_b],
        continuation="GEN",
        generation_result=fake_gen,
    )
    assert ev.ensemble_fidelity is None
    # per-row scores exclude fidelity
    assert ev.per_judge_score["gpt-5.5"] == pytest.approx(5.0)
    assert ev.per_judge_score["claude-sonnet-4-6"] == pytest.approx(4.0)


def test_evaluate_rejects_no_judges():
    with pytest.raises(ValueError, match="at least one judge"):
        evaluate(
            last_user_turn="q",
            ground_truth="gt",
            compression="cmp",
            generator=MockJudgeClient(),
            judges=[],
            continuation="x",
            generation_result=None,
        )


def test_evaluate_rejects_partial_continuation_cache():
    judge = MockJudgeClient()
    with pytest.raises(ValueError, match="either both"):
        evaluate(
            last_user_turn="q",
            ground_truth="gt",
            compression="cmp",
            generator=judge,
            judges=[judge],
            continuation="x",
            generation_result=None,
        )


def test_downstream_evaluation_to_dict_serializable():
    import json
    from compressor.eval.llm_client import (
        JudgeProvenance,
        JudgeResult,
        TokenUsage,
    )

    judge_a = MockJudgeClient(name="gpt-5.5")
    _register_score(judge_a, AxisScores(substance=5, fidelity=4, coherence=5))
    fake_gen = JudgeResult(
        parsed=ContinuationText(text="GEN"),
        raw="GEN",
        provenance=JudgeProvenance(
            judge_name="gen", model="m", snapshot_id="m", backend="mock",
            prompt_hash="h", prompt_name="downstream_continue_v1",
            schema_hash="s", schema_name="ContinuationText",
            temperature=0.0, seed=42, seed_supported=True,
        ),
        usage=TokenUsage(),
        wall_seconds=0.0,
    )
    ev = evaluate(
        last_user_turn="q",
        ground_truth="gt",
        compression="cmp",
        generator=judge_a,
        judges=[judge_a],
        continuation="GEN",
        generation_result=fake_gen,
    )
    # Must be JSON-serializable for result files
    s = json.dumps(ev.to_dict())
    assert "ensemble" in s
