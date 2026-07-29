"""N-judge ensemble aggregation.

The paper uses a two-family judge ensemble (GPT-5.5 + Claude Sonnet 4.6),
headline = mean of the judges, with inter-judge agreement reported
alongside. This module keeps that aggregation generic so a third family
(Gemini) can be added as config only.

Two aggregation entry points:

  * `ensemble_scalar` — combine per-judge scalar scores (float) into a
    mean + spread + per-judge breakdown. Used for M_faithfulness (each
    judge independently runs Stage 1+2 and produces one 0-1 score) and
    for each axis of M_downstream.

  * `ensemble_ordinal` — combine per-judge 1-5 ordinal scores; same as
    scalar but also exposes the integer-rounded mean for reporting.

Agreement statistics live in eval.stats (Phase B.5) — this module only
does the aggregation. We keep the per-judge values so stats can compute
ICC / kappa over the full matrix later.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class EnsembleScore:
    """Aggregated score across N judges for one item.

    `per_judge` maps judge_name -> score so downstream stats can compute
    inter-judge agreement over the whole dataset. `mean` is the headline.
    """

    mean: float
    per_judge: Mapping[str, float]
    n_judges: int
    stdev: float  # 0.0 for a single judge
    spread: float  # max - min across judges (0.0 for single judge)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "per_judge": dict(self.per_judge),
            "n_judges": self.n_judges,
            "stdev": self.stdev,
            "spread": self.spread,
        }


def ensemble_scalar(per_judge: Mapping[str, float]) -> EnsembleScore:
    """Combine per-judge scalar scores into an EnsembleScore.

    Raises ValueError on empty input. For a single judge, stdev/spread
    are 0.0 (degenerate ensemble — still valid, just not robust).
    """
    if not per_judge:
        raise ValueError("ensemble_scalar requires at least one judge score")
    values = list(per_judge.values())
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    spread = (max(values) - min(values)) if len(values) > 1 else 0.0
    return EnsembleScore(
        mean=mean,
        per_judge=dict(per_judge),
        n_judges=len(values),
        stdev=stdev,
        spread=spread,
    )


@dataclass(frozen=True)
class OrdinalEnsembleScore(EnsembleScore):
    """Ordinal (1-5) ensemble adds a rounded integer for display."""

    rounded_mean: int = 0

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["rounded_mean"] = self.rounded_mean
        return d


def ensemble_ordinal(per_judge: Mapping[str, float]) -> OrdinalEnsembleScore:
    """Combine per-judge ordinal 1-5 scores. `rounded_mean` is the mean
    rounded to the nearest integer (ties-to-even via Python round)."""
    base = ensemble_scalar(per_judge)
    return OrdinalEnsembleScore(
        mean=base.mean,
        per_judge=base.per_judge,
        n_judges=base.n_judges,
        stdev=base.stdev,
        spread=base.spread,
        rounded_mean=int(round(base.mean)),
    )


__all__ = [
    "EnsembleScore",
    "OrdinalEnsembleScore",
    "ensemble_ordinal",
    "ensemble_scalar",
]
