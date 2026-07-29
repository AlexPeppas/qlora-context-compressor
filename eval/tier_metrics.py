"""M_tier_appropriate — information-density curve metrics.

Derived from M_faithfulness (no extra LLM calls): given a compressor's
per-tier faithfulness scores (f_recent, f_mid, f_old) for one conversation,
we characterise the SHAPE of the retention curve. A tier-conditioned
compressor should drop retention monotonically across tiers (recent high,
old low); a tier-blind compressor (LLMLingua, base prompted) should be
roughly flat.

Per docs/rubrics/tier_appropriate.md. Curve-shape metrics avoid
pre-registering target retention percentages (which would be tuning the
rubric to our conclusion). The metric only fires when the compressor
actually treats tiers differently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Monotonicity tolerance: ties within EPS count as non-decreasing.
MONOTONIC_EPS = 0.05

TIER_ORDER = ("recent", "mid", "old")


@dataclass(frozen=True)
class CurveStats:
    """Curve-shape stats for one (conversation, system) triple of tier scores."""

    f_recent: float
    f_mid: float
    f_old: float
    delta_recent_old: float
    delta_recent_mid: float
    delta_mid_old: float
    monotonic: bool  # f_recent >= f_mid >= f_old within EPS
    curve_auc: float  # trapezoidal area under (recent, mid, old) at x=0,1,2

    def to_dict(self) -> dict:
        return {
            "f_recent": self.f_recent,
            "f_mid": self.f_mid,
            "f_old": self.f_old,
            "delta_recent_old": self.delta_recent_old,
            "delta_recent_mid": self.delta_recent_mid,
            "delta_mid_old": self.delta_mid_old,
            "monotonic": self.monotonic,
            "curve_auc": self.curve_auc,
        }


def compute_curve(faithfulness_by_tier: Mapping[str, float]) -> CurveStats:
    """Compute curve-shape stats from a {tier: faithfulness} mapping.

    Requires all three tiers present. Raises KeyError otherwise.
    """
    missing = [t for t in TIER_ORDER if t not in faithfulness_by_tier]
    if missing:
        raise KeyError(f"missing tier scores: {missing}")

    fr = float(faithfulness_by_tier["recent"])
    fm = float(faithfulness_by_tier["mid"])
    fo = float(faithfulness_by_tier["old"])

    d_ro = fr - fo
    d_rm = fr - fm
    d_mo = fm - fo

    # Monotone non-increasing within tolerance
    monotonic = (fr - fm >= -MONOTONIC_EPS) and (fm - fo >= -MONOTONIC_EPS)

    # Trapezoidal AUC over x = 0 (recent), 1 (mid), 2 (old)
    curve_auc = 0.5 * (fr + fm) + 0.5 * (fm + fo)

    return CurveStats(
        f_recent=fr,
        f_mid=fm,
        f_old=fo,
        delta_recent_old=d_ro,
        delta_recent_mid=d_rm,
        delta_mid_old=d_mo,
        monotonic=monotonic,
        curve_auc=curve_auc,
    )


@dataclass(frozen=True)
class TierAggregate:
    """Per-system aggregation of curve stats across a conversation set."""

    n: int
    mean_delta_recent_old: float
    monotonicity_rate: float  # fraction of conversations that are monotonic
    mean_curve_auc: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_delta_recent_old": self.mean_delta_recent_old,
            "monotonicity_rate": self.monotonicity_rate,
            "mean_curve_auc": self.mean_curve_auc,
        }


def aggregate_curves(curves: list[CurveStats]) -> TierAggregate:
    """Aggregate a system's per-conversation curves into headline tier metrics."""
    if not curves:
        raise ValueError("aggregate_curves requires at least one curve")
    n = len(curves)
    mean_dro = sum(c.delta_recent_old for c in curves) / n
    mono_rate = sum(1 for c in curves if c.monotonic) / n
    mean_auc = sum(c.curve_auc for c in curves) / n
    return TierAggregate(
        n=n,
        mean_delta_recent_old=mean_dro,
        monotonicity_rate=mono_rate,
        mean_curve_auc=mean_auc,
    )


__all__ = [
    "MONOTONIC_EPS",
    "TIER_ORDER",
    "CurveStats",
    "TierAggregate",
    "aggregate_curves",
    "compute_curve",
]
