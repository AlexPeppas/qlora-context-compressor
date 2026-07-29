"""Statistics layer for the head-to-head bake-off (Phase B.5).

Implements the v5 statistical design:

  * clustered_bootstrap_ci  — confidence intervals via resampling
    CONVERSATIONS (clusters), not rows. This is the PRIMARY CI method and
    works for deterministic extractive baselines (which have no seed
    variance). Rows within a conversation share structure, so resampling
    at the conversation level avoids overstating precision.

  * mcnemar_test            — paired significance on binarized win/loss
    (ours vs a competitor, per conversation). Exact binomial for small
    discordant counts, chi-square with continuity correction otherwise.

  * wilcoxon_paired         — paired signed-rank test for ordinal/continuous
    per-conversation deltas (keeps magnitude info McNemar discards).

  * cohens_kappa            — inter-judge agreement on categorical labels
    (e.g. faithfulness present/partial/false on a shared item list).

  * icc21                   — intraclass correlation ICC(2,1), two-way
    random effects single measure, for inter-judge agreement on
    continuous/ordinal scores (e.g. per-row faithfulness or downstream).

  * paired_win_loss         — helper to binarize two systems' per-conversation
    scores into (a_wins, b_wins, ties) for McNemar.

Design notes:
  * numpy for vectorised resampling; scipy for the Wilcoxon test and the
    exact binomial p-value only. kappa + ICC implemented directly so the
    formulas are auditable in the paper appendix.
  * All functions are deterministic given a seed (bootstrap RNG is seeded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Clustered bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    n_clusters: int
    n_boot: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "n_clusters": self.n_clusters,
            "n_boot": self.n_boot,
            "confidence": self.confidence,
        }


def clustered_bootstrap_ci(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI, resampling CLUSTERS (conversations) with
    replacement. `values` and `cluster_ids` are parallel arrays (one entry
    per row). On each bootstrap iteration we resample the set of unique
    clusters with replacement, gather all rows in the chosen clusters, and
    recompute `statistic`.

    Raises ValueError on empty/mismatched input.
    """
    if len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must be the same length")
    if len(values) == 0:
        raise ValueError("empty input")

    values_arr = np.asarray(values, dtype=float)
    clusters = np.asarray(cluster_ids)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Pre-index rows by cluster for fast gathering
    rows_by_cluster = {c: np.where(clusters == c)[0] for c in unique_clusters}

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in chosen])
        boot_stats[b] = statistic(values_arr[idx])

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    point = float(statistic(values_arr))
    return BootstrapCI(
        point=point,
        lower=lower,
        upper=upper,
        n_clusters=n_clusters,
        n_boot=n_boot,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Paired win/loss + McNemar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedWinLoss:
    a_wins: int
    b_wins: int
    ties: int

    @property
    def n(self) -> int:
        return self.a_wins + self.b_wins + self.ties


def paired_win_loss(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    *,
    eps: float = 1e-9,
) -> PairedWinLoss:
    """Binarize two systems' per-conversation scores into win/loss/tie.

    Both maps are keyed by conversation_id; only shared keys are compared.
    A wins if scores_a > scores_b + eps, B wins if the reverse, else tie.
    """
    shared = set(scores_a) & set(scores_b)
    a_wins = b_wins = ties = 0
    for k in shared:
        d = scores_a[k] - scores_b[k]
        if d > eps:
            a_wins += 1
        elif d < -eps:
            b_wins += 1
        else:
            ties += 1
    return PairedWinLoss(a_wins=a_wins, b_wins=b_wins, ties=ties)


@dataclass(frozen=True)
class McNemarResult:
    statistic: float
    p_value: float
    method: str  # "exact-binomial" | "chi2-continuity"
    a_wins: int
    b_wins: int

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "method": self.method,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
        }


def mcnemar_test(wl: PairedWinLoss, *, exact_threshold: int = 25) -> McNemarResult:
    """McNemar test on the discordant pairs (a_wins vs b_wins); ties are
    ignored (that's the design of the test). Uses the exact binomial test
    when the number of discordant pairs is small (<= exact_threshold),
    otherwise the chi-square approximation with continuity correction.
    """
    b = wl.a_wins
    c = wl.b_wins
    n_disc = b + c
    if n_disc == 0:
        return McNemarResult(
            statistic=0.0, p_value=1.0, method="degenerate-no-discordant",
            a_wins=b, b_wins=c,
        )

    if n_disc <= exact_threshold:
        from scipy.stats import binomtest  # noqa: PLC0415

        # Two-sided exact binomial with p=0.5
        res = binomtest(min(b, c), n=n_disc, p=0.5, alternative="two-sided")
        return McNemarResult(
            statistic=float(min(b, c)),
            p_value=float(res.pvalue),
            method="exact-binomial",
            a_wins=b,
            b_wins=c,
        )
    # Chi-square with continuity correction
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    from scipy.stats import chi2  # noqa: PLC0415

    p = float(chi2.sf(stat, df=1))
    return McNemarResult(
        statistic=float(stat), p_value=p, method="chi2-continuity",
        a_wins=b, b_wins=c,
    )


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank (ordinal / continuous paired)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n_nonzero: int

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_nonzero": self.n_nonzero,
        }


def wilcoxon_paired(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
) -> WilcoxonResult:
    """Wilcoxon signed-rank test on paired per-conversation scores. Keeps
    magnitude information McNemar discards. Uses scipy; returns a degenerate
    (p=1.0) result if all differences are zero."""
    shared = sorted(set(scores_a) & set(scores_b))
    diffs = np.array([scores_a[k] - scores_b[k] for k in shared], dtype=float)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return WilcoxonResult(statistic=0.0, p_value=1.0, n_nonzero=0)
    from scipy.stats import wilcoxon  # noqa: PLC0415

    res = wilcoxon(diffs, zero_method="wilcox", correction=False, mode="auto")
    return WilcoxonResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        n_nonzero=int(len(nonzero)),
    )


# ---------------------------------------------------------------------------
# Inter-judge agreement: Cohen's kappa (categorical)
# ---------------------------------------------------------------------------


def cohens_kappa(
    labels_a: Sequence[str], labels_b: Sequence[str]
) -> float:
    """Cohen's kappa for two raters over categorical labels. Parallel
    sequences; each element is a label string (e.g. 'present'/'partial'/'false').
    Returns kappa in [-1, 1]. Raises ValueError on length mismatch/empty."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must be the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("empty input")

    categories = sorted(set(labels_a) | set(labels_b))
    cat_index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    conf = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        conf[cat_index[a], cat_index[b]] += 1

    po = np.trace(conf) / n
    row_marg = conf.sum(axis=1) / n
    col_marg = conf.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))
    if pe == 1.0:
        # Perfect agreement by chance (all one category) -> define kappa=1
        return 1.0
    return float((po - pe) / (1.0 - pe))


# ---------------------------------------------------------------------------
# Inter-judge agreement: ICC(2,1) (continuous)
# ---------------------------------------------------------------------------


def icc21(ratings: Sequence[Sequence[float]]) -> float:
    """Intraclass correlation ICC(2,1): two-way random effects, single
    rater, absolute agreement.

    `ratings` is an n_subjects x n_raters matrix (each row = one item, each
    column = one judge). Returns ICC in (-inf, 1]; typically [0, 1].

    Formula (Shrout & Fleiss 1979):
        ICC(2,1) = (MSR - MSE) /
                   (MSR + (k-1)*MSE + k*(MSC - MSE)/n)
    where MSR = between-subjects mean square, MSC = between-raters mean
    square, MSE = residual mean square, n = subjects, k = raters.
    """
    mat = np.asarray(ratings, dtype=float)
    if mat.ndim != 2:
        raise ValueError("ratings must be a 2D n_subjects x n_raters matrix")
    n, k = mat.shape
    if n < 2 or k < 2:
        raise ValueError("ICC needs at least 2 subjects and 2 raters")

    grand_mean = mat.mean()
    row_means = mat.mean(axis=1)
    col_means = mat.mean(axis=0)

    # Sum of squares
    ss_total = ((mat - grand_mean) ** 2).sum()
    ss_row = k * ((row_means - grand_mean) ** 2).sum()  # between subjects
    ss_col = n * ((col_means - grand_mean) ** 2).sum()  # between raters
    ss_err = ss_total - ss_row - ss_col

    df_row = n - 1
    df_col = k - 1
    df_err = (n - 1) * (k - 1)

    msr = ss_row / df_row
    msc = ss_col / df_col
    mse = ss_err / df_err if df_err > 0 else 0.0

    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom == 0:
        return 1.0 if msr == mse else 0.0
    return float((msr - mse) / denom)


__all__ = [
    "BootstrapCI",
    "McNemarResult",
    "PairedWinLoss",
    "WilcoxonResult",
    "clustered_bootstrap_ci",
    "cohens_kappa",
    "icc21",
    "mcnemar_test",
    "paired_win_loss",
    "wilcoxon_paired",
]
