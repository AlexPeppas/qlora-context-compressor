"""Tests for tier_metrics, sanity_metrics, and stats.

Stats tests validate against hand-computed or scipy-cross-checked values.
"""
from __future__ import annotations

import numpy as np
import pytest

from compressor.eval.sanity_metrics import (
    all_sanity_metrics,
    bracketed_tag_format,
    meta_leakage,
    repetition_n8,
    surface_natural_end,
)
from compressor.eval.stats import (
    clustered_bootstrap_ci,
    cohens_kappa,
    icc21,
    mcnemar_test,
    paired_win_loss,
    wilcoxon_paired,
)
from compressor.eval.tier_metrics import aggregate_curves, compute_curve


# ===========================================================================
# tier_metrics
# ===========================================================================


def test_curve_monotonic_decreasing():
    c = compute_curve({"recent": 0.95, "mid": 0.70, "old": 0.30})
    assert c.delta_recent_old == pytest.approx(0.65)
    assert c.monotonic is True
    # AUC = 0.5*(0.95+0.70) + 0.5*(0.70+0.30) = 0.825 + 0.5 = 1.325
    assert c.curve_auc == pytest.approx(1.325)


def test_curve_flat_is_monotonic_but_zero_delta():
    c = compute_curve({"recent": 0.6, "mid": 0.6, "old": 0.6})
    assert c.delta_recent_old == pytest.approx(0.0)
    assert c.monotonic is True  # flat within eps


def test_curve_reversed_not_monotonic():
    c = compute_curve({"recent": 0.3, "mid": 0.5, "old": 0.9})
    assert c.delta_recent_old == pytest.approx(-0.6)
    assert c.monotonic is False


def test_curve_within_eps_tolerance():
    # small upward blip within EPS still counts monotonic
    c = compute_curve({"recent": 0.70, "mid": 0.73, "old": 0.40})
    assert c.monotonic is True  # 0.70 -> 0.73 is within 0.05


def test_curve_missing_tier_raises():
    with pytest.raises(KeyError):
        compute_curve({"recent": 0.9, "mid": 0.5})


def test_aggregate_curves():
    curves = [
        compute_curve({"recent": 0.9, "mid": 0.6, "old": 0.3}),  # monotonic
        compute_curve({"recent": 0.8, "mid": 0.5, "old": 0.2}),  # monotonic
        compute_curve({"recent": 0.3, "mid": 0.5, "old": 0.9}),  # not
    ]
    agg = aggregate_curves(curves)
    assert agg.n == 3
    assert agg.monotonicity_rate == pytest.approx(2 / 3)
    # mean delta = (0.6 + 0.6 + -0.6)/3 = 0.2
    assert agg.mean_delta_recent_old == pytest.approx(0.2)


# ===========================================================================
# sanity_metrics
# ===========================================================================


def test_repetition_n8_clean_text_is_low():
    text = " ".join(f"word{i}" for i in range(50))  # all distinct
    r = repetition_n8(text)
    assert r.value == 0.0


def test_repetition_n8_detects_loop():
    phrase = "the conversation is ongoing and continues forever without end please"
    text = (phrase + " ") * 10  # heavy repetition
    r = repetition_n8(text)
    assert r.value > 0.5
    assert len(r.matches) > 0


def test_repetition_n8_short_text_safe():
    r = repetition_n8("too short")
    assert r.value == 0.0


def test_bracketed_tag_format_detects():
    text = "Summary. [Status: resolved] [Priority: high] done."
    r = bracketed_tag_format(text)
    assert r.value == 2.0
    assert "[Status: resolved]" in r.matches


def test_bracketed_tag_format_ignores_plain_brackets():
    text = "See [1] and [2] for details."
    r = bracketed_tag_format(text)
    assert r.value == 0.0


def test_meta_leakage_detects():
    text = "The user is asking about redis. The conversation is ongoing."
    r = meta_leakage(text)
    assert r.value >= 2.0


def test_meta_leakage_clean():
    text = "Redis runs on port 5432. Use a connection pool."
    r = meta_leakage(text)
    assert r.value == 0.0


def test_surface_natural_end_positive():
    assert surface_natural_end("This ends properly.").value == 1.0
    assert surface_natural_end("Code block:\n```\nx=1\n```").value == 1.0


def test_surface_natural_end_negative():
    assert surface_natural_end("This trails off and").value == 0.0
    assert surface_natural_end("").value == 0.0


def test_all_sanity_metrics_keys():
    d = all_sanity_metrics("Some text. [Status: ok]")
    assert set(d.keys()) == {
        "repetition_n8",
        "bracketed_tag_format",
        "meta_leakage",
        "surface_natural_end",
    }


# ===========================================================================
# stats: clustered bootstrap
# ===========================================================================


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    values = list(rng.normal(0.7, 0.1, size=60))
    clusters = [f"conv{i // 3}" for i in range(60)]  # 20 clusters of 3
    ci = clustered_bootstrap_ci(values, clusters, n_boot=2000, seed=1)
    assert ci.lower <= ci.point <= ci.upper
    assert ci.n_clusters == 20
    # point should be near the true mean 0.7
    assert ci.point == pytest.approx(0.7, abs=0.1)


def test_bootstrap_ci_deterministic_with_seed():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    clusters = ["a", "a", "b", "b", "c", "c"]
    ci1 = clustered_bootstrap_ci(values, clusters, n_boot=500, seed=7)
    ci2 = clustered_bootstrap_ci(values, clusters, n_boot=500, seed=7)
    assert ci1.lower == ci2.lower
    assert ci1.upper == ci2.upper


def test_bootstrap_ci_mismatched_raises():
    with pytest.raises(ValueError):
        clustered_bootstrap_ci([1.0, 2.0], ["a"], n_boot=10)


# ===========================================================================
# stats: paired win/loss + McNemar
# ===========================================================================


def test_paired_win_loss_counts():
    a = {"c1": 0.9, "c2": 0.5, "c3": 0.7, "c4": 0.6}
    b = {"c1": 0.6, "c2": 0.8, "c3": 0.7, "c4": 0.4}
    wl = paired_win_loss(a, b)
    assert wl.a_wins == 2  # c1, c4
    assert wl.b_wins == 1  # c2
    assert wl.ties == 1  # c3


def test_mcnemar_exact_small():
    wl = paired_win_loss(
        {f"c{i}": 1.0 for i in range(10)},
        {f"c{i}": 0.0 for i in range(10)},
    )
    # a wins all 10, b wins 0 -> strongly significant
    res = mcnemar_test(wl)
    assert res.method == "exact-binomial"
    assert res.a_wins == 10 and res.b_wins == 0
    assert res.p_value < 0.01


def test_mcnemar_no_discordant():
    wl = paired_win_loss({"c1": 0.5}, {"c1": 0.5})
    res = mcnemar_test(wl)
    assert res.p_value == 1.0


def test_mcnemar_chi2_large():
    # 40 discordant pairs -> chi2 path
    a = {f"c{i}": (1.0 if i < 30 else 0.0) for i in range(40)}
    b = {f"c{i}": (0.0 if i < 30 else 1.0) for i in range(40)}
    wl = paired_win_loss(a, b)
    res = mcnemar_test(wl)
    assert res.method == "chi2-continuity"
    assert res.a_wins == 30 and res.b_wins == 10


# ===========================================================================
# stats: wilcoxon
# ===========================================================================


def test_wilcoxon_detects_shift():
    a = {f"c{i}": 0.8 for i in range(15)}
    b = {f"c{i}": 0.5 for i in range(15)}
    res = wilcoxon_paired(a, b)
    assert res.n_nonzero == 15
    assert res.p_value < 0.01


def test_wilcoxon_all_zero():
    a = {f"c{i}": 0.5 for i in range(5)}
    b = {f"c{i}": 0.5 for i in range(5)}
    res = wilcoxon_paired(a, b)
    assert res.p_value == 1.0
    assert res.n_nonzero == 0


# ===========================================================================
# stats: cohen's kappa
# ===========================================================================


def test_cohens_kappa_perfect_agreement():
    a = ["present", "false", "partial", "present"]
    b = ["present", "false", "partial", "present"]
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_cohens_kappa_chance_agreement_near_zero():
    # Construct raters that agree only at chance level
    rng = np.random.default_rng(0)
    labels = ["present", "false", "partial"]
    a = list(rng.choice(labels, size=300))
    b = list(rng.choice(labels, size=300))
    k = cohens_kappa(a, b)
    assert abs(k) < 0.15  # near zero


def test_cohens_kappa_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


# ===========================================================================
# stats: ICC
# ===========================================================================


def test_icc_high_agreement():
    # Two judges giving very similar scores across subjects
    ratings = [
        [4.0, 4.1],
        [2.0, 2.1],
        [5.0, 4.9],
        [3.0, 3.2],
        [1.0, 1.1],
    ]
    icc = icc21(ratings)
    assert icc > 0.9


def test_icc_low_agreement():
    # Judges uncorrelated
    ratings = [
        [1.0, 5.0],
        [5.0, 1.0],
        [2.0, 4.0],
        [4.0, 2.0],
        [3.0, 3.0],
    ]
    icc = icc21(ratings)
    assert icc < 0.3


def test_icc_needs_2x2():
    with pytest.raises(ValueError):
        icc21([[1.0, 2.0]])  # only 1 subject
