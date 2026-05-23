# M_tier_appropriate — Information-Density Curve Rubric

## What this measures

Whether retention of critical information **drops appropriately across tiers**, matching the tier semantics our paper defines:

- `recent`: should retain near-everything (target ratio 3×, large budget)
- `mid`: should retain main thread + decisions, drop elaboration (target ratio 5×)
- `old`: should retain only outcomes, drop intermediate steps (target ratio 10×, small budget)

A **tier-conditioned** compressor should produce a steep monotonic decrease in retention across tiers. A **tier-blind** compressor (Cmprsr, LLMLingua-2 — none of these read the turn-age tier) should produce a flatter curve, achieving its compression ratio by uniform dropping rather than tier-aware selection.

The rubric measures the **shape of the retention curve**, not absolute numbers. This avoids the methodological trap of pre-registering target retention percentages (which would be tuning the rubric to favor our model).

## Inputs

Piggybacks on M_faithfulness:

- The same per-row item list (Stage 1 of M_faithfulness) is reused
- The same per-row coverage check (Stage 2) is reused
- Per-row faithfulness score is denoted `f(conv, system, tier)`

## Per-conversation curve

For each (conversation, system) pair, we have three faithfulness scores — one per tier:

```
f_recent = f(conv, system, recent)
f_mid    = f(conv, system, mid)
f_old    = f(conv, system, old)
```

The retention curve is the triple `(f_recent, f_mid, f_old)`.

## Curve-shape statistics

For each (conversation, system), compute:

### Curve metrics

| Metric | Definition | Tier-conditioned compressor expectation |
|---|---|---|
| **Δ_recent_old** | `f_recent − f_old` | Large positive (e.g., > 0.4) |
| **Δ_recent_mid** | `f_recent − f_mid` | Small-to-moderate positive |
| **Δ_mid_old** | `f_mid − f_old` | Moderate positive |
| **Monotonicity** | Boolean: `f_recent ≥ f_mid ≥ f_old` (allowing ties within ε=0.05) | Should be `True` for tier-conditioned |
| **Curve-AUC** | Trapezoidal area under (`f_recent`, `f_mid`, `f_old`) | Higher is better — measures overall quality across tiers |

### Per-system aggregation

Aggregate across the conversation set:

- Mean `Δ_recent_old` per system → headline tier-spread metric
- Fraction of conversations where monotonicity holds per system → robustness
- Bootstrap CIs on `Δ_recent_old` clustered by conversation_id

## What "winning on M_tier_appropriate" looks like

A tier-conditioned compressor wins by:

1. **Larger Δ_recent_old** than tier-blind baselines — i.e., it actually behaves differently at different tiers
2. **Higher monotonicity rate** — e.g., >85% of conversations satisfy `f_recent ≥ f_mid ≥ f_old` for tier-conditioned, vs ~50% (chance) for tier-blind
3. **Higher Curve-AUC** at matched compression ratios — tier-conditioning lets you keep more of what matters at each tier

If our model has small Δ_recent_old (< 0.2), the paper claim is weakened: tier-conditioning isn't doing much. If tier-blind baselines have similar Δ_recent_old (also large), the claim is also weakened: turn-age conditioning isn't novel because LLMLingua-2 happens to produce similar curves.

## Why curve-shape, not absolute numbers

Two failure modes if we used absolute target percentages:
1. **Reviewer attack**: "you set target=95% for recent because you knew your model would hit it." Even if untrue, it's hard to defend.
2. **Crisis on bad ratios**: if the compressor undershoots target ratio (compresses to 1.5× when target was 3×), absolute retention may be 99% across all tiers. The curve has wrong *shape* (flat) even though absolute retention is high.

Using `Δ_recent_old` and monotonicity rate sidesteps both. The metric *only* fires when the compressor actually treats tiers differently. A baseline that ignores tier produces near-zero `Δ_recent_old` regardless of its absolute quality.

## Edge cases and decisions

| Case | Decision |
|---|---|
| Compressor outputs nearly-identical text across tiers | Δ values near 0; monotonicity rate ~ 50% (random); curve-AUC reflects single quality value. This is the expected pattern for tier-blind baselines. |
| Compressor reverses the curve (`f_recent < f_old`) | Negative Δ; monotonicity = False. Pathological but reportable. |
| Compressor crashes on one tier | Excluded from per-row aggregation; row reported as missing. |
| Faithfulness scores are all very low across all tiers | Δ values still computable; curve-AUC near zero. We'd report this as "compressor is bad on this conversation" without it affecting tier-appropriateness call. |

## Validation requirement

This metric is mechanically derived from M_faithfulness; if M_faithfulness validates (per its own validation requirement), M_tier_appropriate is derived correctly by construction. No separate human-labeling round needed.

## Reporting in the paper

For each system, report:
- Mean `Δ_recent_old` ± clustered bootstrap 95% CI
- Monotonicity rate (% of conversations with `f_recent ≥ f_mid ≥ f_old`)
- Curve-AUC mean ± CI
- Visualization: per-system retention curve (3 points: recent/mid/old, with conversation-level error bars)
