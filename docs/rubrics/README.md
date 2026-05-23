# Pre-registered evaluation rubrics — Tiered Progressive Compression

> **Pre-registration statement.** The rubrics in this directory were locked at git commit time prior to running any Phase C pilot or Phase E head-to-head bake-off. Any subsequent modification must be (a) committed in a clearly-labeled "post-hoc" amendment file with rationale, and (b) results re-computed with both rubric versions reported.

## Methodology overview

Each compression output is graded against a documented rubric by an LLM judge. Three rubrics, one per evaluation metric:

| Metric | Rubric file | What it measures |
|---|---|---|
| `M_faithfulness` | `faithfulness.md` | Does the compression preserve critical information from the source? Uses two-stage extract-then-check methodology. |
| `M_downstream` | `downstream.md` | Given only the compression and the next user turn, can a fresh assistant produce a response close to the ground-truth assistant response? Continuation task. |
| `M_tier_appropriate` | `tier_appropriate.md` | Does retention of critical information drop appropriately across tiers (recent/mid/old)? Curve-shape metric, piggybacks on M_faithfulness's extraction step. |

## Judges

**Two independent LLM judges**, both apply each rubric independently. Headline numbers report the GPT-5.5 judge. Claude Sonnet 4.6 judge is reported alongside as a robustness check, plus inter-judge agreement (Cohen's kappa for binary calls; intraclass correlation for ordinal scores).

| Judge | Model ID (planned) | Role |
|---|---|---|
| Primary | `gpt-5.5-<snapshot>` (pin specific snapshot at experiment time) | Headline scores |
| Secondary | `claude-sonnet-4-6-<snapshot>` (pin specific snapshot at experiment time) | Robustness check + agreement |

**Why latest-generation judges, not GPT-4o?** Judge quality is the dominant lever on result validity — a weaker judge introduces noise that can't be recovered downstream. Reproducibility is preserved by pinning snapshot IDs at experiment time.

**Conflict-of-interest note.** GPT-5.5 is also one of our baselines (frontier ceiling). To avoid the "judge favors its own outputs" critique, the secondary Claude Sonnet 4.6 judge is reported with equal prominence in the paper. If GPT-5.5 judge and Claude judge disagree on the relative ordering of baselines, that is an explicit reportable finding, not a problem to suppress.

## Decoding settings for judges

- `temperature=0.0` for reproducibility
- `seed=42` (where supported)
- Structured output: JSON schema enforced (Pydantic / OpenAI structured outputs / Claude tool-use), so judges can't return malformed responses

## Blinding

For the M_faithfulness and M_downstream rubrics, when a judge is asked to score a (source, compression) pair, the system identity (which baseline produced it) is **not** revealed to the judge. The judge sees only:

```
Source conversation: <text>
Compressed version: <text>
```

System identity is recorded in the result file only after scoring is complete.

## Inter-judge agreement validation

After the Phase C pilot run, before Phase D iteration, we compute inter-judge agreement on the 108 rows. Reportable thresholds:

- **Acceptable** (proceed to Phase D): kappa ≥ 0.4 on binary calls, ICC ≥ 0.5 on ordinal scores
- **Marginal** (acceptable but flagged in paper as a limitation): kappa 0.2-0.4 / ICC 0.3-0.5
- **Unacceptable** (rubric needs redesign before paper): kappa < 0.2 / ICC < 0.3

If we land in the **unacceptable** zone we treat it as a design failure: the rubrics are not measuring what we want, and we revise *before* running the larger Phase E. This is the only sanctioned post-hoc revision.

## Human validation

Inter-judge agreement is necessary but not sufficient — two LLMs can agree on the wrong thing. The strongest paper version includes single-annotator (the author) human validation on a small subset (~50 rows balanced across baselines and tiers). Per-judge precision/recall vs human labels are reported as a calibration appendix.

This validation is treated as **post-pilot but pre-Phase-E**. The schedule:

1. Phase C pilot bake-off → judge scores produced
2. Author labels 50 random rows blinded
3. Compute judge-vs-human agreement
4. If judges look reliable, proceed to Phase E
5. If judges look unreliable on a specific metric, rubric is revised (one allowed revision)

## Files in this directory

- `faithfulness.md` — M_faithfulness rubric
- `downstream.md` — M_downstream rubric
- `tier_appropriate.md` — M_tier_appropriate rubric
- `judge_prompts.md` — full judge prompt templates (the actual strings sent to the API)
