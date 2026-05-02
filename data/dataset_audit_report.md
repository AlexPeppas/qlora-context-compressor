# Dataset Quality Audit: synthetic_dataset.jsonl
**150 examples | 50 unique conversations × 3 ratio tiers**
**Audit date: 2026-04-27**

---

## Executive Summary

**Do not use this dataset for Stage 1 SFT as-is.** The ratio targeting is severely miscalibrated across all three tiers — only 41 of 150 examples (~27%) fall within ±30% of their target ratio. Every single example over-compresses; none under-compress. The dataset has only 50 unique source conversations, several scenarios recycle near-identical topics, and `turn_age` is a confounded variable that adds no independent signal. There are confirmed hallucinated facts in at least 3 examples, and anchors are copy-pasted identically across all three compression tiers for every conversation group. Compression fidelity is generally decent; tool stub formatting is clean. The problems are structural (ratio calibration, diversity, label design) and require generation-prompt changes before the 1,000-example run, not just post-hoc filtering.

---

## 1. Ratio Adherence — Critical Failure

### Measured actual ratios (whitespace-token counts)

| Target | Mean Actual | Median | Stdev | Min | Max | Within ±30% | Within ±60% |
|--------|------------|--------|-------|-----|-----|-------------|-------------|
| 3×     | **4.7×**   | 4.7×   | 0.97  | 2.4× | 7.2× | 14/50 (28%) | 26/50 (52%) |
| 5×     | **8.1×**   | 7.9×   | 2.28  | 4.3× | 13.1× | 16/50 (32%) | 26/50 (52%) |
| 10×    | **17.3×**  | 18.1×  | 4.37  | 9.5× | 28.0× | 11/50 (22%) | 21/50 (42%) |

**Every tier is off by roughly 1.7–1.8× in the over-compressed direction.** There is not a single example that under-compresses (actual ratio < 0.5× target). The model has a consistent, systematic bias toward compressing too aggressively.

### The worst offenders

38 of 150 examples (25%) have an actual ratio more than 2× their target:

- `48af941b` coding 10× target → **28.0× actual** (981 tokens → 35 tokens)
- `8d43ae1f` support 10× target → **25.3× actual**
- `95a0c6e9` support 10× target → **24.2× actual**
- `d48760a3` support 10× target → **23.8× actual**
- `aa529d40` support 5× target → **13.1× actual** (a 5× example that's actually 13×)
- `ea5931f3` support 3× target → **7.2× actual** (a 3× example that's actually 7×)

The worst 10× compression is:

> *Original (981 tokens):* Full debugging conversation about FastAPI serialization and pagination
>
> *"10×" output (35 tokens):* `FastAPI /api/v1/orders 500 for user_id=4821 caused by unserializable Decimal fields (order_total, discount_applied). Fixed with Pydantic v2 response_model using from_attributes=True. New issue: 800ms-1.2s slowness; user 7293 has 3,400 orders timing out — needs indexing and pagination.`

At 28× actual compression, this is a lossless headline, not a 10× compressed memory. The model would learn that "10×" means "reduce to a 2-sentence abstract."

### Breakdown by scenario type

| Scenario | 3× mean | 5× mean | 10× mean |
|----------|---------|---------|----------|
| coding   | 5.4×    | 9.6×    | **19.8×** |
| research | 3.3×    | 5.7×    | 12.9×   |
| support  | 6.2×    | 10.6×   | **21.7×** |
| tool_heavy | 3.7×  | 6.1×    | 13.9×   |
| analysis | 5.2×    | 8.6×    | 18.1×   |

Research and tool_heavy are the closest to target. Coding and support are badly off at every tier — a 3× coding example averages 5.4× actual. Support 10× averages 21.7×. **The tiers barely separate from each other for coding and support**: coding 3× (5.4×) and coding 5× (9.6×) are closer to each other than either is to its target.

---

## 2. Dataset Structure: turn_age is Confounded

`turn_age` is 100% perfectly correlated with `target_ratio`:

| turn_age | ratio | n |
|----------|-------|---|
| recent   | 3.0×  | 50 |
| mid      | 5.0×  | 50 |
| old      | 10.0× | 50 |

The same 752-word conversation is labeled `turn_age=recent` for its 3× variant, `turn_age=mid` for its 5× variant, and `turn_age=old` for its 10× variant — despite the original being identical text. The `turn_age` field carries no independent information; it is an alias for the compression tier. If the model learns from this, it learns "old conversations get compressed 10×" not "old context that is less likely to be needed gets compressed more aggressively." This is a label design bug.

---

## 3. Anchor Quality — Identical Across All Tiers

Every single one of the 50 conversation groups has **identical anchor lists** at all three compression tiers (50/50 confirmed). The same 8–10 bullet points are copy-pasted regardless of whether the compressed output is a 400-token prose summary or a 40-token telegram.

This creates two problems:

**Problem A — Training signal inconsistency.** The model will observe that a 35-token compressed output has 9 dense anchor facts, while a 400-token output has the same 9 anchor facts. The anchors don't scale with or respond to the compression tier, so they can't teach the model what information is "load-bearing" at each tier.

**Problem B — Anchor count is suspiciously uniform.** Every example has 8–10 anchors (distribution: 8→66 examples, 9→54, 10→30). No example has fewer than 8 or more than 10. This suggests the generator is hard-constrained to produce exactly N anchors rather than deriving them organically from the content.

### One confirmed anchor hallucination

`2e92b9e8` (analysis 3×) and its sibling tiers `f399ef67` (5×) and `04dc18b1` (10×) all include this anchor:

> `'6 slipped deals = $420K; 4 active ($325K), 2 dead ($95K)'`

The original states: *"Six deals totaling $420K slipped"* and *"4 of the 6 are still active"* and *"the other 2 are effectively dead ($95K combined)."* The $95K appears in the original. The **$325K does not** — it is computed by subtraction ($420K − $95K = $325K) and silently asserted as a stated fact. A model trained on this anchor learns to present derived figures as retrieved facts.

Other apparent mismatches ($1.2M, $2,987, $3,400) turned out to be format differences (e.g., `1200000` → `$1.2M`, `3,400` → `3400`) rather than hallucinations.

---

## 4. Scenario Diversity — Two Scenarios Are Broken

### Research: 9/10 conversations are about sleep deprivation

```
1.  "I'm researching the long-term cognitive effects of ultra-processed food..."   ← only non-sleep topic
2.  "I'm researching the link between chronic sleep deprivation and metabolic syndrome."
3.  "I'm researching the relationship between sleep deprivation and cognitive performance in shift workers..."
4.  "I'm looking into the relationship between gut microbiome diversity and cognitive decline..."  ← only other non-sleep
5.  "I'm researching the relationship between sleep deprivation and cognitive decline in older adults."
6.  "I'm researching the relationship between sleep deprivation and neuroinflammation..."
7.  "I'm researching the cognitive effects of sleep deprivation on decision-making in high-stakes professions."
8.  "I'm researching the relationship between sleep deprivation and neuroinflammation..."  ← near-duplicate of 6
9.  "I'm researching the relationship between sleep deprivation and working memory capacity."
10. "I'm researching the relationship between sleep deprivation and cognitive performance in adolescents."
```

Examples 6 and 8 are near-duplicates (similarity score 0.76 on first 300 chars): both ask about sleep deprivation and neuroinflammation with a focus on microglial activation. They diverge slightly in how the follow-up questions develop, but the topic, framing, and opening are essentially the same conversation generated twice.

Examples 5, 6, 8, and 10 all share similarity > 0.5 with each other. Of 10 research conversations, there is effectively 1 domain cluster (sleep/cognition) with minor subtype variation, 1 UPF conversation, and 1 gut microbiome conversation. This is not a research scenario — it's a sleep science scenario with two outliers.

### Tool-heavy: 8/10 conversations involve "Meridian"

8 of 10 tool_heavy originals involve a company called "Meridian" (either Meridian Healthcare Solutions or Meridian Logistics). The remaining 2 also involve account lookup workflows. No tool_heavy conversation involves a meaningfully different use case (e.g., code execution tools, search tools, calendar/scheduling agents, data analysis pipelines). Every conversation follows the pattern: "pull account X, check status Y, draft communication Z."

### Entity repetition across all scenarios

Counting character names across the full 50-conversation set:

- "Meridian" appears in **85** conversation contexts
- "Marcus" in **70**, "Priya" in **42**, "Rachel" in **31**

The dataset has essentially 4–5 recurring characters and 1 recurring company. A model trained on this will have overfit entity priors.

### Coding: 3 near-duplicate conversations

Of 10 coding originals, three (examples 1, 3, 8 — ids `906ecaf1`, `b830036b`, `93bdefbb`) share similarity scores of 0.75–0.88. All three are React `useDebounce` infinite re-render bugs caused by passing an object literal where a primitive was expected. They differ in minor surface details (variable names, follow-up questions) but teach the same debugging pattern. With 5/10 coding conversations involving React hooks and 5/10 involving FastAPI, the coding scenario has collapsed to two templates.

### Analysis: all 10 conversations are Q3 2024 SaaS revenue analysis

Every analysis conversation presents some variant of a SaaS company that missed Q3 2024 revenue targets (usually $4.2M actual vs a ~$5.1M target). The figures and specific segments vary, but the scenario type, the framing ("I need you to analyze our Q3 2024 performance"), and the analytical structure are monotonously similar.

---

## 5. Compression Fidelity — Mostly Acceptable

The compressed outputs generally preserve named entities (account IDs, ticket numbers, person names), specific numbers, and key decisions. For a 3× compression, the outputs read as plausible dense summaries. For 10× compressions that actually hit ~10× (the minority), they function as useful telegrams.

Selected quality examples that work well:

**research 5× (actual 4.9×):** Correctly preserves study names (JAMA Pediatrics 2022, n=3,200), effect sizes (14% WM reduction, d=0.31), mechanism pathway (SCFA → butyrate → LPS → IL-1β → BDNF), and specific pilot results (Shannon +0.41, d=0.28). No hallucinations detected.

**tool_heavy 3× (actual ~3.7×):** Correctly preserves account IDs (ACC-20847), ARR ($184K), health score (67/100), ticket numbers (TKT-91204), competitive signals (Veeva, Salesforce Health Cloud), and CSM name (Priya Nandakumar). High fidelity.

**support 10× at ~24×:** The over-compression causes qualitative loss — a conversation with specific resolution steps, ticket escalation chains, and customer history is reduced to a 57-token headline that drops the resolution path entirely, leaving only the "what happened" not the "how it was fixed."

---

## 6. Tool Stub Quality — Good

All 90 tool stubs (across 30 tool_heavy examples) pass format validation against the pattern `[Tool: X @ turn N — summary]`. Turn numbers are consistent with the original conversation structure. Summaries accurately represent the tool result content. The stubs correctly distinguish between different tool calls (crm_lookup vs crm_get_notes vs support_ticket_lookup, etc.) and carry the right factual load.

No format violations found.

---

## 7. Training Signal Concerns

**A — The 3× tier is effectively absent.** The mean actual ratio for 3× examples is 4.7×, and no example achieves true 3× compression (only 2 examples are below 3.5×, both in research). The model would learn that "3×" means "compress by about half" rather than "retain most structure and detail, reduce by 2/3." This erases the distinction between the 3× and 5× tiers (actual means: 4.7× vs 8.1×).

**B — The 10× tier teaches over-compression.** 43 of 50 examples labeled "10×" actually achieve >12× compression. 29 achieve >16×. The model would learn to compress "old" context to near-nothingness even when the target is a meaningful 10× reduction.

**C — Anchors cannot teach tier-calibration.** Because anchors are identical across all three tiers, they provide no signal about what gets preserved at each compression level. The model cannot learn from this data that some facts should drop out at 10× that survive at 3×.

**D — The output style is inconsistent across tiers and scenarios.** Coding 3× outputs often start with "User had…" or "USER has…" (third-person case report). Research 10× outputs vary between numbered-study enumeration and dense prose. Analysis outputs start with "Q3" telegrams. This stylistic variance is fine if intentional, but it's not clearly tied to scenario type or tier in a systematic way.

---

## 8. Dataset Balance — Structurally Sound, Topically Skewed

By scenario type: 30 examples each — exactly balanced.
By target ratio: 50 examples each — exactly balanced.
By turn_age: 50 each — but this is not an independent dimension (see §2).

The structural balance is fine. The topical skew is the problem: within research (9/10 sleep deprivation) and analysis (10/10 Q3 SaaS revenue), the model sees the same domain repeatedly with only surface variation.

---

## Verdict and Recommendations

**The dataset is not ready for Stage 1 SFT.** The generation prompt needs to be revised before the 1,000-example run.

### Must-fix before generation run

1. **Ratio calibration is the blocking issue.** The generation prompt must enforce token budgets explicitly. Calculate target token count from original length and pass it as a hard constraint: `"Compress to approximately N words (target: {original_tokens / target_ratio:.0f} words)."` Post-generation, filter any example where actual ratio deviates more than 40% from target in either direction. Current pass rate at ±40%: ~35%.

2. **Research scenario must be diversified away from sleep deprivation.** Only 1–2 of 10 conversations per scenario should share a topic cluster. Candidates: climate science, economics, legal research, historical analysis, medical literature (non-neuroscience), policy research.

3. **Tool-heavy must be diversified beyond Meridian account lookups.** Use different company types, different tool categories (code execution, search, calendar, document retrieval, API calls), and different user roles.

4. **Decouple `turn_age` from `target_ratio`.** Either assign turn_age independently (e.g., a recent 10× scenario is a very long conversation where the early turns need heavy compression), or drop `turn_age` as a training label entirely. Currently it adds zero information beyond what `target_ratio` already encodes.

5. **Deduplicate the near-identical conversations.** Remove or replace the duplicate useDebounce pairs (examples 1/3/8 in coding) and the neuroinflammation pair (examples 6/8 in research).

### Should-fix

6. **Make anchors tier-sensitive.** Either generate separate anchor lists per tier (reflecting what a model at that compression level would need to verify) or accept that anchors represent the full conversation, not the compressed version — and document this explicitly.

7. **Fix the $325K anchor hallucination** and audit all derived numbers systematically. Any anchor value not literally present in the original text should be either removed or explicitly flagged as `[derived]`.

8. **Constrain entity vocabulary.** Mandate that each generated conversation use unique company/person names to avoid Meridian/Marcus/Priya overfit.

### Acceptable as-is

- Tool stub formatting is clean and accurate across all 30 tool_heavy examples.
- Compression fidelity (where ratios are not wildly off) is good — no systematic hallucinations detected beyond the $325K case.
- The individual conversations are realistic and read like genuine human-AI exchanges (not formulaic or obviously templated at the turn level).
- Scenario type distinctiveness is maintained: coding, support, tool_heavy, analysis, and research are clearly distinguishable in style and content.

If you filter to only examples within ±40% of their target ratio, approximately 50–55 examples survive. That is not enough for Stage 1 SFT. The generation needs to be rerun with the calibration fix, not just filtered.
