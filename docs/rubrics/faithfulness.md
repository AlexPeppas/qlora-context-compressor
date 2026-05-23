# M_faithfulness — Critical-Information Preservation Rubric

## What this measures

Whether the compression preserves the **critical information** from the source conversation. Two-stage methodology: the judge first extracts a list of critical items from the source, then checks whether each item is present in the compression.

Critical information is content whose loss would prevent a downstream consumer from continuing the conversation correctly:

- **Decisions** taken or proposed
- **Numerical values** (counts, prices, durations, version numbers, IDs)
- **Code blocks** and their key components (function names, signatures, logic)
- **Error messages** and **diagnostic output**
- **Named entities** (people, products, files, services, libraries) referenced
- **Constraints / requirements** stated by the user
- **Action items** or pending tasks

Non-critical information explicitly excluded:

- Filler phrases ("Hey there", "Thanks for the help")
- Repeated reassurances or politeness markers
- Detailed reasoning steps that lead to a stated decision (the *decision* is critical; the steps are not)
- Conversational color (jokes, asides, personal anecdotes that aren't part of the technical content)

## Two-stage process

### Stage 1: Critical-item extraction (from source only)

The judge is given **the source conversation only** and asked to extract a list of critical items. Each item is a short JSON object:

```json
{
  "id": 1,
  "type": "decision" | "number" | "code" | "error" | "entity" | "constraint" | "action",
  "summary": "<≤15 words describing the item>",
  "verbatim_indicator": "<a short anchor string from source that uniquely identifies the item, ≤30 chars>"
}
```

Each critical item must be:
- **Distinct** from other items (no near-duplicates)
- **Anchored** in the source (the verbatim_indicator must appear in the source text)
- **Short** (summary ≤ 15 words)

The judge produces a list of **5 to 25 items**. If fewer than 5 critical items exist (rare, very short conversations), the judge reports the actual count. If more than 25 candidates exist, the judge prioritizes by retention-importance for downstream conversation continuity.

### Stage 2: Coverage check (compression vs item list)

The judge is given **the item list AND the compression** (NOT the original source). For each item it returns:

```json
{
  "id": 1,
  "present": true | false | "partial",
  "evidence": "<≤30-char span from the compression supporting the call, or empty>"
}
```

Definitions:
- **`true`**: the item's content is faithfully represented in the compression. Paraphrase OK; the *meaning* must be preserved. A number is "present" only if the exact number (or an equivalent unit-converted form) appears.
- **`partial`**: the item is referenced but with a meaningful loss (a decision is mentioned but the rationale is gone; a code block is described in prose but the code itself is gone; a number is approximate).
- **`false`**: the item is missing or contradicted.

## Scoring

Compute the per-row score:

```
faithfulness_score = (# present + 0.5 * # partial) / # total_items
```

Range: 0.0 to 1.0, continuous. Higher is better.

## Why this design

- **Two-stage forces grounding**: the judge can't hallucinate "criticality" because it has to anchor each item to a source span
- **Continuous score, not 1-5 Likert**: more statistical power on N=108 rows; bootstrap CIs are well-defined
- **Paraphrase-tolerant**: doesn't penalize abstractive compression (which paraphrases by design); a literal-extractive baseline doesn't get a free pass either
- **Auditable**: the per-row JSON output (item list + coverage) is the paper appendix; reviewer can spot-check any row
- **Tier-agnostic**: same rubric applies across recent/mid/old. Tier-conditioning effects show up as curve shape (M_tier_appropriate), not as different rubrics per tier

## Edge cases and decisions

| Case | Decision |
|---|---|
| Compression contains content NOT in source (hallucination) | Out of scope of this rubric (handled by separate sanity metric `bracketed_tag_format` + meta_leakage detectors). M_faithfulness scores only retention, not fabrication. |
| Compression is empty | All items `false`, score = 0.0. |
| Source has fewer than 5 critical items | Stage 1 reports actual count; Stage 2 proceeds normally; scoring still ratio-based. |
| Compression contains the source verbatim (no compression) | All items `true`, score = 1.0. (This will happen for very low-ratio targets; we report achieved_ratio alongside the score.) |
| Item is partially correct but contradictory (e.g., correct number, wrong unit) | Score as `partial`. |
| Code block paraphrased into prose | Score as `partial` if the prose preserves function names + key parameters; `false` if the prose is so vague that the code can't be reconstructed. |

## Validation requirement

Author manually scores 30 random (source, compression, item-list) tuples after the Phase C pilot. Per-item precision/recall against author labels reported. If precision < 0.7 or recall < 0.7 on either Stage 1 or Stage 2, rubric is revised once before Phase E.
