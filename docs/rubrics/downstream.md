# M_downstream — Conversation Continuation Rubric

## What this measures

Whether the compression preserves enough information for a fresh assistant to **continue the conversation correctly** when given only the compression and the next user turn. Tests the actual use case our system is designed for: replacing earlier conversation turns with a compressed summary so the LLM can keep responding usefully.

## Setup

For each source conversation in the bake-off:

1. **Hold out the last assistant turn** before passing to compressors. The compressor only sees turns up to and including the second-to-last assistant turn (the last user turn is preserved as the "test query").
2. **Each baseline compresses** what they see (same input as faithfulness; the held-out content is invisible to all baselines).
3. **Continuation generation**: a fresh `gpt-5.5` is given:
   ```
   [compressed prior turns from baseline X]
   [user]: <last user turn from source>
   [assistant]:
   ```
   and generates a response. Same model + temperature=0 for all baselines so the only varying input is the compression itself.
4. **Judge scores** the continuation against the **ground-truth assistant turn** that was held out.

## Why hold out the LAST assistant turn (not last user turn)

Original design considered holding out the last user turn (so the assistant has to predict "what user asks next"). That's a harder, noisier task — there's often no single right next user turn. Holding out the assistant response is more constrained: there's a specific way the conversation actually went, recorded in the source.

## Rubric (per continuation)

Judge sees:

- The **ground-truth assistant turn** (held-out, what the source assistant actually said)
- The **generated continuation** (from a fresh GPT-4o given compression + last user turn)

Scores on three axes, each 1-5:

### A. Substantive correctness (1-5)
Does the generated continuation give *substantively the same* answer / advice / code / decision as the ground truth?

| Score | Meaning |
|---|---|
| 5 | Substantively identical answer; same conclusions, same key facts, same recommendations |
| 4 | Minor differences in detail or phrasing; same overall direction |
| 3 | Partially correct; some key points match, some are off-topic or incorrect |
| 2 | Mostly different conclusions; could mislead a reader |
| 1 | Completely wrong, off-topic, or contradicts the ground truth |

### B. Code / numeric fidelity (1-5)
If the ground truth contains code, numbers, error messages, or specific entity names, did the continuation reproduce or correctly reference them?

| Score | Meaning |
|---|---|
| 5 | All code/numbers/entities present and correct |
| 4 | All present but with minor formatting / paraphrase differences |
| 3 | Some present and correct; some missing or distorted |
| 2 | Most missing, distorted, or fabricated |
| 1 | All missing or all fabricated |
| N/A | Ground truth contains no code/numbers/entities (excluded from aggregate) |

### C. Conversation coherence (1-5)
Does the continuation acknowledge prior turns appropriately (no "what are you talking about?" responses, no contradictions with what should be in compressed context)?

| Score | Meaning |
|---|---|
| 5 | Smooth continuation; references prior content as if it had full context |
| 4 | Minor coherence gaps; reader could fill in |
| 3 | Some coherence breakdown; assistant asks for info that should be in compression |
| 2 | Major coherence gaps; assistant restates basics or contradicts prior turns |
| 1 | Complete coherence breakdown; reads as a fresh conversation, not a continuation |

## Aggregate score

For each row, the per-row score is the **mean of A, B (if applicable), C**, on a 1-5 scale.

Headline aggregate per (system, tier): mean of per-row scores. CIs via clustered bootstrap over conversation_id.

## Why three axes instead of one

A single Likert score conflates substance, fidelity, and coherence — three semantically distinct properties. A compression could preserve all the facts but produce an incoherent continuation (low coherence, high fidelity), or be coherent but factually wrong (high coherence, low correctness). The three-axis decomposition lets us see which dimension drives any quality gap, and it's a standard pattern in summarization evaluation literature.

## Edge cases and decisions

| Case | Decision |
|---|---|
| Source has only one assistant turn total | Skip M_downstream for that row (no holdout possible). Excluded from M_downstream aggregate. |
| Ground truth assistant turn is very short / one-line ("OK") | Skip; uninformative. |
| Generated continuation is empty (model failed) | All axes = 1. |
| Generated continuation refuses to answer ("I need more context") | A = 2 (could mislead via abstention), C = 1 (acknowledges the gap, so coherence is broken). B = N/A. |
| Multiple equally-valid ways to continue exist | Judge scores against the SPECIFIC ground truth, not against all valid continuations. We acknowledge in the paper that this slightly under-rewards diverse-but-correct responses. |

## Notes on baseline parity

All baselines feed compression to the **same** continuation generator (`gpt-5.5`, temperature=0, pinned snapshot). This isolates the compression quality from the continuation-model quality. Without this, a baseline's downstream score would conflate "is the compression good" with "is the continuation generator good".

## Validation requirement

Author manually scores 30 random continuations after Phase C pilot, blinded to system identity. Per-axis correlation with judge scores reported (Spearman). If correlation < 0.4 on any axis, that axis is revised.
