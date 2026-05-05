# EOS supervision bug: discovery and fix

## Summary

A one-line cargo-cult idiom in the trainer (`tokeniser.pad_token = tokeniser.eos_token`) silently disabled EOS supervision for the QLoRA fine-tunes of Qwen2.5-7B-Instruct, causing four distinct failure modes in the resulting adapters: repetition collapse, hallucinated bracketed structural tags, meta-leakage, and budget non-adherence. Removing the line resolved all four simultaneously, with no other changes to dataset, hyperparameters, or seed.

## The bug

```python
# train_lora.py, before fix:
tokeniser.pad_token = tokeniser.eos_token
```

This idiom is common in Llama-2-era fine-tuning recipes where the base tokenizer ships without a pad token. Qwen2.5-Instruct, however, defines them as **distinct tokens by default**:

| Token | String | ID |
|-------|--------|-----|
| `pad_token` | `<\|endoftext\|>` | 151643 |
| `eos_token` | `<\|im_end\|>` | 151645 |

After the assignment, both tokens share id 151645. TRL's `SFTTrainer` constructs labels by replacing pad-token ids with `-100` (the cross-entropy ignore index). With `pad_id == eos_id`, **every legitimate end-of-turn `<|im_end|>` in the assistant span gets masked from the loss**. The model sees no gradient signal teaching it to predict EOS.

## Pre-fix evidence (Phase -1 audit)

Bake-off natural sentence-end rate (proportion of generations that terminate with sentence-final punctuation, used as a proxy for healthy EOS emission):

| Source | Natural-end rate |
|--------|-----------------|
| Base Qwen2.5-7B-Instruct (no fine-tune) | 16/18 (89%) |
| cp249 (broken adapter, 2 epochs) | 2/18 (11%) |
| cp372 (broken adapter, 3 epochs) | 1/18 (6%) |

On the `old`-tier (cap=200 tokens), 0/6 generations from each adapter ended naturally. The model had **zero functioning EOS** within the budget envelope.

Source-data sanity check: 988/999 (98.9%) of training compressions end with sentence terminators. The dataset is clean; only the masking is broken.

## The fix

```python
# train_lora.py, after fix:
assert tokeniser.pad_token_id is not None, (
    "Tokeniser has no pad token id"
)
assert tokeniser.pad_token_id != tokeniser.eos_token_id, (
    "pad_token_id == eos_token_id; SFTTrainer would mask every EOS from labels"
)
logger.info(
    "Tokeniser pad/eos distinct: pad=%r (id=%d), eos=%r (id=%d)",
    tokeniser.pad_token, tokeniser.pad_token_id,
    tokeniser.eos_token, tokeniser.eos_token_id,
)
```

Two assertions act as guards against re-introducing the bug. The log line provides positive runtime confirmation.

## Post-fix evidence (Phase 0c bake-off)

Same 6 OOD conversations × 3 turn-age tiers × 3 sources = 54 generations. Same prompts, hyperparameters, and seed as the broken run.

| Metric | Broken (cp249/cp372) | Fixed (tfix250/tfix375) | Δ |
|---|---|---|---|
| Natural sentence-end rate | 0–17% | 83–100% | +~85 pp |
| Meta-leakage on `recent` | 67–83% | 0% | −~75 pp |
| Hallucinated `[Tag: …]` on `recent` | 17–33% | 0% | −~25 pp |
| Repetition collapse | 1–6% | 0% | −100% relative |
| Recent-tier ratio (target 3×) | 1.7–1.8× | **3.5–4.1×** | hits target |
| Mid-tier ratio (target 5×) | 3.2× | **5.8–6.3×** | hits target |
| Old-tier ratio (target 10×) | 7.0–7.3× | **10.9–11.2×** | hits target |
| Recent-tier gen time | 53 s (cap-bound) | 9–11 s | 5× faster |
| Total bake-off wall time | 22 min | 5.5 min | 4× faster |

Failure-mode definitions (heuristic detectors used here, to be formalized in Phase 0.5):

- **Natural sentence-end rate** — output (after `rstrip`) ends with `[.!?\")\]\}\``].
- **Meta-leakage** — output contains tokens that describe the assistant rather than compress the content (regex over `\\b(assistant is|user is asking|the assistant|the user (asks|is)|conversation is ongoing)\\b`, case-insensitive).
- **Hallucinated bracketed tags** — output contains regex `\\[[A-Z][A-Za-z ]+:\\s*[^\\]]+\\]`, capturing fabricated structural markers like `[Status: ...]`, `[Priority: ...]`.
- **Repetition collapse** — fraction of duplicate 8-grams in output, max across rows reported.

Wall-time speedup is itself a passive signal: with EOS broken, every generation hit `max_new_tokens` (700/400/200 per tier); with EOS healthy, generations stop naturally well below the cap.

## Pre-registered prediction

Before running the post-fix bake-off, the following predictions were locked in:

1. Natural-end rate jumps from ~10% to ~80%+. **Confirmed** (83–100%).
2. Repetition collapse on `recent` tier drops to near zero. **Confirmed** (0%).
3. No more "Conversation is ongoing..." loops. **Confirmed**.
4. No more mid-word truncation. **Confirmed** (implied by ~100% natural-end rate).
5. Achieved ratios move closer to targets. **Confirmed** (target hit on all three tiers, slight overshoot on `mid` and `recent` for tfix250).

## Differential dose-response

`tfix375` (3 epochs) outperforms `tfix250` (2 epochs) on every measured dimension:

| Tier | tfix250 → tfix375 | Direction |
|------|-------------------|-----------|
| Recent natural-end | 83% → 100% | better |
| Recent ratio (target 3) | 4.06 → 3.50 | closer to target |
| Mid ratio (target 5) | 6.30 → 5.81 | closer to target |
| Old ratio (target 10) | 11.20 → 10.93 | closer to target |

Monotonic improvement with additional training rules out the "underfit" alternative explanation: the broken adapters were not undertrained, they were trained on a loss that excluded EOS.

## Causal isolation

The fix changes exactly one variable. Every other element of the pipeline is held constant:

- Same training corpus (`data/synthetic_dataset.jsonl`, byte-identical)
- Same hyperparameters (rank=32, alpha=64, lr=2e-4, cosine schedule, 3 epochs)
- Same seed (42)
- Same model and quantization (Qwen2.5-7B-Instruct, NF4)
- Same inference harness (greedy, identical per-tier `max_new_tokens` caps, identical bake-off conversations)
- Same evaluation script

The base column in the post-fix bake-off is byte-identical to the pre-fix run (greedy decoding is deterministic), confirming no environmental drift.

## What remains to validate

- **Phase 0d (decoding ablation):** show that no decoding strategy (greedy / sampling / repetition-penalty) applied to the broken adapters can recover the failure-mode resolution that the EOS fix produced. Rules out the "you just had bad decoding" alternative.
- **Phase 0.5 (formal metrics):** replace the heuristic detectors used here with a documented, unit-tested evaluation library.
- **Phase 2 (crossed-generator dataset):** confirm the bug-fix benefit generalizes beyond a Claude-generated training set.

## Artifacts

- `train_lora.py` (commit `f3abb62`): the EOS fix.
- `data/bakeoff_results.jsonl` / `.md`: pre-fix bake-off (broken adapters, 22 min).
- `data/bakeoff_results_eosfix.jsonl` / `.md`: post-fix bake-off (5.5 min).
- `data/training_eosfix.log`: post-fix training log (375 steps, 3 epochs, final loss 0.33).
- `data/bakeoff_eosfix.log`: post-fix bake-off log.
- `checkpoints/qwen2.5-7b-compressor/`: broken baseline adapters (preserved for paper).
- `checkpoints/qwen2.5-7b-compressor-eosfix/`: fixed adapters.
