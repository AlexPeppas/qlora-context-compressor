# Tiered Progressive Context Compression with QLoRA

A turn-age-conditioned context compressor for long-running LLM conversations.
The learned compressor is a QLoRA adapter for
[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) that
targets three retention tiers:

| Turn age | Target ratio | Intended retention |
|---|---:|---|
| `recent` | 3x | Preserve facts, reasoning, code, errors, and numeric values |
| `mid` | 5x | Preserve the main thread, decisions, entities, and unresolved issues |
| `old` | 10x | Preserve outcomes, constraints, dependencies, and still-relevant entities |

The repository contains both:

- a CPU-only progressive runtime using the existing heuristic compressor; and
- the CUDA training, baseline, public-corpus, judging, and statistics stack
  used to evaluate the learned adapter.

The research claim is **not yet established**. The public-data pilot is in
progress; this README distinguishes implemented methodology from pending
results.

## Current status

As of August 2026:

- **Runtime stack:** complete and CPU-compatible.
- **Training data:** 999 synthetic teacher-distilled examples.
- **QLoRA training:** complete; the adapter under evaluation is `tfix375`.
- **EOS supervision fix:** complete and retrained. The original trainer
  accidentally made padding and Qwen's `<|im_end|>` token identical, masking
  EOS labels and causing repetition/truncation pathologies.
- **Public benchmark ingestion:** complete for WildChat, OASST2, UltraChat,
  and MT-Bench adapters.
- **Baseline matrix:** complete for prompted Qwen, LLMLingua-2,
  LongLLMLingua, GPT-5.4, GPT-4o-mini, and our LoRA adapter.
- **Evaluation harness:** complete for faithfulness, downstream continuation,
  tier retention, deterministic sanity metrics, clustered bootstrap,
  McNemar, Wilcoxon, Cohen's kappa, and ICC(2,1).
- **Pod pipeline:** complete, resumable, revision-pinned, and provenance
  tracked.
- **Phase C public pilot:** compression is complete on 9 conversations;
  ensemble judging is in progress.
- **Runtime `LoraCompressor`:** not yet wired into the production compressor
  interface.

No headline quality result should be inferred until pilot judging, agreement
checks, and the larger public benchmark are complete.

## Research questions

The experiment tests whether tier conditioning provides a useful retention
curve that a single generic compression prompt or tier-blind extractive method
does not:

1. Does the adapter preserve critical information at matched target ratios?
2. Can a fixed continuation model answer the next user turn using only the
   compressed prior context?
3. Does retained information decrease appropriately from recent to mid to old?
4. Does a model trained entirely on synthetic conversations generalize to
   public, out-of-distribution conversations?

The EOS bug fix is supporting methodology, not the paper's headline
contribution.

## Training provenance

### Dataset

`data/synthetic_dataset.jsonl` contains 999 fully synthetic examples generated
with Claude Sonnet 4.6:

| Dimension | Distribution |
|---|---|
| Scenarios | coding 201; research 201; support 195; tool-heavy 201; analysis 201 |
| Target ratios | 333 each at 3x, 5x, and 10x |
| Turn ages | 333 each for recent, mid, and old |
| Mean source length | 7,512 characters |
| Mean target length | 1,190 characters |
| Mean achieved ratio | 6.31x |

The adapter has not been trained on WildChat, OASST2, or UltraChat. We
intentionally evaluate `tfix375` before any real-conversation augmentation so
the public benchmark measures out-of-distribution generalization. Retraining
is evidence-gated: it is considered only if the pilot identifies distribution
mismatch as the bottleneck.

### QLoRA recipe

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Quantization: NF4 4-bit with double quantization and BF16 compute
- LoRA: rank 16, alpha 32, dropout 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- Training: 3 epochs, learning rate `2e-4`, cosine schedule, 5% warmup
- Batch configuration: per-device batch 2, gradient accumulation 4
- Sequence length: 4,096
- Optimizer: paged AdamW 8-bit
- Training hardware: one RTX 4090

`train_lora.py` is the executable source of truth for these settings.

## Public benchmark policy

Headline evaluation uses public third-party conversations, not the synthetic
training corpus:

| Dataset | Role | Provenance/license |
|---|---|---|
| WildChat-1M | Primary real-user benchmark | real user prompts, ODC-BY |
| OASST2 | Primary human conversation benchmark | human prompts/replies, Apache-2.0 |
| UltraChat 200k | Public secondary benchmark | synthetic conversations, MIT |
| MT-Bench | Adapter available; yield depends on length filters | curated, CC-BY-4.0 |

Shared filters require English, at least four alternating turns, at least
3,000 content characters, and a final user/assistant pair suitable for
holdout evaluation. Dataset repository revisions and normalized artifact
hashes are recorded in each run manifest.

The Phase C pilot uses 3 accepted conversations from each of WildChat,
OASST2, and UltraChat: 9 conversations total. Final sample counts will be
locked after pilot yield and judge-agreement review. The six hand-written OOD
conversations remain diagnostic data only, not headline evidence.

## Systems compared

All runnable systems process the same conversations and tier targets:

| Output source | System | Role |
|---|---|---|
| `base-qwen` | Prompted Qwen2.5-7B-Instruct | same-base zero-shot baseline |
| `llmlingua2` | LLMLingua-2 | deterministic extractive baseline |
| `longllmlingua` | LongLLMLingua | long-context extractive baseline |
| `frontier-gpt54` | `gpt-5.4-2026-03-05` | frontier quality ceiling |
| `practical-gpt4o-mini` | `gpt-4o-mini-2024-07-18` | practical API competitor |
| `tfix375` | Qwen2.5-7B + EOS-fixed LoRA | our primary greedy system |

Our adapter is also decoded with three fixed sampled seeds
(`tfix375-s11`, `tfix375-s22`, `tfix375-s33`) as a secondary robustness
analysis. These variants are not separate headline competitors.

Cmprsr is discussed as related work but is not a runnable matrix column:
no public implementation or weights were found. We do not claim a direct
reproduction of its reported results.

## Pre-registered evaluation

The detailed measurement contracts are in [`docs/rubrics`](docs/rubrics).
The committed rubric prose preserves its historical pre-registration wording;
the executed OpenAI pin is GPT-5.4 because GPT-5.5 is not a valid model ID.

### 1. Critical-information faithfulness

1. A primary judge extracts a shared list of critical source items.
2. Both judge families independently label every shared item
   `present`, `partial`, or `false` in each compression.
3. Every positive label must include verbatim evidence from the compression.
4. Evidence is checked deterministically; unsupported positive calls are
   downgraded to `false`.
5. Score:
   `(present + 0.5 * partial) / total`.

This is paraphrase-tolerant without relying on an ungrounded holistic Likert
score. It does not require a hand-authored golden-facts corpus.

### 2. Downstream continuation

The last assistant response is held out before compression. A fixed,
temperature-zero continuation generator receives only:

```text
[compressed prior turns] + [last user turn]
```

Two judges compare the generated continuation with the held-out assistant
response on:

- substantive correctness;
- code/numeric/entity fidelity; and
- conversation coherence.

The same generator and configuration are used for every compression system.

### 3. Tier appropriateness

Per-conversation faithfulness scores form a recent/mid/old retention curve.
We report recent-to-old delta, monotonicity, and curve AUC rather than asking a
separate judge to guess whether a summary "looks old."

### Deterministic sanity metrics

Secondary diagnostics include duplicate 8-gram rate, bracketed structural
tags, meta-conversational leakage, natural text endings, achieved ratio, and
model-reported EOS stopping where available.

## Judges and agreement gates

The pilot uses two model families with the same rubric prompts and
temperature-zero configuration:

- OpenAI `gpt-5.4-2026-03-05`
- Anthropic Claude Sonnet 4.6

System identity is hidden from the judges. Structured outputs are schema
validated, API calls are cached append-only, and malformed provider tool
payloads are either narrowly normalized and fully revalidated or rejected.

Agreement is checked before scaling the experiment:

| Outcome | Cohen's kappa on binary item calls | ICC on ordinal/continuous scores |
|---|---:|---:|
| Acceptable | >= 0.4 | >= 0.5 |
| Marginal | 0.2-0.4 | 0.3-0.5 |
| Unacceptable | < 0.2 | < 0.3 |

An unacceptable result is treated as a rubric design failure and blocks the
larger benchmark. A blinded human calibration subset is planned after the
pilot and before the final experiment.

## Statistical design

- **Primary confidence intervals:** clustered bootstrap over conversation IDs,
  preserving all within-conversation system/tier dependence.
- **Paired binary comparison:** McNemar exact/continuity-corrected test on
  conversation-level wins and losses.
- **Paired magnitude comparison:** Wilcoxon signed-rank test.
- **Judge agreement:** Cohen's kappa for aligned categorical calls and
  ICC(2,1) for continuous/ordinal scores.
- **Seed robustness:** three sampled LoRA seeds reported separately from the
  primary greedy comparison.

McNemar and Wilcoxon are both reported because the former answers how often
systems win while the latter preserves the magnitude and ranking of paired
differences.

## Reproducible pod pipeline

`eval/pod_pipeline.py` is the canonical experiment entry point. It provides:

- parallel, revision-pinned dataset ingestion;
- deterministic normalization, validation, deduplication, and corpus hashing;
- bounded concurrent API compression;
- a sequential VRAM-safe GPU lane;
- one shared Qwen load for base, greedy LoRA, and sampled LoRA variants;
- separate full-context and contamination-free holdout outputs;
- append-only per-baseline shards with crash-safe resume;
- stage logs, failure JSONL, environment/model provenance, and manifests; and
- an exact completeness check before merged outputs are published.

Example public pilot:

```bash
python -m eval.pod_pipeline \
  --run-id public-pilot-01 \
  --datasets wildchat=3,oasst2=3,ultrachat=3 \
  --adapter checkpoints/qwen2.5-7b-compressor-eosfix/checkpoint-375
```

Stages may be run independently and resumed:

```bash
python -m eval.pod_pipeline --run-id public-pilot-01 --stages ingest \
  --datasets wildchat=3,oasst2=3,ultrachat=3
```

Run artifacts are written under `runs/<run-id>/` and excluded from Git.

## Pilot judging

The headline pilot judges the six systems listed above; sampled LoRA variants
remain a secondary robustness analysis:

```bash
python -m eval.run_pilot \
  --compressions runs/public-pilot-01/outputs/compressions_standard.jsonl \
  --downstream-compressions runs/public-pilot-01/outputs/compressions_downstream.jsonl \
  --conversations runs/public-pilot-01/corpus.jsonl \
  --systems base-qwen,frontier-gpt54,llmlingua2,longllmlingua,practical-gpt4o-mini,tfix375 \
  --our-system tfix375 \
  --metrics faithfulness,downstream,tier,sanity \
  --judges openai:gpt-5.4-2026-03-05:gpt-primary,anthropic:claude-sonnet-4-6:claude-secondary \
  --generator openai:gpt-5.4-2026-03-05:continuation-generator \
  --cache runs/public-pilot-01/judge_cache.jsonl \
  --out runs/public-pilot-01/pilot_results.json
```

Use `--dry-run` before paid judging. The cache allows identical commands to
resume after rate limits, pod restarts, or provider schema failures.

## RunPod environment

The canonical setup uses:

- RTX 4090, 24 GB;
- persistent network storage mounted at `/workspace`;
- image
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`;
- persistent virtual environment at `/workspace/.venv`; and
- Hugging Face/pip caches on the network volume.

After creating or restarting the pod:

```bash
cd /workspace/compressor
git pull origin main
cp pod_bootstrap.sh /workspace/pod_bootstrap.sh
source /workspace/pod_bootstrap.sh
```

Required secrets are exposed as environment variables:

```text
HF_TOKEN
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

`pod_bootstrap.sh` pins the CUDA-compatible training/evaluation stack and
upgrades an existing persistent environment without discarding cached model
weights.

## Repository layout

| Path | Purpose |
|---|---|
| `context_store.py`, `anchor_store.py`, `tool_store.py` | Runtime context and disk-backed stores |
| `compressor.py`, `async_pipeline.py` | Heuristic compressor and progressive background pipeline |
| `dataset_gen.py`, `train_lora.py` | Synthetic teacher-data generation and QLoRA training |
| `baselines/` | Unified prompted, LoRA, extractive, and API baselines |
| `eval/conversations.py` | Canonical structured conversation schema and holdout logic |
| `eval/ingest_corpus.py` | Public-dataset adapters and filters |
| `eval/pod_pipeline.py` | Pod-native data-to-compression orchestration |
| `eval/run_pilot.py` | Resumable judging, aggregation, agreement, and statistics |
| `eval/prompts/` | Versioned and hashed judge/continuation prompts |
| `docs/rubrics/` | Pre-registered measurement specifications |
| `tests/` | Offline tests for schemas, metrics, statistics, caching, and orchestration |
| `pod_bootstrap.sh` | Persistent RunPod environment setup |

## Local development

The heuristic runtime and offline tests do not require CUDA. Run the test
suite from the repository's parent directory so Python resolves this
directory as the `compressor` package rather than the sibling
`compressor.py` module:

```bash
cd ..
python -m pytest compressor/tests -q
```

Training and Qwen/LLMLingua inference require a supported NVIDIA CUDA
environment. Model adapters and experiment runs are intentionally excluded
from Git.

## License

A repository-level license has not yet been selected. Dataset and model
licenses remain those of their respective upstream sources; run manifests
record dataset provenance for generated benchmark artifacts.
