# Compressor — Project Notes for Future Sessions

> Read this first. It captures the intent, current state, and every operational
> learning we've accumulated. The goal is for the next session to skip the
> back-and-forth and pick up from the current state cleanly.

---

## 1. What this project is

A **progressive, asynchronous context compressor** for long-running LLM
conversations. Two halves:

### 1a. Runtime stack (CPU-only, no GPU needed)

The `ContextCompressor` class in `__init__.py` wires together:

| Module | Role |
|---|---|
| `context_store.py` | Ordered turns + segments + token-budget tracking |
| `anchor_store.py` | Append-only disk store for extracted facts (numbers, names, decisions) |
| `tool_store.py` | Disk store for large tool results, replaced inline with stub strings |
| `compressor.py` | Heuristic extractive summariser (sumy LexRank/LSA + fallback) |
| `async_pipeline.py` | Background thread compressing old segments when budget pressure rises |
| `llm_loop.py` | Anthropic-API-driven conversation loop demoing the compressor |

This stack works **today** with no GPU. The heuristic compressor (`compressor.py`)
is fine for shipping; it just isn't as good as a fine-tuned LLM compressor.

### 1b. Training stack (CUDA required)

`train_lora.py` + `dataset_gen.py` produce a QLoRA-fine-tuned compressor
that should outperform the heuristic one. The output is a small (~150 MB)
LoRA adapter that gets grafted onto Qwen2.5-7B-Instruct at inference time.

---

## 2. Goal

Replace `HeuristicCompressor` with a **learned compressor** that:

- Respects three turn-age tiers: `recent` (preserve everything),
  `mid` (preserve main thread + decisions), `old` (preserve only outcomes).
- Hits the target compression ratio (3x / 5x / 10x) within ±10%.
- Generalises across scenario types (coding, research, support, tool_heavy,
  analysis) and to **unseen** conversation styles.

The end-state for the project is a swap-in `LoraCompressor` class that
implements the same interface as `HeuristicCompressor` but uses the trained
adapter. The runtime pipeline is otherwise unchanged.

---

## 3. Where we are now (state at end of last session)

| Phase | Status |
|---|---|
| Runtime stack (heuristic) | ✅ Shippable |
| Synthetic dataset (999 examples, 5 scenarios × 67 seeds × 3 ratios) | ✅ At `data/synthetic_dataset.jsonl` |
| QLoRA training (3 epochs, RTX 4090 on RunPod) | ✅ 75 min, $1.20 cost |
| Two checkpoints downloaded locally | ✅ `checkpoints/qwen2.5-7b-compressor/{checkpoint-249, checkpoint-372}` |
| Bake-off: sweet-spot vs overfit | 🟡 In progress — adapters loaded on pod, awaiting test corpus + `infer.py` |
| Swap LoraCompressor into runtime pipeline | ⏳ Pending |

Loss trajectory:
- Step 124 (epoch 1.0): ~0.90
- **Step 249 (epoch 2.0): 0.510** ← presumed sweet spot
- **Step 372 (epoch 3.0): 0.341** ← presumed memorising

The 0.51 → 0.34 drop across epoch 3 happened with characteristic step-function
behaviour at the epoch boundary, plus calm `grad_norm`. That's the fingerprint
of memorisation onset, not continued generalisation. The bake-off should
confirm that `checkpoint-249` produces better outputs on unseen conversations
despite higher train loss.

The 6 test conversations for the bake-off live at `data/bakeoff_conversations.jsonl`.

---

## 4. Setup runbook — distilled from everything we learned the hard way

### 4a. Local machine (Windows, what we have)

| Component | Notes |
|---|---|
| OS | Windows 11 |
| Python | 3.14.3 (system install) |
| GPU | AMD Radeon 8060S (Strix Halo iGPU) — **no CUDA, no ROCm-on-Windows** |
| Shell | PowerShell 7.6.1 (`pwsh.exe`) — required by tooling; PowerShell 5 / cmd not enough |

**Implication:** all GPU work happens on RunPod. Local box is for code edits,
zipping artefacts, and CPU-only inference of the heuristic compressor.

`.env` (in repo root, git-ignored) contains `ANTHROPIC_API_KEY=...` for
`dataset_gen.py`. **Note:** `train_lora.py` line 67 calls
`load_dotenv(".env.txt")` — wrong filename, but harmless since training auth
comes from `huggingface-cli login`, not the env file. Fix or leave; doesn't
block anything.

### 4b. RunPod — the canonical pod recipe

Copy these settings exactly when creating a new pod:

| Setting | Value |
|---|---|
| Cloud type | **Secure Cloud** |
| GPU | **RTX 4090 (24 GB)** — sufficient for QLoRA-7B at bs=1, seq=4096 |
| Region | **EU-RO-1** (locked by network volume) |
| Network volume | **`compressor_nw_vol`** (40 GB, persists Qwen weights + adapters between sessions) |
| Container disk | 30 GB |
| Template | **`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`** |
| Pricing | On-Demand $0.69/hr |
| Connection | Jupyter Lab on port 8888 (do **not** use the older 2.4 PyTorch template — it errored with "layer does not exist" and required retries) |

### 4c. Pod first-boot setup (skip steps 2-4 if reusing the network volume)

**Step 1 — environment variables (always run):**

```bash
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
source ~/.bashrc
```

`HF_HOME` keeps the 14 GB Qwen weights cached on the network volume across
pod terminations. `expandable_segments` prevents the CUDA OOM-from-fragmentation
that hit us at step 23 of the training run. `TOKENIZERS_PARALLELISM=false`
silences the noisy fork warnings.

> **Do NOT** set `TRANSFORMERS_CACHE` — it's deprecated and `HF_HOME`
> alone is sufficient. Setting it triggers a deprecation warning every run.

**Step 2 — fix the torch / CUDA mismatch (only on a fresh pod):**

The 2.8.0 template ships torch built against CUDA 13.0 but the host driver
caps at CUDA 12.8. Symptom: `torch.cuda.is_available() == False`. Fix:

```bash
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0
# torchvision/torchaudio NOT needed — we don't use them.
```

**Step 3 — install training/inference stack with the correct version pins:**

```bash
pip install --force-reinstall --no-deps bitsandbytes==0.45.2
pip install --force-reinstall "trl==0.11.4"
pip install --force-reinstall "transformers>=4.40,<4.46"
# Then re-pin torch — the transformers force-reinstall yanks it back to 2.11+cu13.
pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0
pip install peft accelerate datasets python-dotenv huggingface_hub hf_transfer
```

**Why these pins matter:**

| Pin | Reason |
|---|---|
| `torch==2.6.0+cu124` | Matches the pod driver's CUDA 12.x cap. Newer torch ships +cu13 wheels that won't init. |
| `bitsandbytes==0.45.2` | Last release with rock-solid CUDA-12 kernels and torch-2.6 compat. |
| `trl==0.11.4` | Last release before `liger_kernel` became a hard dep. Also keeps the `tokenizer=` and `dataset_text_field=` kwargs that our `train_lora.py` passes. |
| `transformers <4.46` | Older `trl` versions can't keep up with the latest transformers internals. |
| `--no-deps` on torch reinstall | Without this, pip's resolver yanks torch back to the latest +cu13 wheel via transitive deps. **The single most repeatedly-bitten issue in our setup history.** |

**Step 4 — HuggingFace auth (only on a fresh pod):**

```bash
huggingface-cli login   # paste fine-grained token with "Read access to gated repos" enabled
```

The Qwen2.5-7B-Instruct repo requires accepting its license once, in your
browser, at <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>. Approval is
instant.

**Step 5 — verify (one-liner, indentation-safe for bash `-c`):**

```bash
python -c "import torch, bitsandbytes, transformers, peft; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0)); print('bnb:', bitsandbytes.__version__, '| transformers:', transformers.__version__, '| peft:', peft.__version__)"
```

Expected output:
```
torch: 2.6.0+cu124 | CUDA: 12.4 | GPU: NVIDIA GeForce RTX 4090
bnb: 0.45.2 | transformers: 4.4x.x | peft: 0.x.x
```

If anything's off, fix before launching training/inference. CUDA mismatches
will fail loud thanks to the assertion at the top of `train_lora.main()`,
but inference scripts may silently fall back to CPU.

### 4d. Uploading code

Use the JupyterLab file browser (drag-drop or upload arrow). Always zip
locally first to preserve folder structure:

```powershell
# Run from C:\Users\apeppas\endeavors\research
$root = (Get-Location).Path; $stage = "$root\_stage"; $zip = "$root\compressor.zip"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
if (Test-Path $zip)   { Remove-Item $zip }
New-Item $stage -ItemType Directory | Out-Null
robocopy "$root\compressor" "$stage\compressor" /E /XD __pycache__ compressor.egg-info .claude checkpoints /XF .env *.pyc | Out-Null
Add-Type -Assembly System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($stage, $zip)
Remove-Item $stage -Recurse -Force
```

> **Don't use** `Compress-Archive` with piped `Get-ChildItem` — it flattens
> directories. `[ZipFile]::CreateFromDirectory` preserves structure correctly.

> **Always exclude** `.env`, `__pycache__`, `compressor.egg-info`, `.claude`,
> and `checkpoints` from uploads. The exclude pattern above handles all five.

### 4e. Training-specific gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: liger_kernel` | trl ≥0.13 hard-requires it | Downgrade to `trl==0.11.4` (use `--force-reinstall`, plain install won't downgrade) |
| `ValueError: hf_transfer` not found | Pod template enables `HF_HUB_ENABLE_HF_TRANSFER=1` but doesn't ship the package | `pip install hf_transfer` |
| `CUDA OOM` mid-training | Qwen's 152k-vocab logits tensor at seq=4096 + bs=2 + fragmentation | Set `bs=1` + `grad_accum=8` (effective batch unchanged), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `train_loss → nan` in first 20 steps | LR too high without warmup | Drop `learning_rate` from 2e-4 to 1e-4 in `TRAINING_ARGS` |
| Loss flat at starting value after 50 steps | Forgot `prepare_model_for_kbit_training` | Already in current `train_lora.py` — don't remove |
| Model loads but `torch.cuda.is_available() == False` | torch / driver mismatch | Reinstall torch with `--index-url https://download.pytorch.org/whl/cu124` |

### 4f. Pod teardown

After downloading what you need from the network volume:
1. **Stop** the pod (immediately stops GPU billing).
2. **Terminate** the pod (frees container disk; network volume persists with cached HF models + adapters).
3. The 40 GB network volume stays at ~$2.80/month — fine.

---

## 5. Cost log

| Run | Date | Duration | Cost |
|---|---|---|---|
| First training (3 epochs, RTX 4090) | 2026-04-30 | ~75 min + ~30 min setup | ~$1.20 |
| Bake-off inference (TBD) | 2026-05-02 | ~15 min projected | ~$0.20 projected |

---

## 6. Open questions / future work

- [ ] Bake-off `checkpoint-249` vs `checkpoint-372` to confirm the sweet-spot hypothesis. **Next session.**
- [ ] Wire the winning adapter into a `LoraCompressor` class implementing the same interface as `HeuristicCompressor`.
- [ ] Carve a 10% validation split from `synthetic_dataset.jsonl` and add `evaluation_strategy="epoch"` + `load_best_model_at_end=True` to `train_lora.py`. This would catch overfitting automatically next time, removing the manual bake-off step.
- [ ] Consider regenerating dataset with **explicit token budgets** (currently character budgets) — the model is being trained to hit char counts but Qwen's tokenizer doesn't preserve that uniformly across languages or code blocks.
- [ ] Fix `train_lora.py` line 67: `load_dotenv(".env.txt")` → `load_dotenv(".env")` for consistency with `dataset_gen.py`. Cosmetic; doesn't affect training.
