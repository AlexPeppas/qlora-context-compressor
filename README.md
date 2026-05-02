# qlora-context-compressor

A QLoRA fine-tune of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) that learns to **compress long-running LLM conversation context** into compact, semantically faithful summaries. The compression behaviour is conditioned on a `turn_age` signal (recent / mid / old) so older turns get aggressive outcome-only summaries while recent turns are preserved more verbatim.

The end goal is to drop the trained adapter into a runtime context-management pipeline as a learned alternative to a hand-written heuristic compressor (`compressor.py`).

## Status

- ✅ Synthetic dataset generated (999 examples, 5 scenario types × 67 seed conversations × 3 turn-age tiers) via teacher-LLM distillation in `dataset_gen.py`
- ✅ QLoRA training pipeline (`train_lora.py`) — 4-bit base, rank-32 LoRA on q/k/v/o + gate/up/down, paged 8-bit AdamW, gradient checkpointing
- ✅ One full training run completed on RTX 4090 (3 epochs, 372 steps, ~75 min, train loss 1.72 → 0.34)
- 🟡 Bake-off comparing two candidate checkpoints (epoch-2 sweet-spot vs epoch-3 presumed-overfit) on 6 fresh out-of-distribution conversations — pending
- 🔲 Wire the winning adapter into a `LoraCompressor` class behind the same interface as `compressor.HeuristicCompressor`

## Repo layout

| File | Purpose |
|---|---|
| `compressor.py` | Existing heuristic compressor — the baseline the LoRA needs to beat |
| `dataset_gen.py` | Teacher-LLM dataset generator (Anthropic API) |
| `train_lora.py` | QLoRA training script (Qwen2.5-7B + PEFT + bitsandbytes + TRL) |
| `data/synthetic_dataset.jsonl` | 999 training examples |
| `data/bakeoff_conversations.jsonl` | 6 hand-written, OOD-verified evaluation conversations |
| `data/dataset_audit_report.md` | Quality-audit notes on the synthetic dataset |
| `context_store.py`, `anchor_store.py`, `tool_store.py`, `async_pipeline.py`, `llm_loop.py` | Runtime context-management plumbing |
| `CLAUDE.md` | Project memory — full setup runbook, pin list, gotchas, training cost log |
| `lora_injection.png` | Pedagogical diagram of how LoRA injects into a frozen linear layer |

## Trained checkpoints

**Not in the repo** — the trained adapters are ~150 MB each and the optimizer state is ~300 MB each, well above GitHub's 100 MB per-file limit. To get adapters:

1. **Retrain from scratch** — see `CLAUDE.md` for the full RunPod runbook (RTX 4090, ~$1.20, ~75 min, all version pins documented). Then run `python -m train_lora`.
2. **Download pre-trained** — once we publish to the Hugging Face Hub, the path will go here. *(TODO)*

## Quickstart (training)

Real training requires a CUDA GPU with ≥24 GB VRAM. See `CLAUDE.md` Section 4 for the production runbook. For local dry-run sanity check:

```bash
pip install -r requirements.txt
python -m train_lora --dry-run  # tokenizer + dataset wiring check, no GPU needed
```

## License

TBD.
