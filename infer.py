"""
Bake-off inference: compare two trained QLoRA adapters against the base Qwen2.5-7B
on a fresh, hand-written, OOD-verified set of conversations.

For each (conversation × turn-age tier × source) we run greedy decoding and write
both a JSONL of raw outputs and a side-by-side Markdown report for human review.

Usage on the pod:
    python -m infer \\
        --bakeoff-data data/bakeoff_conversations.jsonl \\
        --adapter-a checkpoints/qwen2.5-7b-compressor/checkpoint-249 \\
        --adapter-b checkpoints/qwen2.5-7b-compressor/checkpoint-372 \\
        --include-base \\
        --out-md data/bakeoff_results.md \\
        --out-jsonl data/bakeoff_results.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("infer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Mirrors train_lora.py exactly so the model sees its training-time conditioning.
_SYSTEM_TEMPLATE = (
    "You are a context compressor.\n"
    "Turn age: {turn_age_desc}\n"
    "Target compression: ~1/{ratio}x\n"
    "Output only the compressed text."
)

_TURN_AGE_DESC: dict[str, str] = {
    "recent": (
        "recent — preserve ALL facts, reasoning arc, code, error messages, "
        "and numeric values; the reader must be able to continue the "
        "conversation from this summary alone"
    ),
    "mid": (
        "mid-age — preserve MOST facts and the main narrative thread; drop "
        "elaboration and worked examples; keep every decision, specific value, "
        "named entity, and any unresolved issue"
    ),
    "old": (
        "old — keep ONLY final decisions, key constraints, named entities still "
        "referenced downstream, and causal dependencies; no reasoning, no "
        "intermediate steps"
    ),
}

# (turn_age, target_ratio, max_new_tokens) — generation budget scales with ratio
TIERS = [
    ("recent", 3, 700),
    ("mid",    5, 400),
    ("old",   10, 200),
]


def build_inference_prompt(original: str, turn_age: str, ratio: int) -> str:
    """Build the same Qwen2.5 chat-template prompt used during training, but with
    the assistant turn left OPEN so the model generates the compression."""
    system_msg = _SYSTEM_TEMPLATE.format(
        ratio=ratio,
        turn_age_desc=_TURN_AGE_DESC[turn_age],
    )
    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{original}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def load_base_model() -> tuple:
    logger.info("Loading base model %s in 4-bit ...", BASE_MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def attach_adapters(base_model, adapter_paths: dict[str, Path]):
    """Attach 1+ adapters to the base model and return the wrapped PeftModel.
    Adapters can later be activated via model.set_adapter(name)."""
    items = list(adapter_paths.items())
    first_name, first_path = items[0]
    logger.info("Attaching adapter '%s' from %s", first_name, first_path)
    model = PeftModel.from_pretrained(base_model, str(first_path), adapter_name=first_name)
    for name, path in items[1:]:
        logger.info("Attaching adapter '%s' from %s", name, path)
        model.load_adapter(str(path), adapter_name=name)
    model.eval()
    return model


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    """Greedy-decode one generation. Returns (text, seconds)."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt = inputs["input_ids"].shape[1]
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    dt = time.time() - t0
    new_tokens = out[0, n_prompt:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, dt


def run_bake_off(
    bakeoff_path: Path,
    adapter_paths: dict[str, Path],
    include_base: bool,
    out_md: Path,
    out_jsonl: Path,
) -> None:
    conversations = load_jsonl(bakeoff_path)
    logger.info("Loaded %d bakeoff conversations from %s", len(conversations), bakeoff_path)

    base, tokenizer = load_base_model()
    model = attach_adapters(base, adapter_paths)

    # source label -> context manager that activates that source
    sources: list[tuple[str, callable]] = []
    if include_base:
        sources.append(("base", lambda: model.disable_adapter()))
    for name in adapter_paths:
        # Capture name in default-arg to avoid late-binding gotcha
        sources.append((name, lambda n=name: _set_adapter_ctx(model, n)))

    results: list[dict] = []
    n_total = len(conversations) * len(TIERS) * len(sources)
    done = 0
    overall_start = time.time()

    for convo in conversations:
        original = convo["conversation"]
        cid = convo["id"]
        for turn_age, ratio, max_new in TIERS:
            prompt = build_inference_prompt(original, turn_age, ratio)
            for source_name, ctx_fn in sources:
                done += 1
                with ctx_fn():
                    text, secs = generate_one(model, tokenizer, prompt, max_new)
                results.append({
                    "conversation_id": cid,
                    "scenario_type": convo.get("scenario_type", ""),
                    "turn_age": turn_age,
                    "target_ratio": ratio,
                    "source": source_name,
                    "input_chars": len(original),
                    "output_chars": len(text),
                    "achieved_ratio": (len(original) / max(len(text), 1)),
                    "gen_seconds": round(secs, 2),
                    "compressed": text,
                })
                logger.info(
                    "[%d/%d] %s  age=%-6s  src=%-7s  in=%d  out=%d  ratio=%.1fx  %.1fs",
                    done, n_total, cid, turn_age, source_name,
                    len(original), len(text),
                    len(original) / max(len(text), 1), secs,
                )

    overall_secs = time.time() - overall_start
    logger.info("All %d generations done in %.1f sec", n_total, overall_secs)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(results), out_jsonl)

    write_markdown_report(out_md, conversations, results, sources, overall_secs)
    logger.info("Wrote human-readable report to %s", out_md)


def _set_adapter_ctx(model, name: str):
    """Helper to make set_adapter work as a context manager (PEFT exposes it
    only as a function call, not a context). We just activate and never
    deactivate, since the next ctx in the loop will re-activate something."""
    model.set_adapter(name)
    return nullcontext()


def write_markdown_report(out_path: Path, conversations, results, sources, overall_secs: float):
    src_names = [s[0] for s in sources]
    by_key = {(r["conversation_id"], r["turn_age"], r["source"]): r for r in results}

    lines: list[str] = []
    lines.append("# Bake-off Results\n")
    lines.append(f"- Conversations: **{len(conversations)}**, "
                 f"tiers: **{len(TIERS)}** (recent/mid/old), "
                 f"sources: **{len(sources)}** ({', '.join(src_names)})\n")
    lines.append(f"- Total generations: **{len(results)}**, "
                 f"wall time: **{overall_secs:.1f}s**\n")
    lines.append("\n## Achieved-ratio summary (avg per tier × source)\n\n")
    lines.append("| tier | " + " | ".join(src_names) + " |\n")
    lines.append("|---" * (len(src_names) + 1) + "|\n")
    for turn_age, ratio, _ in TIERS:
        row = [f"{turn_age} (target {ratio}x)"]
        for s in src_names:
            ratios = [r["achieved_ratio"] for r in results
                      if r["turn_age"] == turn_age and r["source"] == s]
            avg = sum(ratios) / len(ratios) if ratios else 0
            row.append(f"{avg:.2f}x")
        lines.append("| " + " | ".join(row) + " |\n")
    lines.append("\n---\n")

    for convo in conversations:
        cid = convo["id"]
        lines.append(f"\n## {cid} ({convo.get('scenario_type', '')})\n")
        lines.append(f"\n*Input length: {len(convo['conversation'])} chars*\n")
        lines.append("\n<details><summary>📜 Original conversation</summary>\n\n```\n")
        lines.append(convo["conversation"])
        lines.append("\n```\n\n</details>\n")
        for turn_age, ratio, _ in TIERS:
            lines.append(f"\n### turn_age = `{turn_age}` (target ~{ratio}x compression)\n")
            for s in src_names:
                r = by_key.get((cid, turn_age, s))
                if r is None:
                    continue
                lines.append(f"\n**`{s}`** — {r['output_chars']} chars, "
                             f"achieved {r['achieved_ratio']:.2f}x, "
                             f"{r['gen_seconds']}s\n")
                lines.append(f"\n> {r['compressed'].replace(chr(10), chr(10) + '> ')}\n")
        lines.append("\n---\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="QLoRA compressor bake-off inference")
    parser.add_argument("--bakeoff-data", type=Path,
                        default=Path("data/bakeoff_conversations.jsonl"))
    parser.add_argument("--adapter-a", type=Path,
                        default=Path("checkpoints/qwen2.5-7b-compressor/checkpoint-249"))
    parser.add_argument("--adapter-b", type=Path,
                        default=Path("checkpoints/qwen2.5-7b-compressor/checkpoint-372"))
    parser.add_argument("--name-a", default="cp249", help="Display name for adapter A")
    parser.add_argument("--name-b", default="cp372", help="Display name for adapter B")
    parser.add_argument("--include-base", action="store_true",
                        help="Also run the base Qwen2.5-7B with no adapter, as a control")
    parser.add_argument("--out-md", type=Path, default=Path("data/bakeoff_results.md"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/bakeoff_results.jsonl"))
    args = parser.parse_args()

    adapter_paths = {args.name_a: args.adapter_a, args.name_b: args.adapter_b}
    for name, p in adapter_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Adapter path for '{name}' not found: {p}")
    if not args.bakeoff_data.exists():
        raise FileNotFoundError(f"Bakeoff data not found: {args.bakeoff_data}")

    run_bake_off(
        bakeoff_path=args.bakeoff_data,
        adapter_paths=adapter_paths,
        include_base=args.include_base,
        out_md=args.out_md,
        out_jsonl=args.out_jsonl,
    )

    # Free GPU mem on exit (paranoia for shared pods)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
