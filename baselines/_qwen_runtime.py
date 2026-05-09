"""
Shared Qwen2.5 runtime helpers for the GPU baselines.

Centralizes:
  * The compression system prompt (one text used by base_qwen, qwen_lora,
    and frontier — keeps the "fair-comparison" promise honest)
  * 4-bit base-model loading
  * Greedy generation with explicit EOS handling and stop-reason reporting

Lifted out of `infer.py` so we can keep `infer.py` as the legacy bake-off
harness while the new baselines share one implementation.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_EOS_TOKEN = "<|im_end|>"

# (turn_age, target_ratio, max_new_tokens) — same caps as legacy bake-off so
# results are directly comparable to bakeoff_results_eosfix.jsonl
TIERS: list[tuple[str, int, int]] = [
    ("recent", 3, 700),
    ("mid", 5, 400),
    ("old", 10, 200),
]
TIER_MAX_NEW = {age: cap for age, _ratio, cap in TIERS}


# Identical wording to train_lora.py / infer.py so the LoRA adapter sees its
# training-time conditioning. Modify with care: any change invalidates prior
# bake-off results for the LoRA baseline.
SYSTEM_TEMPLATE = (
    "You are a context compressor.\n"
    "Turn age: {turn_age_desc}\n"
    "Target compression: ~1/{ratio}x\n"
    "Output only the compressed text."
)

TURN_AGE_DESC: dict[str, str] = {
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


def build_system_prompt(turn_age: str, ratio: int) -> str:
    """The system block shared by all prompt-based compressors (base Qwen,
    LoRA-adapter Qwen, GPT-4o). Identical wording across all three so the
    head-to-head measures the model, not the prompt."""
    if turn_age not in TURN_AGE_DESC:
        raise ValueError(f"unknown turn_age: {turn_age!r}")
    return SYSTEM_TEMPLATE.format(ratio=ratio, turn_age_desc=TURN_AGE_DESC[turn_age])


def build_qwen_chat_prompt(conversation: str, turn_age: str, ratio: int) -> str:
    """The full Qwen2.5 chat-template prompt with the assistant turn left
    OPEN so the model continues with the compression."""
    system_msg = build_system_prompt(turn_age, ratio)
    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{conversation}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_base_qwen_4bit() -> tuple[Any, Any]:
    """Load Qwen2.5-7B-Instruct in NF4 4-bit via bitsandbytes. Same recipe as
    infer.py so memory footprint and numerics are identical."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy-decode one continuation. Returns a dict with text + token-level
    metadata (counts, stop reason). Greedy is deterministic so re-runs match.

    The returned dict is shaped to flow directly into CompressionResult fields:

        text:                 str   decoded continuation, special tokens stripped
        gen_seconds:          float wall-clock of model.generate()
        input_tokens:         int   prompt length in tokens
        output_tokens:        int   number of NEW tokens generated
        max_new_tokens:       int   the budget that was passed in
        stop_reason:          str   "eos" | "max_new_tokens"
        stopped_on_eos:       bool  whether the last new token was eos
    """
    import torch

    eos_id = tokenizer.convert_tokens_to_ids(QWEN_EOS_TOKEN)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    n_prompt = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_id,
        )
    dt = time.time() - t0

    new_token_ids = out[0, n_prompt:]
    output_tokens = int(new_token_ids.shape[0])

    # Stop reason: if the last token is EOS, generation halted naturally;
    # otherwise we hit the cap. (HF returns at most max_new_tokens new tokens.)
    last_tok = int(new_token_ids[-1].item()) if output_tokens > 0 else -1
    stopped_on_eos = last_tok == eos_id
    stop_reason = "eos" if stopped_on_eos else "max_new_tokens"

    text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    return {
        "text": text,
        "gen_seconds": dt,
        "input_tokens": n_prompt,
        "output_tokens": output_tokens,
        "max_new_tokens": max_new_tokens,
        "stop_reason": stop_reason,
        "stopped_on_eos": stopped_on_eos,
    }
