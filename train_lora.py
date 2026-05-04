"""train_lora.py — QLoRA fine-tuning script for Qwen2.5-7B-Instruct.

Fine-tunes Qwen2.5-7B-Instruct (or the 14B variant) on the synthetic
compression dataset produced by ``dataset_gen.py``, using QLoRA (4-bit
quantisation + LoRA adapters).

Training configuration
----------------------
LoRA:
    r=16, lora_alpha=32, lora_dropout=0.05
    target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

QLoRA:
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bfloat16, double_quant=True

SFT:
    epochs=3, per_device_batch=2, grad_accumulation=4,
    lr=2e-4, scheduler=cosine, warmup_ratio=0.05, bf16=True

Prompt format (Qwen2.5 chat template):
    <|im_start|>system
    You are a context compressor. ...
    <|im_end|>
    <|im_start|>user
    {original}
    <|im_end|>
    <|im_start|>assistant
    {compressed}<|im_end|>

Usage
-----
    # Standard training run
    python -m compressor.train_lora

    # Dry-run: load model, tokenise one example, print prompt, exit
    python -m compressor.train_lora --dry-run

    # Use the 14B model (quality reference)
    python -m compressor.train_lora --model Qwen/Qwen2.5-14B-Instruct

    # Custom dataset path
    python -m compressor.train_lora --dataset /path/to/dataset.jsonl

Notes
-----
* torch, transformers, peft, trl, and bitsandbytes are imported lazily inside
  ``main()`` so that the module is safely importable in environments where
  those packages are not installed (e.g. the sandbox used for testing the
  heuristic compressor).
* Requires a CUDA GPU with ≥24 GB VRAM for the 7B model (≥48 GB for 14B).
* Set ANTHROPIC_API_KEY in the environment / .env file if you want to call
  the teacher model for additional data generation before training.
* **EOS supervision invariant** — for chat-template models where ``eos_token``
  doubles as the assistant-turn terminator (Qwen2.5: ``<|im_end|>``,
  Llama-3: ``<|eot_id|>``), ``pad_token_id`` MUST be distinct from
  ``eos_token_id``. The common idiom ``tokenizer.pad_token = tokenizer.eos_token``
  causes SFTTrainer's collator to mask every legitimate EOS in the labels with
  -100, so the model never learns to stop. We assert this invariant in
  ``main()`` and fail loudly if violated.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(".env.txt")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = str(Path(__file__).parent / "data" / "synthetic_dataset.jsonl")
# NOTE: default output dir suffix `-eosfix` distinguishes runs done with the
# corrected pad_token != eos_token configuration. The original (pre-fix)
# checkpoints in `qwen2.5-7b-compressor/` are preserved as the broken baseline
# for the paper's before/after analysis.
DEFAULT_OUTPUT_DIR = str(Path(__file__).parent / "checkpoints" / "qwen2.5-7b-compressor-eosfix")

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

TRAINING_ARGS = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    tf32=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

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


def build_prompt(example: dict[str, Any]) -> str:
    """
    Format a training example using the Qwen2.5 chat template.

    Args:
        example: A dataset dict with keys ``original``, ``compressed``,
                 ``target_ratio``, and ``turn_age``.

    Returns:
        The full prompt string ready for tokenisation.
    """
    ratio = int(example.get("target_ratio", 3))
    turn_age = example.get("turn_age", "recent")
    system_msg = _SYSTEM_TEMPLATE.format(
        ratio=ratio,
        turn_age_desc=_TURN_AGE_DESC.get(turn_age, _TURN_AGE_DESC["recent"]),
    )
    original = example.get("original", "")
    compressed = example.get("compressed", "")

    return (
        f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
        f"<|im_start|>user\n{original}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{compressed}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset_from_jsonl(path: str) -> list[dict[str, Any]]:
    """
    Load examples from a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of example dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run `python -m compressor.dataset_gen` first to generate it."
        )
    examples: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Entry point for the QLoRA training script."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )

    # ------------------------------------------------------------------
    # Guard: import heavy training libraries lazily so this module is
    # importable in environments without GPU / torch installed.
    # ------------------------------------------------------------------
    try:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import (  # noqa: PLC0415
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from trl import SFTTrainer, SFTConfig  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"[train_lora] Required training library not available: {exc}\n"
            f"Install the full training stack with:\n"
            f"  pip install torch transformers peft trl bitsandbytes accelerate",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # CUDA / hardware sanity check — fail loud and early if no GPU.
    # QLoRA requires CUDA; running on CPU silently is a multi-hour mistake.
    # ------------------------------------------------------------------
    if not args.dry_run:
        if not torch.cuda.is_available():
            print(
                "[train_lora] FATAL: CUDA is not available.\n"
                "  QLoRA requires an NVIDIA GPU with bitsandbytes support.\n"
                "  torch was built with CUDA: "
                f"{torch.version.cuda}\n"
                "  Install a CUDA-enabled torch build, or run with --dry-run.",
                file=sys.stderr,
            )
            sys.exit(1)
        device_name = torch.cuda.get_device_name(0)
        device_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        bf16_ok = torch.cuda.is_bf16_supported()
        logger.info(
            "CUDA device 0: %s (%.1f GB VRAM, bf16=%s, torch=%s, cuda=%s)",
            device_name, device_mem_gb, bf16_ok,
            torch.__version__, torch.version.cuda,
        )
        if not bf16_ok:
            logger.warning(
                "GPU does not support bf16 — training will likely fail. "
                "Switch bf16=True → fp16=True in TRAINING_ARGS for older GPUs."
            )
        # Enable TF32 matmul on Ampere+ for an extra speed bump.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset from %s", args.dataset)
    examples = load_dataset_from_jsonl(args.dataset)
    logger.info("%d training examples loaded", len(examples))

    # ------------------------------------------------------------------
    # Build prompts
    # ------------------------------------------------------------------
    prompts = [build_prompt(ex) for ex in examples]

    # ------------------------------------------------------------------
    # Dry-run mode: tokenise first example, print, exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] Loading tokeniser…")
        tokeniser = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True
        )
        sample_prompt = prompts[0]
        tokens = tokeniser(sample_prompt, return_tensors="pt")
        print(f"\nModel:          {args.model}")
        print(f"Dataset:        {args.dataset}  ({len(examples)} examples)")
        print(f"Token budget:   {tokens['input_ids'].shape[1]} tokens for example 0")
        print(f"\n{'─'*72}")
        print("Formatted prompt (example 0):")
        print('─' * 72)
        print(sample_prompt)
        print('─' * 72)
        print("\n[DRY RUN] Complete — exiting without training.")
        return

    # ------------------------------------------------------------------
    # QLoRA quantisation config
    # ------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ------------------------------------------------------------------
    # Load tokeniser and model
    # ------------------------------------------------------------------
    logger.info("Loading tokeniser: %s", args.model)
    tokeniser = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # CRITICAL: do NOT set pad_token = eos_token. SFTTrainer's default collator
    # masks every pad_token_id from labels with -100. If pad_id == eos_id, every
    # <|im_end|> (Qwen2.5 EOS) at the end of each training example gets masked
    # from the loss; the model never learns to predict its own EOS, which
    # produces severe repetition collapse and budget non-adherence at inference.
    # Qwen2.5-Instruct ships with pad_token=<|endoftext|> distinct from
    # eos_token=<|im_end|>; we keep that distinction.
    assert tokeniser.pad_token_id is not None, (
        "Tokeniser has no pad_token. Add a dedicated pad token via "
        "tokeniser.add_special_tokens({'pad_token': '<|pad|>'}) and resize "
        "model embeddings; do NOT reuse eos_token as pad."
    )
    assert tokeniser.pad_token_id != tokeniser.eos_token_id, (
        f"pad_token_id ({tokeniser.pad_token_id}) must differ from "
        f"eos_token_id ({tokeniser.eos_token_id}). When they share an id, "
        f"SFTTrainer masks the legitimate EOS at the end of every training "
        f"example, breaking stop-token supervision."
    )
    logger.info(
        "Tokeniser pad/eos distinct: pad=%r (id=%d), eos=%r (id=%d)",
        tokeniser.pad_token, tokeniser.pad_token_id,
        tokeniser.eos_token, tokeniser.eos_token_id,
    )
    tokeniser.padding_side = "right"

    logger.info("Loading model in 4-bit: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # Prepare 4-bit model for k-bit training.
    # This casts layer norms to fp32, enables input gradients on the
    # frozen embedding layer, and is required for stable QLoRA training.
    # Must be called BEFORE get_peft_model.
    # ------------------------------------------------------------------
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )

    # ------------------------------------------------------------------
    # Apply LoRA adapters
    # ------------------------------------------------------------------
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoints will be saved to: %s", output_dir)

    # ------------------------------------------------------------------
    # SFTTrainer
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_seq_length=4096,
        **TRAINING_ARGS,
    )

    # Build a lightweight HuggingFace-compatible dataset from our prompts
    try:
        from datasets import Dataset as HFDataset  # noqa: PLC0415
    except ImportError as exc:
        print(
            "[train_lora] The 'datasets' package is required for SFTTrainer.\n"
            "Install it with: pip install datasets",
            file=sys.stderr,
        )
        raise

    hf_dataset = HFDataset.from_dict({"text": prompts})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokeniser,
        train_dataset=hf_dataset,
        args=sft_config,
        dataset_text_field="text",
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting training…")
    trainer.train()

    # ------------------------------------------------------------------
    # Save final adapter
    # ------------------------------------------------------------------
    logger.info("Saving LoRA adapter to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokeniser.save_pretrained(str(output_dir))
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Qwen2.5-7B-Instruct for context compression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID to fine-tune.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to the .jsonl training dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save LoRA checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model/tokeniser, print one formatted prompt, and exit.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
