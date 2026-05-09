"""
QwenLoRA baseline — Qwen2.5-7B-Instruct + our tier-conditioned LoRA adapter.

This is "ours" in the head-to-head. Single instance can switch between
multiple adapters via PeftModel.set_adapter() so we can A/B between, e.g.,
checkpoint-250 (sweet spot) and checkpoint-375 (final) without reloading
the 4-bit base.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import Baseline, CompressionRequest, CompressionResult
from ._qwen_runtime import (
    TIER_MAX_NEW,
    build_qwen_chat_prompt,
    greedy_generate,
    load_base_qwen_4bit,
)

logger = logging.getLogger(__name__)


class QwenLoRABaseline:
    """Wraps a single LoRA adapter on top of the 4-bit Qwen base.

    Construct with the adapter directory and a short name (used as the
    `source` field in JSONL output). To compare multiple checkpoints, build
    one instance per checkpoint — the base model is loaded lazily so the
    overhead is only the small adapter weights per extra checkpoint.

    For multi-adapter A/B with shared base memory, prefer the legacy
    `infer.py:run_bake_off()` path until we extend this class.
    """

    def __init__(self, adapter_path: str | Path, name: str) -> None:
        self._adapter_path = Path(adapter_path)
        self.name = name
        self._model: Any = None
        self._tokenizer: Any = None
        self._adapter_name = "primary"

    def load(self) -> None:
        if self._model is not None:
            return
        from peft import PeftModel

        if not self._adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {self._adapter_path}"
            )

        base, tokenizer = load_base_qwen_4bit()
        logger.info(
            "Attaching adapter '%s' from %s", self._adapter_name, self._adapter_path
        )
        model = PeftModel.from_pretrained(
            base, str(self._adapter_path), adapter_name=self._adapter_name
        )
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        logger.info("QwenLoRABaseline %r loaded.", self.name)

    def unload(self) -> None:
        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @contextmanager
    def _adapter_active(self) -> Iterator[None]:
        """Ensure the LoRA adapter is active for the duration of the block.
        Defends against accidental disable_adapter() bleed-through across
        multi-baseline runs."""
        self._model.set_adapter(self._adapter_name)
        try:
            yield
        finally:
            pass  # adapter remains set; idempotent across calls

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("QwenLoRABaseline.load() must be called first")

        max_new = TIER_MAX_NEW[request.turn_age]
        prompt = build_qwen_chat_prompt(
            request.conversation, request.turn_age, request.target_ratio
        )
        with self._adapter_active():
            gen = greedy_generate(self._model, self._tokenizer, prompt, max_new)

        text = gen["text"]
        return CompressionResult(
            conversation_id=request.conversation_id,
            scenario_type=request.scenario_type,
            turn_age=request.turn_age,
            target_ratio=request.target_ratio,
            source=self.name,
            compressed=text,
            input_chars=len(request.conversation),
            output_chars=len(text),
            achieved_ratio=len(request.conversation) / max(len(text), 1),
            gen_seconds=round(gen["gen_seconds"], 3),
            input_tokens=gen["input_tokens"],
            output_tokens=gen["output_tokens"],
            max_new_tokens=gen["max_new_tokens"],
            stop_reason=gen["stop_reason"],
            stopped_on_eos=gen["stopped_on_eos"],
            extras={"adapter_path": str(self._adapter_path)},
        )


# Note: we don't instantiate-and-check QwenLoRABaseline at import time the way
# BaseQwenBaseline does, because the constructor requires an adapter path.
# Protocol conformance is validated structurally on first use.
