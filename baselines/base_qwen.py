"""
BaseQwen baseline — Qwen2.5-7B-Instruct, zero-shot prompted, no LoRA adapter.

This is the head-to-head baseline that says "what does our base model do
with the same compression prompt, before fine-tuning?" Identical inference
recipe to the LoRA-adapter baseline (greedy, 4-bit, same EOS handling) so
the LoRA-vs-base comparison isolates the effect of fine-tuning alone.
"""
from __future__ import annotations

import logging
from typing import Any

from . import Baseline, CompressionRequest, CompressionResult
from ._qwen_runtime import (
    TIER_MAX_NEW,
    build_qwen_chat_prompt,
    greedy_generate,
    load_base_qwen_4bit,
)

logger = logging.getLogger(__name__)


class BaseQwenBaseline:
    """Implements `Baseline`. Loads Qwen2.5-7B-Instruct and runs greedy."""

    name = "base-qwen"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model, self._tokenizer = load_base_qwen_4bit()
        logger.info("BaseQwenBaseline loaded.")

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

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("BaseQwenBaseline.load() must be called first")

        max_new = TIER_MAX_NEW[request.turn_age]
        prompt = build_qwen_chat_prompt(
            request.conversation, request.turn_age, request.target_ratio
        )
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
        )


# Verify at import time that we satisfy the Protocol
_baseline: Baseline = BaseQwenBaseline()  # type: ignore[assignment]
