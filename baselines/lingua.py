"""
LLMLingua baselines — extractive prompt compression by token pruning.

Two flavours:

  * LLMLingua-2 — XLM-RoBERTa-large-based, BERT-level, 0.55B params, 3-6x
    faster than original LLMLingua. Task-agnostic. Our default extractive
    baseline because it's the strongest published extractive method.

  * LongLLMLingua — original LLMLingua (LLaMA-7B-based) with the long-context
    knobs flipped on: condition_in_question, dynamic_context_compression_ratio,
    rank_method="longllmlingua". Heavier (uses Llama-2-7B as the small model)
    but designed for multi-turn / long-context prompts.

Both wrap the same `llmlingua.PromptCompressor` class with different config.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from . import Baseline, CompressionRequest, CompressionResult
from ._qwen_runtime import TIER_MAX_NEW

logger = logging.getLogger(__name__)


# Default model names per the LLMLingua README. The bert-base multilingual
# variant is smaller (~280M) but the xlm-roberta-large variant is the one
# evaluated in the paper, so we use that as default for paper-defensibility.
LLMLINGUA2_DEFAULT_MODEL = (
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
)
LONGLLMLINGUA_DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"


class LLMLingua2Baseline:
    """Extractive token-pruning compressor (LLMLingua-2).

    `target_ratio` is honoured by passing `rate=1/target_ratio` to the
    underlying compressor. The compressor uses character-level token
    importance via its XLM-RoBERTa classifier and removes low-importance
    tokens until the rate is hit.

    Note: extractive compressors do NOT see `turn_age` — they have no
    knowledge of conversation structure. We still record the tier so the
    bake-off comparison is per-tier-fair, but expect tier-appropriateness
    scores to be low for extractive baselines.
    """

    name = "llmlingua2"

    def __init__(self, model_name: str = LLMLINGUA2_DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._compressor: Any = None

    def load(self) -> None:
        if self._compressor is not None:
            return
        from llmlingua import PromptCompressor

        logger.info("Loading LLMLingua-2 model %s ...", self._model_name)
        self._compressor = PromptCompressor(
            model_name=self._model_name,
            use_llmlingua2=True,
        )
        logger.info("LLMLingua2Baseline loaded.")

    def unload(self) -> None:
        import gc

        self._compressor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self._compressor is None:
            raise RuntimeError("LLMLingua2Baseline.load() must be called first")

        rate = request.target_compression_rate  # 1/target_ratio

        t0 = time.time()
        # force_tokens preserves newlines and question marks — important for
        # multi-turn conversation structure. Recommended by LLMLingua-2 docs.
        result = self._compressor.compress_prompt(
            request.conversation,
            rate=rate,
            force_tokens=["\n", "?"],
        )
        dt = time.time() - t0

        text = result["compressed_prompt"]

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
            gen_seconds=round(dt, 3),
            input_tokens=int(result.get("origin_tokens", 0)) or None,
            output_tokens=int(result.get("compressed_tokens", 0)) or None,
            max_new_tokens=TIER_MAX_NEW[request.turn_age],
            stop_reason=None,  # extractive — no generation, no stop reason
            stopped_on_eos=None,
            extras={
                "rate_target": rate,
                "rate_achieved": result.get("ratio"),
                "saving": result.get("saving"),
                "model_name": self._model_name,
            },
        )


class LongLLMLinguaBaseline:
    """Original LLMLingua with long-context knobs (LongLLMLingua-2024).

    Uses a small LM (default Llama-2-7B per paper config) to score token
    perplexity. Significantly heavier than LLMLingua-2 in both VRAM and
    wall-clock. Included because the paper-headline comparison should
    cover both extractive variants, and LongLLMLingua specifically targets
    multi-turn / long-context scenarios — which is our paper's setting.

    Tunables (defaults follow the LLMLingua README example):
      * condition_in_question="after_condition"   query-aware reordering
      * reorder_context="sort"                    re-rank context paragraphs
      * dynamic_context_compression_ratio=0.3     dynamic per-paragraph rate
      * condition_compare=True                    contrastive scoring
      * context_budget="+100"                     small slack on top of budget
      * rank_method="longllmlingua"               LongLLMLingua reranker

    LongLLMLingua expects a `question` arg conditioning the compression on a
    downstream query. Conversations don't naturally have a single "question",
    so we synthesize one from the turn-age tier (e.g., "What were the final
    decisions in this conversation?" for `old`). This is honest; the
    alternative — passing question="" — gives the LongLLMLingua paper its
    weakest configuration and would be unfair to the baseline.
    """

    name = "longllmlingua"

    _QUESTION_BY_TIER = {
        "recent": "What facts, decisions, code, errors, and numbers does this conversation contain?",
        "mid": "What is the main thread of this conversation, including key decisions and unresolved issues?",
        "old": "What were the final decisions, key constraints, and named entities in this conversation?",
    }

    def __init__(self, model_name: str = LONGLLMLINGUA_DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._compressor: Any = None

    def load(self) -> None:
        if self._compressor is not None:
            return
        from llmlingua import PromptCompressor

        logger.info("Loading LongLLMLingua small-LM %s ...", self._model_name)
        self._compressor = PromptCompressor(self._model_name)
        logger.info("LongLLMLinguaBaseline loaded.")

    def unload(self) -> None:
        import gc

        self._compressor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self._compressor is None:
            raise RuntimeError("LongLLMLinguaBaseline.load() must be called first")

        rate = request.target_compression_rate
        question = self._QUESTION_BY_TIER[request.turn_age]

        t0 = time.time()
        result = self._compressor.compress_prompt(
            [request.conversation],
            question=question,
            rate=rate,
            condition_in_question="after_condition",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua",
        )
        dt = time.time() - t0

        text = result["compressed_prompt"]

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
            gen_seconds=round(dt, 3),
            input_tokens=int(result.get("origin_tokens", 0)) or None,
            output_tokens=int(result.get("compressed_tokens", 0)) or None,
            max_new_tokens=TIER_MAX_NEW[request.turn_age],
            stop_reason=None,
            stopped_on_eos=None,
            extras={
                "rate_target": rate,
                "rate_achieved": result.get("ratio"),
                "saving": result.get("saving"),
                "model_name": self._model_name,
                "question": question,
            },
        )


# Verify Protocol conformance at import time
_b1: Baseline = LLMLingua2Baseline()  # type: ignore[assignment]
_b2: Baseline = LongLLMLinguaBaseline()  # type: ignore[assignment]
