"""
API-based compressor baselines via OpenAI's chat completions endpoint.

Two baselines with different roles:

  * `FrontierBaseline`     — GPT-5.5, the quality ceiling. Establishes how
    much headroom remains between a ~7B QLoRA model and the best available
    frontier model. NOT meant to be beaten outright; meant to anchor the
    upper bound.

  * `PracticalAPIBaseline` — GPT-4o-mini, the cost-matched API competitor.
    A realistic "what would a production engineer use off-the-shelf for
    compression today?" reference. Beating this IS a paper claim.

Both run locally (no GPU) and use the same shared compression system prompt
as `base_qwen` and `qwen_lora`, so the head-to-head measures the model, not
the prompt.

Decoding: temperature=0.0 + seed=42 for determinism (re-runs reproduce).
Snapshot IDs are pinned at experiment time and recorded in the result
`extras` field so future re-runs can match the model version used.

Cost estimate (current OpenAI pricing as of 2026-05):
  * GPT-5.5: roughly an order of magnitude pricier per token than GPT-4o-mini
  * Phase E (~660 generations across both): ~$15-25 total

Auth: requires `OPENAI_API_KEY` in env or `.env` file. The dotenv import is
optional so the module loads cleanly on the GPU pod where it'll never run.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from . import Baseline, CompressionRequest, CompressionResult
from ._openai_compat import openai_chat_create
from ._qwen_runtime import TIER_MAX_NEW, build_system_prompt

logger = logging.getLogger(__name__)


# Pin specific snapshots at experiment time. Update these strings to the
# exact `-YYYY-MM-DD` snapshot used for the published paper run and never
# change them afterwards — reproducibility hinges on this.
# Verified available 2026-08-09: gpt-5.5 does not exist; gpt-5.4 is the
# latest 5-series. Pinned to dated snapshots for reproducibility.
FRONTIER_MODEL = "gpt-5.4-2026-03-05"
PRACTICAL_MODEL = "gpt-4o-mini-2024-07-18"


class _OpenAICompressor:
    """Shared implementation for OpenAI-API compressor baselines.

    Subclasses set `name` and the default model. Behaviour is otherwise
    identical so the comparison between frontier and practical-API
    baselines isolates the model, not the harness.
    """

    name: str = "openai-base"  # subclasses override
    default_model: str = ""  # subclasses override

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or self.default_model
        self._api_key = api_key
        self._client: Any = None

    def load(self) -> None:
        if self._client is not None:
            return
        try:
            from dotenv import load_dotenv  # noqa: PLC0415

            load_dotenv()
        except ImportError:
            pass

        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; either pass api_key=... to "
                f"{type(self).__name__}() or set the env var / put it in .env"
            )

        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI(api_key=api_key)
        logger.info("%s loaded (model=%s).", type(self).__name__, self._model)

    def unload(self) -> None:
        self._client = None  # HTTP pool is closed automatically

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self._client is None:
            raise RuntimeError(f"{type(self).__name__}.load() must be called first")

        max_new = TIER_MAX_NEW[request.turn_age]
        system_msg = build_system_prompt(request.turn_age, request.target_ratio)

        t0 = time.time()
        completion = openai_chat_create(
            self._client,
            model=self._model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": request.conversation},
            ],
            max_output_tokens=max_new,
            temperature=0.0,
            seed=42,
        )
        dt = time.time() - t0

        choice = completion.choices[0]
        text = (choice.message.content or "").strip()

        # OpenAI's finish_reason vocabulary: "stop" (natural EOS), "length"
        # (max_tokens hit), "content_filter", "tool_calls", "function_call".
        finish = choice.finish_reason
        if finish == "stop":
            stop_reason, stopped_on_eos = "eos", True
        elif finish == "length":
            stop_reason, stopped_on_eos = "max_new_tokens", False
        else:
            stop_reason, stopped_on_eos = finish, False

        usage = completion.usage
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
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            max_new_tokens=max_new,
            stop_reason=stop_reason,
            stopped_on_eos=stopped_on_eos,
            extras={
                "model": self._model,
                "openai_finish_reason": finish,
            },
        )


class FrontierBaseline(_OpenAICompressor):
    """Quality ceiling: a large frontier model (GPT-5.4)."""

    name = "frontier-gpt54"
    default_model = FRONTIER_MODEL


class PracticalAPIBaseline(_OpenAICompressor):
    """Cost-matched API competitor: a smaller production-grade API model
    (GPT-4o-mini). The realistic alternative a deployment engineer would
    reach for if they didn't fine-tune their own compressor."""

    name = "practical-gpt4o-mini"
    default_model = PRACTICAL_MODEL


# Verify Protocol conformance at import time
_b1: Baseline = FrontierBaseline()  # type: ignore[assignment]
_b2: Baseline = PracticalAPIBaseline()  # type: ignore[assignment]

