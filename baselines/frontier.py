"""
Frontier ceiling baseline — GPT-4o via OpenAI API.

Runs locally (no GPU). Uses the same compression system prompt as base_qwen
and qwen_lora so the comparison is "what does a frontier model do with the
same instructions?" — interpreted as a quality ceiling for prompted
compression.

Decoding: temperature=0 for determinism (re-runs reproduce). The
`max_completion_tokens` cap mirrors the per-tier caps used by GPU baselines
(700 / 400 / 200 for recent / mid / old) so compute budget is matched.

Cost estimate (GPT-4o pricing, 2024 rates):
  * Input: $5 / 1M tokens
  * Output: $15 / 1M tokens
  * 6 conv x 3 tiers ~ ~80K input tokens + ~10K output tokens = ~$0.55
  * 40 conv x 3 tiers x 5 systems (this is one of them) ~ ~$5-10 total

Auth: requires `OPENAI_API_KEY` in env or `.env` file. The dotenv import is
optional so the module loads cleanly on the GPU pod where it'll never run.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from . import Baseline, CompressionRequest, CompressionResult
from ._qwen_runtime import TIER_MAX_NEW, build_system_prompt

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gpt-4o-2024-08-06"


class FrontierBaseline:
    """OpenAI GPT-4o called with the shared compression prompt.

    Deterministic via temperature=0. Same per-tier max-output-token caps
    as GPU baselines so all prompted compressors see matched budgets.
    """

    name = "gpt-4o"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self._model = model
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
                "FrontierBaseline() or set the env var / put it in .env"
            )

        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI(api_key=api_key)
        logger.info("FrontierBaseline loaded (model=%s).", self._model)

    def unload(self) -> None:
        self._client = None  # nothing to release; closing the HTTP pool is automatic

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self._client is None:
            raise RuntimeError("FrontierBaseline.load() must be called first")

        max_new = TIER_MAX_NEW[request.turn_age]
        system_msg = build_system_prompt(request.turn_age, request.target_ratio)

        t0 = time.time()
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": request.conversation},
            ],
            max_tokens=max_new,
            temperature=0.0,
            seed=42,
        )
        dt = time.time() - t0

        choice = completion.choices[0]
        text = (choice.message.content or "").strip()

        # OpenAI's finish_reason vocabulary: "stop" (natural EOS), "length"
        # (max_tokens hit), "content_filter", "tool_calls", "function_call".
        # We map to our schema: "eos" / "max_new_tokens" / other.
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


# Verify Protocol conformance at import time
_baseline: Baseline = FrontierBaseline()  # type: ignore[assignment]
