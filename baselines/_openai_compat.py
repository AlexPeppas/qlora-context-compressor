"""OpenAI chat-completions compatibility shim.

The OpenAI API changed parameters across model generations:

  * Older models (gpt-4o, gpt-4o-mini): accept `max_tokens`, `temperature`,
    `seed`.
  * Newer models (gpt-5.x reasoning family): require `max_completion_tokens`
    instead of `max_tokens`, and some reject `temperature` != 1 and/or `seed`.

`openai_chat_create` retries with progressively-adjusted parameters when the
API rejects one, so the SAME calling code works across both generations. It
records which parameters were actually accepted (in the returned completion's
`_compat` attribute is NOT set — callers who need provenance should inspect
their own request). This keeps the baselines + judges backend-uniform.

We prefer detecting via the error response rather than hard-coding model-name
prefixes, because the model list changes faster than we can maintain a table.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def openai_chat_create(
    client: Any,
    *,
    model: str,
    messages: list[dict],
    max_output_tokens: int,
    temperature: float = 0.0,
    seed: int | None = 42,
    response_format: Any = None,
    **extra: Any,
) -> Any:
    """Call client.chat.completions.create (or .parse when response_format is a
    Pydantic model) with parameters adapted to the model generation.

    Strategy: attempt the modern parameter set first-fallback style. We try
    the call, and on a 400 that names an unsupported parameter, we drop/rename
    that parameter and retry. This handles:
      * max_tokens -> max_completion_tokens
      * temperature unsupported (drop it)
      * seed unsupported (drop it)
    """
    from openai import BadRequestError  # noqa: PLC0415

    use_parse = response_format is not None and hasattr(
        client.chat.completions, "parse"
    ) and _is_pydantic_model(response_format)

    # Start with the modern token param; most current models accept it.
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if seed is not None:
        params["seed"] = seed
    if response_format is not None:
        params["response_format"] = response_format
    params.update(extra)

    # Retry loop: drop/rename offending params based on the API error text.
    tried_legacy_tokens = False
    for _attempt in range(6):
        try:
            if use_parse:
                return client.chat.completions.parse(**params)
            return client.chat.completions.create(**params)
        except BadRequestError as e:
            msg = str(e).lower()
            if "max_completion_tokens" in msg and "max_tokens" in msg and not tried_legacy_tokens:
                # Model wants the legacy name
                params.pop("max_completion_tokens", None)
                params["max_tokens"] = max_output_tokens
                tried_legacy_tokens = True
                logger.info("openai_compat: switching to legacy max_tokens for %s", model)
                continue
            if "'temperature'" in msg or "temperature" in msg and "unsupported" in msg:
                if "temperature" in params:
                    params.pop("temperature")
                    logger.info("openai_compat: dropping temperature for %s", model)
                    continue
            if "'seed'" in msg or ("seed" in msg and "unsupported" in msg):
                if "seed" in params:
                    params.pop("seed")
                    logger.info("openai_compat: dropping seed for %s", model)
                    continue
            # Unhandled 400 — re-raise
            raise
    # Exhausted retries
    raise RuntimeError(f"openai_chat_create: could not find a working param set for {model}")


def _is_pydantic_model(obj: Any) -> bool:
    try:
        from pydantic import BaseModel  # noqa: PLC0415

        return isinstance(obj, type) and issubclass(obj, BaseModel)
    except ImportError:
        return False


__all__ = ["openai_chat_create"]
