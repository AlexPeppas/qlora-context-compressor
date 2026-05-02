"""llm_loop.py — LLMConversationLoop: ContextCompressor wired to the Anthropic API.

``LLMConversationLoop`` is the main integration point between the progressive
context compressor and a real Claude conversation.  It manages:

* Maintaining a rolling context window via ``ContextCompressor``
* Calling the Anthropic Messages API with the compressed context
* Offloading large tool results to disk (via ``ToolStore``) before the next call
* Exposing per-turn stats: token counts, compression ratio, anchor count, etc.

Quick start::

    from compressor.llm_loop import LLMConversationLoop

    loop = LLMConversationLoop(token_budget=8192)
    reply = loop.chat("What is the capital of France?")
    print(reply)
    print(loop.get_stats())

API key
-------
The API key is resolved in the following order:
1. The ``api_key`` constructor argument.
2. The ``ANTHROPIC_API_KEY`` environment variable.
3. A ``.env`` file in the working directory (loaded automatically via
   ``python-dotenv``).

If none of these provide a key, the constructor raises ``ValueError``.
"""
from __future__ import annotations

import os
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv(".env.txt")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLMConversationLoop
# ---------------------------------------------------------------------------


class LLMConversationLoop:
    """
    High-level conversation loop that wires ``ContextCompressor`` to the
    Anthropic Messages API.

    Args:
        api_key:      Anthropic API key.  Falls back to the
                      ``ANTHROPIC_API_KEY`` env var (loaded via dotenv).
        model:        Claude model identifier.
        token_budget: Soft token budget for the context window.  The compressor
                      triggers background compression once 80% is used.
        base_dir:     Base directory for anchor / tool-result disk stores.
                      Defaults to the current working directory.
        max_tokens:   Maximum tokens to generate per assistant response.
        compress_always: If True, compress after every turn regardless of
                         budget (useful for testing).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        token_budget: int = 4096,
        base_dir: str = ".",
        max_tokens: int = 1024,
        compress_always: bool = False,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for LLMConversationLoop. "
                "Install it with: pip install anthropic"
            ) from exc

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Anthropic API key found.  Pass api_key= or set "
                "ANTHROPIC_API_KEY in your environment / .env file."
            )

        self.model = model
        self.max_tokens = max_tokens

        # Lazy import to keep the module importable without anthropic installed
        self._client = _anthropic.Anthropic(api_key=self.api_key)

        # Wire up the compressor — import here to avoid circular imports
        from compressor import ContextCompressor  # noqa: PLC0415
        self._cc = ContextCompressor(
            base_dir=base_dir,
            token_budget=token_budget,
            compress_always=compress_always,
        )

        # Track raw token totals for compression-ratio reporting
        self._raw_token_total: int = 0

    # ------------------------------------------------------------------
    # Public conversation API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Send a user message and return the assistant's reply.

        The method:
        1. Retrieves the current compressed context from the store.
        2. Appends the new user message.
        3. Calls the Anthropic Messages API.
        4. Adds both turns to the context store.
        5. Triggers async background compression.

        Args:
            user_message: The user's text input.

        Returns:
            The assistant's response text.
        """
        # 1. Get the current context (waits for any in-flight compression)
        context_messages = self._cc.get_context()

        # 2. Append the new user message for the API call
        api_messages = context_messages + [{"role": "user", "content": user_message}]

        # 3. Call the Anthropic API
        logger.debug(
            "Calling %s with %d context messages + 1 new user message",
            self.model, len(context_messages),
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_messages,
        )
        assistant_text = response.content[0].text

        # 4. Add both turns to the context store + extract anchors
        self._cc.add_turn("user", user_message)
        self._cc.add_turn("assistant", assistant_text)

        # Track raw token totals for compression ratio reporting
        self._raw_token_total += len(user_message) // 4 + len(assistant_text) // 4

        # 5. async_pipeline.on_turn_complete() is already called inside
        #    ContextCompressor.add_turn(), so nothing extra needed here.

        logger.debug("assistant replied (%d chars)", len(assistant_text))
        return assistant_text

    def chat_with_tool(
        self,
        user_message: str,
        tool_name: str,
        tool_result: str,
    ) -> str:
        """
        Record a tool result, inject a stub reference, then call ``chat()``.

        The full tool result is offloaded to disk (via ``ToolStore``) before
        the next LLM call, keeping large payloads out of the context window.
        A compact stub string is injected into ``user_message`` in its place.

        Args:
            user_message: The user's text (or agent's follow-up instruction).
            tool_name:    Name of the tool that produced the result.
            tool_result:  The full tool output text to offload.

        Returns:
            The assistant's response text.
        """
        # The tool result belongs to the *next* turn that will be added
        next_turn_idx = self._cc.store._next_turn_idx
        stub = self._cc.save_tool_result(next_turn_idx, tool_name, tool_result)
        logger.debug("tool result offloaded → stub: %s", stub)

        # Inject stub into the user message
        augmented_message = f"{user_message}\n\n{stub}"
        return self.chat(augmented_message)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """
        Return a snapshot of token counts, compression ratio, and store metrics.

        Returns a dict with keys:
            current_tokens      — tokens currently in the compressed context
            budget_tokens       — configured token budget
            utilization_pct     — percentage of budget used
            raw_tokens          — estimated uncompressed token total
            compression_ratio   — raw / compressed (1.0 = no compression yet)
            anchor_count        — number of turns with extracted anchor records
            tool_result_count   — number of tool results offloaded to disk
            jobs_scheduled      — background compression jobs scheduled so far
            jobs_completed      — background compression jobs completed so far
            avg_compress_ms     — average time per compression job (ms)
        """
        usage = self._cc.store.token_usage()

        # Compute raw vs effective tokens from segment data
        with self._cc.store._lock:
            raw_from_segments = sum(
                seg.raw_tokens for seg in self._cc.store._segments
            )
            effective_tokens = sum(
                seg.effective_tokens for seg in self._cc.store._segments
            )

        compression_ratio = round(
            raw_from_segments / max(1, effective_tokens), 2
        )

        anchor_count = len(self._cc.anchors.all_turn_indices())
        tool_result_count = len(self._cc.tools.list_all())
        pipeline_stats = self._cc.pipeline.stats()

        return {
            "current_tokens": usage["current_tokens"],
            "budget_tokens": usage["budget_tokens"],
            "utilization_pct": usage["utilization_pct"],
            "raw_tokens": raw_from_segments,
            "compression_ratio": compression_ratio,
            "anchor_count": anchor_count,
            "tool_result_count": tool_result_count,
            **pipeline_stats,
        }

    def shutdown(self) -> None:
        """Gracefully shut down the background compression thread pool."""
        self._cc.shutdown()
