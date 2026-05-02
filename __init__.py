"""compressor — Local progressive asynchronous context compressor.

Public API
----------
The primary entry point is ``ContextCompressor``, which wires together all
sub-components: the context store, anchor store, tool store, heuristic
compressor, and async compression pipeline.

Quick start::

    from compressor import ContextCompressor

    cc = ContextCompressor(base_dir=".", token_budget=4_000)

    # Add turns (user / assistant / tool)
    cc.add_turn("user", "What is the Q3 revenue forecast?")
    cc.add_turn("assistant", "Based on current trends the forecast is $2.4M…")

    # Offload a large tool result
    stub = cc.save_tool_result(1, "crm_query", big_json_string)
    # → "[Tool: crm_query @ turn 1 — Returns 87 rows from pipeline]"

    # Get context ready for the next LLM call (waits for in-flight compression)
    messages = cc.get_context()

    # Inspect compression state
    print(cc.status())
    print(cc.store.summary_line())

Architecture
------------
``ContextStore``    — ordered turns + segments + budget tracking
``AnchorStore``     — append-only disk store for extracted facts
``ToolStore``       — disk store for large tool results
``HeuristicCompressor`` — extractive summariser (LexRank / fallback)
``AsyncPipeline``   — background thread that compresses old segments
"""
from __future__ import annotations

from .anchor_store import AnchorStore
from .async_pipeline import AsyncPipeline
from .compressor import HeuristicCompressor
from .context_store import ContextStore
from .tool_store import ToolStore

__all__ = [
    "ContextCompressor",
    "AnchorStore",
    "AsyncPipeline",
    "ContextStore",
    "HeuristicCompressor",
    "ToolStore",
]


class ContextCompressor:
    """
    High-level façade that wires all compressor sub-components together.

    This is the class most callers should instantiate directly.

    Args:
        base_dir:        Root directory for disk stores (anchors/, tool_results/).
                         Defaults to the current working directory.
        token_budget:    Soft token budget for the context window.  Compression
                         is triggered when usage exceeds 80% of this value.
        compress_always: If True, compression is scheduled after every turn
                         regardless of budget.  Useful for demos and testing.
        prefer_lsa:      If True, use sumy's LSA summariser instead of LexRank.
    """

    def __init__(
        self,
        base_dir: str = ".",
        token_budget: int = 4_000,
        compress_always: bool = False,
        prefer_lsa: bool = False,
    ) -> None:
        self.store = ContextStore(token_budget=token_budget)
        self.anchors = AnchorStore(base_dir=base_dir)
        self.tools = ToolStore(base_dir=base_dir)
        self._compressor = HeuristicCompressor(prefer_lsa=prefer_lsa)
        self.pipeline = AsyncPipeline(
            context_store=self.store,
            compressor=self._compressor,
            compress_always=compress_always,
        )

    # ------------------------------------------------------------------
    # Core conversation loop methods
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> int:
        """
        Add a conversation turn, extract anchors, and schedule compression.

        Args:
            role:    Message role: ``"user"``, ``"assistant"``, ``"system"``,
                     or ``"tool"``.
            content: The text content of the turn.

        Returns:
            The integer turn index.
        """
        turn = self.store.add_turn(role, content)
        self.anchors.extract_and_save(turn.turn_idx, content)
        self.pipeline.on_turn_complete(turn.turn_idx)
        return turn.turn_idx

    def save_tool_result(
        self,
        turn_idx: int,
        tool_name: str,
        result_content: str,
    ) -> str:
        """
        Offload a large tool result to disk and return a stub reference.

        The stub is a short string suitable for injecting into the LLM context
        in place of the full result.

        Args:
            turn_idx:       Index of the turn that produced the result.
            tool_name:      Name of the tool (e.g. ``"search_web"``).
            result_content: The full tool output.

        Returns:
            A stub string, e.g.
            ``"[Tool: search_web @ turn 3 — Returns top 10 results for 'revenue']"``.
        """
        return self.tools.save_tool_result(turn_idx, tool_name, result_content)

    def get_context(self) -> list[dict[str, str]]:
        """
        Return the current context representation ready for an LLM call.

        Waits for any in-flight background compression to finish before
        assembling the context, ensuring the most compressed state is returned.

        Returns:
            A list of OpenAI-style ``{"role": ..., "content": ...}`` dicts.
        """
        self.pipeline.wait_if_needed()
        return self.store.get_context_for_llm()

    # ------------------------------------------------------------------
    # Search / retrieval
    # ------------------------------------------------------------------

    def search_anchors(self, query: str) -> list[dict]:
        """Search all extracted anchor records for *query*."""
        return self.anchors.search(query)

    def search_tools(self, query: str) -> list[str]:
        """Search all stored tool results for *query* and return stub strings."""
        return self.tools.search(query)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """
        Return a combined status snapshot.

        Includes token usage, pipeline metrics, and compressor method.
        """
        usage = self.store.token_usage()
        usage["budget_exceeded"] = self.store.budget_exceeded()  # type: ignore[assignment]
        usage.update(self.pipeline.stats())
        usage["compressor_method"] = self._compressor._method
        return usage

    def shutdown(self) -> None:
        """Gracefully shut down the background compression thread pool."""
        self.pipeline.shutdown(wait=True)
