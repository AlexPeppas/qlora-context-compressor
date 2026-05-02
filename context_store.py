"""context_store.py — Core data structure for the progressive context compressor.

Tracks conversation turns, compressed segments, compression depth per segment,
and a token budget counter. Implements the chronological ratio gradient:
recent turns (last RECENT_TURN_COUNT) use a lower max compression ratio (3:1),
older turns may be compressed at up to 10:1.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

RECENT_TURN_COUNT: int = 3      # Turns considered "recent" (low compression)
RECENT_MAX_RATIO: float = 3.0   # Max compression ratio for recent segments
OLD_MAX_RATIO: float = 10.0     # Max compression ratio for older segments
MAX_COMPRESSION_DEPTH: int = 2  # Segments at this depth are frozen (no more compression)

TOKEN_BUDGET: int = 4_000       # Soft token budget for the context window
BUDGET_THRESHOLD: float = 0.80  # Fraction at which budget_exceeded() fires


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single conversation turn (one role + content pair)."""

    turn_idx: int
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    estimated_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.estimated_tokens = _estimate_tokens(self.content)

    def as_message(self) -> dict[str, str]:
        """Return an OpenAI-style message dict."""
        return {"role": self.role, "content": self.content}


@dataclass
class Segment:
    """
    A contiguous block of one or more turns, potentially compressed.

    A Segment is the atomic unit of compression.  Each turn starts its life
    in its own Segment; the async pipeline may later merge adjacent old
    Segments and compress them together.

    Attributes:
        segment_id:        Unique monotonically-increasing identifier.
        turns:             The raw turns this segment covers.
        compressed_text:   The compressed representation, or None if verbatim.
        compression_depth: How many times this segment has been compressed.
                           Segments at depth >= MAX_COMPRESSION_DEPTH are frozen.
    """

    segment_id: int
    turns: list[Turn]
    compressed_text: str | None = None
    compression_depth: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        """True if the segment has reached its maximum compression depth."""
        return self.compression_depth >= MAX_COMPRESSION_DEPTH

    @property
    def effective_text(self) -> str:
        """The text actually injected into the LLM context."""
        if self.compressed_text is not None:
            return self.compressed_text
        return "\n".join(
            f"[{t.role.upper()} turn {t.turn_idx}]: {t.content}"
            for t in self.turns
        )

    @property
    def effective_tokens(self) -> int:
        """Estimated token count of the effective text."""
        return _estimate_tokens(self.effective_text)

    @property
    def raw_tokens(self) -> int:
        """Estimated token count of all raw turns combined."""
        return sum(t.estimated_tokens for t in self.turns)

    @property
    def compression_ratio(self) -> float:
        """Actual compression achieved so far (1.0 means no compression)."""
        if self.raw_tokens == 0:
            return 1.0
        return self.raw_tokens / max(1, self.effective_tokens)

    def max_allowed_ratio(self, is_recent: bool) -> float:
        """Return the max compression ratio allowed given the segment's age."""
        return RECENT_MAX_RATIO if is_recent else OLD_MAX_RATIO

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def apply_compression(self, compressed_text: str) -> None:
        """
        Update the segment with a new compressed representation.

        Increments compression_depth.  Callers should check is_frozen before
        calling this.
        """
        self.compressed_text = compressed_text
        self.compression_depth += 1


# ---------------------------------------------------------------------------
# Context store
# ---------------------------------------------------------------------------

class ContextStore:
    """
    Central data structure for the progressive context compressor.

    Maintains an ordered list of Segments derived from conversation turns.
    Tracks total token usage against a configurable budget, and returns a
    mixed verbatim/compressed context representation for LLM consumption.

    Thread-safe: all public methods acquire an internal RLock.
    """

    def __init__(
        self,
        token_budget: int = TOKEN_BUDGET,
        budget_threshold: float = BUDGET_THRESHOLD,
        recent_turn_count: int = RECENT_TURN_COUNT,
    ) -> None:
        self._budget = token_budget
        self._threshold = budget_threshold
        self._recent_count = recent_turn_count
        self._segments: list[Segment] = []
        self._turns: list[Turn] = []
        self._next_turn_idx: int = 0
        self._next_segment_id: int = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> Turn:
        """
        Append a new conversation turn and wrap it in a fresh Segment.

        Each call creates one Turn and one single-turn Segment.  The async
        pipeline may later merge and compress segments in the background.

        Returns:
            The newly created Turn object.
        """
        with self._lock:
            turn = Turn(
                turn_idx=self._next_turn_idx,
                role=role,  # type: ignore[arg-type]
                content=content,
            )
            self._turns.append(turn)
            segment = Segment(
                segment_id=self._next_segment_id,
                turns=[turn],
            )
            self._segments.append(segment)
            self._next_turn_idx += 1
            self._next_segment_id += 1
            return turn

    def get_context_for_llm(self) -> list[dict[str, str]]:
        """
        Return the current context representation suitable for an LLM call.

        Recent segments (last RECENT_TURN_COUNT segments) are emitted verbatim
        as individual turn messages.  Older segments are emitted as a single
        compressed-history system message if they have been compressed, or
        verbatim if they have not yet been processed.

        Returns:
            A list of OpenAI-style {"role": ..., "content": ...} dicts.
        """
        with self._lock:
            messages: list[dict[str, str]] = []
            recent_cutoff = max(0, len(self._segments) - self._recent_count)

            for i, seg in enumerate(self._segments):
                is_recent = i >= recent_cutoff
                if is_recent or seg.compressed_text is None:
                    # Verbatim: emit each raw turn individually
                    for turn in seg.turns:
                        messages.append(turn.as_message())
                else:
                    # Compressed: emit a single system message
                    messages.append({
                        "role": "system",
                        "content": (
                            f"[COMPRESSED HISTORY — segment {seg.segment_id}, "
                            f"depth {seg.compression_depth}, "
                            f"ratio {seg.compression_ratio:.1f}x]:\n"
                            f"{seg.compressed_text}"
                        ),
                    })
            return messages

    def budget_exceeded(self) -> bool:
        """
        Return True if the current context token estimate exceeds the budget threshold.

        This is the signal that triggers background compression.
        """
        with self._lock:
            return self._total_tokens() > int(self._budget * self._threshold)

    def get_compressible_segments(self) -> list[tuple[Segment, bool]]:
        """
        Return all non-frozen segments eligible for compression.

        Returns:
            A list of (Segment, is_recent) tuples.  is_recent is True for
            segments within the last RECENT_TURN_COUNT segments.
        """
        with self._lock:
            recent_cutoff = max(0, len(self._segments) - self._recent_count)
            return [
                (seg, i >= recent_cutoff)
                for i, seg in enumerate(self._segments)
                if not seg.is_frozen
            ]

    def get_oldest_uncompressed_segment(self) -> tuple[Segment, bool] | None:
        """
        Return the oldest segment that has not yet been compressed and is not frozen.

        Returns:
            A (Segment, is_recent) tuple, or None if all segments are compressed
            or frozen.
        """
        with self._lock:
            recent_cutoff = max(0, len(self._segments) - self._recent_count)
            for i, seg in enumerate(self._segments):
                if seg.compressed_text is None and not seg.is_frozen:
                    return (seg, i >= recent_cutoff)
            return None

    def apply_compression(self, segment_id: int, compressed_text: str) -> bool:
        """
        Apply a compressed representation to the segment with the given ID.

        Args:
            segment_id:      The ID of the segment to update.
            compressed_text: The new compressed text.

        Returns:
            True if the update was applied; False if the segment was not found
            or is already frozen.
        """
        with self._lock:
            for seg in self._segments:
                if seg.segment_id == segment_id:
                    if seg.is_frozen:
                        return False
                    seg.apply_compression(compressed_text)
                    return True
            return False

    def token_usage(self) -> dict[str, int | float]:
        """
        Return a snapshot of current token usage and budget figures.

        Returns:
            Dict with keys: current_tokens, budget_tokens, threshold_tokens,
            segment_count, turn_count, utilization_pct.
        """
        with self._lock:
            current = self._total_tokens()
            return {
                "current_tokens": current,
                "budget_tokens": self._budget,
                "threshold_tokens": int(self._budget * self._threshold),
                "segment_count": len(self._segments),
                "turn_count": len(self._turns),
                "utilization_pct": round(100.0 * current / self._budget, 1),
            }

    def summary_line(self) -> str:
        """Return a one-line human-readable status string."""
        u = self.token_usage()
        compressed = sum(
            1 for s in self._segments if s.compressed_text is not None
        )
        return (
            f"turns={u['turn_count']}  segs={u['segment_count']}  "
            f"compressed={compressed}  "
            f"tokens={u['current_tokens']}/{u['budget_tokens']} "
            f"({u['utilization_pct']}%)  "
            f"budget_exceeded={self.budget_exceeded()}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _total_tokens(self) -> int:
        """Sum of effective tokens across all segments (must hold lock)."""
        return sum(seg.effective_tokens for seg in self._segments)
