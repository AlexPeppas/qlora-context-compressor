"""tool_store.py — Disk store for tool call results.

Tool results from the LLM pipeline can be very large (e.g., raw API payloads,
file contents, database query outputs).  Instead of keeping them in the main
context window, this store offloads them to disk and injects a compact stub
reference in their place.  The full result can be retrieved on demand.

Files are written to ``{base_dir}/tool_results/turn_{n}_{tool_name}.txt``.

Usage::

    store = ToolStore(base_dir=".")
    stub = store.save_tool_result(4, "search_db", big_json_string)
    # stub == "[Tool: search_db @ turn 4 — Returns 42 rows from customers table]"

    full = store.retrieve(4, "search_db")
    hits = store.search("customers")
"""
from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# One-line summary heuristic
# ---------------------------------------------------------------------------

def _one_line_summary(content: str, max_chars: int = 80) -> str:
    """
    Produce a short one-line summary of a tool result.

    Strategy (in order):
    1. Use the first non-empty line if it's short enough.
    2. Take the first sentence.
    3. Truncate to max_chars with an ellipsis.
    """
    stripped = content.strip()
    if not stripped:
        return "(empty result)"

    # Try first non-empty line
    first_line = next((ln.strip() for ln in stripped.splitlines() if ln.strip()), "")
    if first_line and len(first_line) <= max_chars:
        return first_line

    # Try first sentence (split on . ! ?)
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    first_sentence = sentences[0].strip() if sentences else stripped
    if len(first_sentence) <= max_chars:
        return first_sentence

    # Truncate
    return stripped[:max_chars - 1].rstrip() + "…"


def _sanitise_tool_name(name: str) -> str:
    """Replace characters unsafe for filenames with underscores."""
    return re.sub(r"[^\w\-]", "_", name)


# ---------------------------------------------------------------------------
# ToolStore
# ---------------------------------------------------------------------------

class ToolStore:
    """
    Disk store for tool call results with stub-reference injection.

    Keeps large tool payloads off the LLM context window by writing them to
    disk and returning a compact stub string.  The full payload can be
    retrieved later via ``retrieve()`` or discovered via ``search()``.

    Args:
        base_dir: Root directory under which the ``tool_results/`` folder lives.
    """

    def __init__(self, base_dir: str = ".") -> None:
        self._results_dir = Path(base_dir) / "tool_results"
        self._results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_tool_result(
        self,
        turn_idx: int,
        tool_name: str,
        result_content: str,
    ) -> str:
        """
        Write *result_content* to disk and return a compact stub string.

        The stub is suitable for injecting into the LLM context window in
        place of the full result.

        Args:
            turn_idx:       Index of the turn that produced this tool result.
            tool_name:      Name of the tool (e.g. ``"search_web"``).
            result_content: The full tool result text to offload.

        Returns:
            A stub string of the form
            ``[Tool: {tool_name} @ turn {turn_idx} — {one_line_summary}]``.
        """
        safe_name = _sanitise_tool_name(tool_name)
        path = self._results_dir / f"turn_{turn_idx}_{safe_name}.txt"
        path.write_text(result_content, encoding="utf-8")

        summary = _one_line_summary(result_content)
        return f"[Tool: {tool_name} @ turn {turn_idx} — {summary}]"

    def retrieve(self, turn_idx: int, tool_name: str) -> str | None:
        """
        Read and return the full tool result for a given turn and tool name.

        Args:
            turn_idx:  Index of the turn.
            tool_name: Name of the tool.

        Returns:
            The full result string, or None if no result is stored.
        """
        safe_name = _sanitise_tool_name(tool_name)
        path = self._results_dir / f"turn_{turn_idx}_{safe_name}.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def search(self, query: str) -> list[str]:
        """
        Search all stored tool result files for the given query string.

        Performs case-insensitive substring search over file *contents*.

        Args:
            query: The search string.

        Returns:
            A list of stub strings for matching results, sorted by filename.
        """
        query_lower = query.lower()
        stubs: list[str] = []

        for path in sorted(self._results_dir.glob("turn_*.txt")):
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue

            if query_lower in content.lower():
                # Reconstruct metadata from filename: turn_{idx}_{tool_name}.txt
                stem = path.stem  # e.g. "turn_4_search_db"
                parts = stem.split("_", 2)  # ["turn", "4", "search_db"]
                turn_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
                tool_name = parts[2] if len(parts) > 2 else stem
                summary = _one_line_summary(content)
                stubs.append(f"[Tool: {tool_name} @ turn {turn_idx} — {summary}]")

        return stubs

    def list_all(self) -> list[dict[str, str | int]]:
        """
        Return metadata for every stored tool result.

        Returns:
            A list of dicts with keys: turn_idx, tool_name, stub, size_chars.
        """
        results: list[dict[str, str | int]] = []
        for path in sorted(self._results_dir.glob("turn_*.txt")):
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            stem = path.stem
            parts = stem.split("_", 2)
            turn_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
            tool_name = parts[2] if len(parts) > 2 else stem
            summary = _one_line_summary(content)
            results.append({
                "turn_idx": turn_idx,
                "tool_name": tool_name,
                "stub": f"[Tool: {tool_name} @ turn {turn_idx} — {summary}]",
                "size_chars": len(content),
            })
        return results
