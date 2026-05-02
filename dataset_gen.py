"""dataset_gen.py — Synthetic training-pair generator for the context compressor.

Uses ``claude-sonnet-4-6`` as a teacher LLM to produce ~1,000 examples (67 seeds × 5 types × 3 ratios)
of (original, compressed) conversation pairs across five scenario types (coding, research, support,
tool_heavy, analysis).

For each scenario type the generator:
1. Calls Claude to produce SEEDS_PER_TYPE realistic multi-turn conversations
   (800-2000 tokens each).
2. For every seed, calls Claude once more requesting compressions at 3x / 5x /
   10x ratios together with extracted anchors and tool stubs.  The result is
   decoded from JSON and split into three separate training examples.

Output files
------------
``compressor/data/synthetic_dataset.jsonl``
    One JSON object per line with keys:
        id, scenario_type, turn_age, target_ratio, original, compressed,
        anchors, tool_stubs

``compressor/data/dataset_stats.json``
    Summary statistics for the generated dataset.

Run directly
------------
    python -m compressor.dataset_gen

Or with custom concurrency / output dir::

    python -m compressor.dataset_gen --workers 4 --out-dir compressor/data
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIO_TYPES: list[str] = ["coding", "research", "support", "tool_heavy", "analysis"]
SEEDS_PER_TYPE: int = 67         # 5 types × 67 seeds × 3 ratios ≈ 1,005 examples
DEFAULT_MODEL: str = "claude-sonnet-4-6"
DEFAULT_OUT_DIR: str = str(Path(__file__).parent / "data")

RATIO_KEYS: list[tuple[str, float, str]] = [
    ("compressed_3x",  3.0,  "recent"),
    ("compressed_5x",  5.0,  "mid"),
    ("compressed_10x", 10.0, "old"),
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SEED_PROMPT = """\
Generate a realistic, detailed multi-turn LLM conversation for the scenario type: **{scenario_type}**.

Requirements:
- 4 to 8 turns, alternating [USER] and [ASSISTANT] roles
- 800-2000 tokens total (be thorough)
- Include specific names, numbers, dates, decisions, and technical details
- For "tool_heavy" scenarios include [TOOL CALL: tool_name] and [TOOL RESULT: ...] blocks
- Make it feel like a real production conversation

Scenario descriptions:
  coding      — developer debugging, code review, architecture decisions
  research    — literature review, hypothesis exploration, data analysis
  support     — customer/IT support ticket escalation and resolution
  tool_heavy  — agent workflow with multiple tool calls (CRM, search, APIs)
  analysis    — business or financial analysis with numbers, charts, decisions

Format each turn exactly as:
[USER]: <message>
[ASSISTANT]: <response>

Return ONLY the conversation — no preamble, no commentary.
"""

_COMPRESS_PROMPT = """\
You are an expert context-compression assistant. Compress the conversation below
into three progressively shorter summaries representing different turn-age tiers.

CONVERSATION:
{conversation}

Original length: {orig_chars} characters.

Produce exactly three compressions. Treat the character targets as hard budgets —
hit them as closely as possible (within ±10%). Each version must be strictly
shorter than the previous one.

compressed_3x  [RECENT TURN — target {target_3x} characters]
  This turn just happened; it is still active context. Preserve ALL key facts,
  numbers, named entities, decisions, code snippets, error messages, and the
  full reasoning arc. The reader must be able to continue the conversation from
  this summary alone without asking clarifying questions.

compressed_5x  [MID-AGE TURN — target {target_5x} characters]
  This turn is a few exchanges back; context is warm but not immediate. Preserve
  MOST key facts and the main narrative thread. Drop elaboration, worked examples
  used only to explain, and tangents. Keep every decision, specific numeric value,
  named entity, and any unresolved issue that may resurface.

compressed_10x  [OLD TURN — target {target_10x} characters]
  This turn is early context; only its final outcomes still matter. Keep ONLY
  the critical result: final decisions made, key constraints established, named
  entities still referenced downstream, and anything the current conversation
  causally depends on. No reasoning, no intermediate steps, no alternatives
  that were rejected.

Also produce:
anchors    — 4-10 short strings capturing key facts (numbers, names, decisions,
             states) that must survive ANY level of compression
tool_stubs — for any tool calls: "[Tool: <name> @ turn <N> — <one-line summary>]"
             (empty list if no tool calls)

Return a single valid JSON object with EXACTLY these five keys:
{{
  "compressed_3x": "...",
  "compressed_5x": "...",
  "compressed_10x": "...",
  "anchors": ["...", "..."],
  "tool_stubs": []
}}

Return ONLY the JSON — no markdown fences, no explanation.
"""


# ---------------------------------------------------------------------------
# DatasetGenerator
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    Generates synthetic (original, compressed) training pairs using Claude as
    the teacher LLM.

    Args:
        api_key: Anthropic API key.  Defaults to the ``ANTHROPIC_API_KEY``
                 environment variable (which can be loaded from a ``.env`` file
                 via ``load_dotenv()``).
        model:   Claude model to use as teacher.
        out_dir: Directory where output files are written.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        out_dir: str = DEFAULT_OUT_DIR,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for DatasetGenerator. "
                "Install it with: pip install anthropic"
            ) from exc

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Anthropic API key found.  Set ANTHROPIC_API_KEY in your "
                "environment or in a .env file."
            )
        self.model = model
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._client = _anthropic.Anthropic(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, workers: int = 3) -> list[dict[str, Any]]:
        """
        Generate all training examples.

        Args:
            workers: Number of parallel threads for API calls.

        Returns:
            List of training example dicts.
        """
        tasks: list[tuple[str, int]] = [
            (scenario, seed_idx)
            for scenario in SCENARIO_TYPES
            for seed_idx in range(SEEDS_PER_TYPE)
        ]
        total_tasks = len(tasks)
        examples: list[dict[str, Any]] = []
        errors: int = 0

        print(
            f"Generating {total_tasks} seed conversations "
            f"({total_tasks * 3} examples total) "
            f"with {workers} parallel workers…"
        )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._process_seed, scenario, seed_idx): (scenario, seed_idx)
                for scenario, seed_idx in tasks
            }
            for i, future in enumerate(as_completed(futures), 1):
                scenario, seed_idx = futures[future]
                try:
                    seed_examples = future.result()
                    examples.extend(seed_examples)
                    print(
                        f"  [{i:02d}/{total_tasks}] {scenario} seed {seed_idx + 1} "
                        f"→ {len(seed_examples)} examples"
                    )
                except Exception as exc:
                    errors += 1
                    logger.warning(
                        "Failed to process %s seed %d: %s", scenario, seed_idx, exc
                    )
                    print(
                        f"  [{i:02d}/{total_tasks}] {scenario} seed {seed_idx + 1} "
                        f"→ ERROR: {exc}"
                    )

        print(
            f"\nDone.  {len(examples)} examples generated, {errors} seed(s) failed."
        )
        return examples

    def save(self, examples: list[dict[str, Any]]) -> None:
        """
        Write examples to JSONL and stats to JSON.

        Args:
            examples: List of training example dicts from ``generate()``.
        """
        # JSONL
        jsonl_path = self.out_dir / "synthetic_dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Dataset written to: {jsonl_path}  ({len(examples)} examples)")

        # Stats
        stats = self._compute_stats(examples)
        stats_path = self.out_dir / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        print(f"Stats written to:   {stats_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_seed(
        self, scenario_type: str, seed_idx: int
    ) -> list[dict[str, Any]]:
        """Generate one seed conversation and return three training examples."""
        conversation = self._generate_seed_conversation(scenario_type)
        compressions = self._compress_at_ratios(conversation)
        examples: list[dict[str, Any]] = []
        for ratio_key, target_ratio, turn_age in RATIO_KEYS:
            compressed = compressions.get(ratio_key, "")
            if not compressed:
                logger.warning(
                    "Empty compressed_%s for %s seed %d — skipping",
                    ratio_key, scenario_type, seed_idx,
                )
                continue
            examples.append(
                {
                    "id": str(uuid.uuid4()),
                    "scenario_type": scenario_type,
                    "turn_age": turn_age,
                    "target_ratio": target_ratio,
                    "original": conversation,
                    "compressed": compressed,
                    "anchors": compressions.get("anchors", []),
                    "tool_stubs": compressions.get("tool_stubs", []),
                }
            )
        return examples

    def _generate_seed_conversation(self, scenario_type: str) -> str:
        """Call Claude to produce a realistic multi-turn conversation."""
        prompt = _SEED_PROMPT.format(scenario_type=scenario_type)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _compress_at_ratios(self, conversation: str) -> dict[str, Any]:
        """
        Ask Claude to compress a conversation at 3x / 5x / 10x and extract
        anchors + tool stubs in a single API call.

        Returns the parsed JSON dict.  Raises on JSON decode failure.
        """
        orig_chars = len(conversation)
        prompt = _COMPRESS_PROMPT.format(
            conversation=conversation,
            orig_chars=orig_chars,
            target_3x=orig_chars // 3,
            target_5x=orig_chars // 5,
            target_10x=orig_chars // 10,
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if Claude wrapped the JSON anyway
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Claude returned non-JSON compression response: {raw[:200]}"
            ) from exc

    @staticmethod
    def _compute_stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute summary statistics over the generated dataset."""
        by_scenario: dict[str, int] = {s: 0 for s in SCENARIO_TYPES}
        by_ratio: dict[str, int] = {"3.0": 0, "5.0": 0, "10.0": 0}
        by_turn_age: dict[str, int] = {"recent": 0, "mid": 0, "old": 0}
        total_orig_chars: int = 0
        total_comp_chars: int = 0

        for ex in examples:
            by_scenario[ex["scenario_type"]] = by_scenario.get(ex["scenario_type"], 0) + 1
            ratio_str = str(ex["target_ratio"])
            by_ratio[ratio_str] = by_ratio.get(ratio_str, 0) + 1
            by_turn_age[ex["turn_age"]] = by_turn_age.get(ex["turn_age"], 0) + 1
            total_orig_chars += len(ex.get("original", ""))
            total_comp_chars += len(ex.get("compressed", ""))

        avg_orig = total_orig_chars // max(1, len(examples))
        avg_comp = total_comp_chars // max(1, len(examples))
        avg_ratio = round(total_orig_chars / max(1, total_comp_chars), 2)

        return {
            "total_examples": len(examples),
            "by_scenario_type": by_scenario,
            "by_target_ratio": by_ratio,
            "by_turn_age": by_turn_age,
            "avg_original_chars": avg_orig,
            "avg_compressed_chars": avg_comp,
            "avg_compression_ratio": avg_ratio,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic context-compression training data using Claude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Claude model to use as the teacher LLM.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel API-call threads.",
    )
    parser.add_argument(
        "--seeds-per-type",
        type=int,
        default=SEEDS_PER_TYPE,
        help="Number of seed conversations to generate per scenario type.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )
    args = _parse_args()

    # Allow overriding seeds-per-type at runtime
    if args.seeds_per_type != SEEDS_PER_TYPE:
        SEEDS_PER_TYPE = args.seeds_per_type  # type: ignore[misc]

    gen = DatasetGenerator(model=args.model, out_dir=args.out_dir)
    examples = gen.generate(workers=args.workers)
    if examples:
        gen.save(examples)
    else:
        print("No examples generated — check API key and network.", file=sys.stderr)
        sys.exit(1)
