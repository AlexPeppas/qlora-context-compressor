"""
Baseline compressors for the Phase E head-to-head bake-off.

All baselines implement the `Baseline` Protocol so the bake-off harness can
call any of them through one shared interface. The unit of work is a single
(conversation, target_ratio, turn_age) -> CompressionResult call.

Implementations live in sibling modules:

    base_qwen.py    : Zero-shot prompted Qwen2.5-7B-Instruct (GPU)
    qwen_lora.py    : Our tier-conditioned QLoRA adapter (GPU)
    lingua.py       : LLMLingua-2 + LongLLMLingua extractive (GPU, smaller model)
    frontier.py     : GPT-4o via OpenAI API (no GPU, network only)

Execution model is hybrid: GPU baselines run on a RunPod 4090; the frontier
baseline runs from the local laptop. Both write to the same JSONL schema so
results can be merged with simple file concatenation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class CompressionRequest:
    """Single unit of work for a baseline compressor.

    `conversation` is the full source text the compressor must compress
    (all prior turns concatenated as plain text, NOT a chat template).
    `turn_age` and `target_ratio` define the compression budget;
    extractive baselines that ignore turn_age must still receive it
    so we can record the intended tier.
    """

    conversation: str
    turn_age: str  # "recent" | "mid" | "old"
    target_ratio: int  # e.g. 3, 5, 10
    conversation_id: str = ""
    scenario_type: str = ""

    @property
    def target_compression_rate(self) -> float:
        """Cmprsr / LLMLingua convention: rate = 1/ratio (fraction kept)."""
        return 1.0 / float(self.target_ratio)


@dataclass(frozen=True)
class CompressionResult:
    """Output of a single compressor.compress() call.

    Schema mirrors `data/bakeoff_results_eosfix.jsonl` so existing tooling keeps
    working. Extra fields (stop_reason, generated_new_tokens) are populated when
    the baseline can report them; left empty otherwise.
    """

    # Identifying fields (echoed from the request)
    conversation_id: str
    scenario_type: str
    turn_age: str
    target_ratio: int
    source: str  # baseline name, e.g. "tfix375", "llmlingua2", "gpt-4o"

    # Content
    compressed: str

    # Surface metrics — always populated
    input_chars: int
    output_chars: int
    achieved_ratio: float  # input_chars / max(output_chars, 1)
    gen_seconds: float

    # Token-level metrics — populated when the baseline knows them
    # (transformers-based baselines populate these; extractive / API baselines
    # populate as best-effort or leave None)
    input_tokens: int | None = None
    output_tokens: int | None = None
    max_new_tokens: int | None = None
    stop_reason: str | None = None  # "eos" | "max_new_tokens" | "stop_sequence" | None
    stopped_on_eos: bool | None = None

    # Free-form extras — model-specific debugging info, judge metadata, etc.
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Serialize for the bake-off JSONL. Drops None-valued optional fields
        so legacy rows stay schema-compatible with the eosfix bake-off."""
        d = asdict(self)
        for k in (
            "input_tokens",
            "output_tokens",
            "max_new_tokens",
            "stop_reason",
            "stopped_on_eos",
        ):
            if d[k] is None:
                d.pop(k)
        if not d["extras"]:
            d.pop("extras")
        return d


@runtime_checkable
class Baseline(Protocol):
    """Protocol for a single-shot compressor.

    Implementations should be stateless across requests once `__init__` /
    `load()` is done — i.e. concurrent requests on the same instance must be
    safe in single-threaded use, and re-using the instance across many
    `compress()` calls must not leak memory or accumulate state.

    Heavy resources (model weights, API clients) should be allocated lazily
    in `load()` and released in `unload()` so the bake-off harness can swap
    GPU baselines in/out without exhausting VRAM.
    """

    name: str  # short identifier used as the `source` field in JSONL output

    def load(self) -> None:
        """Allocate heavy resources (load model weights, open API client)."""
        ...

    def unload(self) -> None:
        """Release heavy resources. Must be safe to call after a partial load."""
        ...

    def compress(self, request: CompressionRequest) -> CompressionResult:
        """Compress one conversation. Must be deterministic for greedy/temperature=0
        baselines so re-runs reproduce results exactly."""
        ...


# Public re-exports — consumers import from `compressor.baselines`
__all__ = [
    "Baseline",
    "CompressionRequest",
    "CompressionResult",
]
