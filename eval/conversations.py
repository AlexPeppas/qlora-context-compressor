"""Structured-conversation schema for bake-off corpora.

Replaces the original `{conversation: str}` schema where turns were stored
as a flat string with `[USER]:` / `[ASSISTANT]:` markers. The flat-string
format is fragile when conversation content contains role-like text in
code blocks, logs, or markdown quotes, and string-parsing it loses
information needed for M_downstream (last-assistant-turn holdout).

New schema, one JSONL row per conversation::

    {
        "id": "bakeoff-01-rust-async",
        "scenario_type": "coding",
        "domain_hint": "rust async runtime debugging",
        "turns": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "source_dataset": "synthetic_ood" | "wildchat" | "ultrachat" | ...,
        "source_metadata": {...}   # provenance fields specific to source corpus
    }

Invariants enforced by the validator:
  * len(turns) >= 2
  * roles alternate strictly (user, assistant, user, assistant, ...)
  * first turn is "user"
  * each content is non-empty after strip

The validator is shared between the migration script (one-off ETL) and the
ingest paths for Phase E corpora (WildChat, OASST2, etc.), so all sources
flow through the same schema gate before reaching the bake-off harness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Conversation:
    id: str
    scenario_type: str
    turns: tuple[Turn, ...]
    domain_hint: str = ""
    source_dataset: str = ""
    source_metadata: dict[str, Any] = field(default_factory=dict)

    # Derived views ------------------------------------------------------

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def last_assistant_turn(self) -> Turn | None:
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn
        return None

    @property
    def last_user_turn(self) -> Turn | None:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn
        return None

    def with_holdout(self) -> "tuple[Conversation, Turn, Turn]":
        """Return (prior_conv, last_user_turn, held_out_assistant_turn).

        Used by M_downstream: the compressor sees `prior_conv`, the
        continuation generator is then asked
            [compressed prior_conv] + [last_user_turn]
        and the judge scores the generated continuation against the
        `held_out_assistant_turn`.

        Raises ValueError if the conversation does not end with a
        user-then-assistant pair (so holdout is impossible).
        """
        if self.num_turns < 3:
            raise ValueError(
                f"Conversation {self.id!r} has only {self.num_turns} turns; "
                "need at least 3 (user, assistant, user, assistant) for holdout"
            )
        if self.turns[-1].role != "assistant" or self.turns[-2].role != "user":
            raise ValueError(
                f"Conversation {self.id!r} does not end with (user, assistant); "
                f"got ({self.turns[-2].role}, {self.turns[-1].role})"
            )
        prior = Conversation(
            id=self.id,
            scenario_type=self.scenario_type,
            turns=self.turns[:-2],
            domain_hint=self.domain_hint,
            source_dataset=self.source_dataset,
            source_metadata=dict(self.source_metadata),
        )
        return prior, self.turns[-2], self.turns[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scenario_type": self.scenario_type,
            "domain_hint": self.domain_hint,
            "turns": [t.to_dict() for t in self.turns],
            "source_dataset": self.source_dataset,
            "source_metadata": dict(self.source_metadata),
        }

    def flatten(self) -> str:
        """Render as the legacy flat-string format. Used by compressors that
        take a plain string input (all current baselines except future
        structured-input ones).
        """
        parts = []
        for t in self.turns:
            label = "[USER]" if t.role == "user" else "[ASSISTANT]"
            parts.append(f"{label}: {t.content}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ConversationValidationError(ValueError):
    pass


def validate(conv: Conversation) -> None:
    """Raise ConversationValidationError if invariants are violated."""
    if not conv.id:
        raise ConversationValidationError("conversation has empty id")
    if conv.num_turns < 2:
        raise ConversationValidationError(
            f"{conv.id}: must have at least 2 turns, got {conv.num_turns}"
        )
    if conv.turns[0].role != "user":
        raise ConversationValidationError(
            f"{conv.id}: first turn must be 'user', got {conv.turns[0].role!r}"
        )
    for i, turn in enumerate(conv.turns):
        expected = "user" if i % 2 == 0 else "assistant"
        if turn.role != expected:
            raise ConversationValidationError(
                f"{conv.id}: turn {i} role {turn.role!r} breaks alternation "
                f"(expected {expected!r})"
            )
        if not turn.content.strip():
            raise ConversationValidationError(
                f"{conv.id}: turn {i} ({turn.role}) has empty content"
            )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path | str) -> list[Conversation]:
    """Load structured-turn conversations from JSONL. Validates each row."""
    p = Path(path)
    out: list[Conversation] = []
    for i, line in enumerate(p.open(encoding="utf-8")):
        if not line.strip():
            continue
        d = json.loads(line)
        conv = _from_dict(d)
        validate(conv)
        out.append(conv)
    return out


def write_jsonl(convs: Iterable[Conversation], path: Path | str) -> None:
    """Validate every conversation and write to JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for conv in convs:
            validate(conv)
            fh.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")


def _from_dict(d: dict[str, Any]) -> Conversation:
    turns = tuple(Turn(role=t["role"], content=t["content"]) for t in d["turns"])
    return Conversation(
        id=d["id"],
        scenario_type=d.get("scenario_type", ""),
        turns=turns,
        domain_hint=d.get("domain_hint", ""),
        source_dataset=d.get("source_dataset", ""),
        source_metadata=d.get("source_metadata", {}) or {},
    )


# ---------------------------------------------------------------------------
# Legacy parser (for one-off migration only — do not use in new code paths)
# ---------------------------------------------------------------------------


def parse_legacy_flat_string(
    text: str, *, conv_id: str = "<unknown>"
) -> tuple[Turn, ...]:
    """Parse the legacy `[USER]: ... [ASSISTANT]: ...` flat-string format
    into structured turns.

    Defensive against the failure modes the rubber-duck flagged:
      * Role markers appearing inside content (e.g., a code block discussing
        chat APIs) would silently split a turn. We detect this by validating
        strict alternation after parsing and raising if violated.
      * Trailing whitespace or empty turns get stripped.

    Raises ConversationValidationError if the parsed turns don't satisfy
    the structured-conversation invariants. Caller should NOT silently
    swallow the error — bad parses mean bad downstream metrics.
    """
    USER_MARKER = "[USER]:"
    ASSIST_MARKER = "[ASSISTANT]:"

    # Walk the text, emitting (role, content) chunks at each marker
    chunks: list[tuple[Role, str]] = []
    remaining = text
    current_role: Role | None = None
    current_buf: list[str] = []

    def _flush() -> None:
        if current_role is not None:
            content = "".join(current_buf).strip()
            if content:
                chunks.append((current_role, content))

    i = 0
    while i < len(remaining):
        # Find the next marker (whichever comes first)
        u_idx = remaining.find(USER_MARKER, i)
        a_idx = remaining.find(ASSIST_MARKER, i)
        candidates = [(idx, role, marker) for idx, role, marker in [
            (u_idx, "user", USER_MARKER),
            (a_idx, "assistant", ASSIST_MARKER),
        ] if idx >= 0]
        if not candidates:
            current_buf.append(remaining[i:])
            i = len(remaining)
            break
        idx, role, marker = min(candidates, key=lambda x: x[0])
        # Tail before the marker belongs to the previous turn
        current_buf.append(remaining[i:idx])
        _flush()
        # Start a new chunk
        current_role = role  # type: ignore[assignment]
        current_buf = []
        i = idx + len(marker)
    _flush()

    turns = tuple(Turn(role=role, content=content) for role, content in chunks)

    # Validate via the same gate as the new schema
    conv = Conversation(id=conv_id, scenario_type="", turns=turns)
    validate(conv)
    return turns


# ---------------------------------------------------------------------------
# Helpers used by tests + bake-off harness
# ---------------------------------------------------------------------------


def iter_legacy_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream rows from a legacy `{conversation: str}` JSONL file."""
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        yield json.loads(line)
