"""Ingest public multi-turn conversation datasets into the structured-turn
schema (compressor.eval.conversations).

Track B corpus (v5 decision, 2026-07-29): the headline evaluation uses
PUBLIC, established, third-party multi-turn datasets — not our synthetic
corpus — to neutralize the "overfit to your own distribution" critique.

Per-dataset adapters convert raw rows into `Conversation` objects, all
passing through the same `validate()` gate. The adapters are pure
functions of a raw row dict, so they unit-test with fixtures and require
no network. The `--pull` CLI path uses the `datasets` library to stream
from the HF Hub and apply filtering.

Datasets (priority order — real-human first):
  * wildchat   allenai/WildChat-1M      ODC-BY     real user prompts + GPT
  * oasst2     OpenAssistant/oasst2     Apache-2.0 real human prompts+replies (tree)
  * mtbench    lmsys/mt_bench_...        CC-BY      curated, 2-turn (short)
  * ultrachat  HuggingFaceH4/ultrachat  MIT        synthetic-but-public (secondary)

Shared filters (all datasets): English, >= MIN_TURNS turns, total content
>= MIN_CHARS, ends (user, assistant) so holdout works, no per-turn PII flag.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .conversations import (
    Conversation,
    ConversationValidationError,
    Turn,
    validate,
    write_jsonl,
)

logger = logging.getLogger("ingest_corpus")

# Filtering thresholds (shared). A conversation must have at least
# MIN_TURNS turns so that holdout leaves a non-trivial prior context.
MIN_TURNS = 4  # user, assistant, user, assistant -> prior has >=2 turns after holdout
MIN_CHARS = 3000  # total content; short chats don't stress compression


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_role(role: str) -> str | None:
    """Map dataset-specific role strings to our {user, assistant} vocabulary.
    Returns None for roles we drop (e.g. system messages)."""
    r = role.strip().lower()
    if r in ("user", "human", "prompter"):
        return "user"
    if r in ("assistant", "gpt", "bot", "ai"):
        return "assistant"
    if r in ("system",):
        return None  # drop system turns; not part of the user/assistant dialogue
    return None


def _turns_from_role_content(
    pairs: Iterable[tuple[str, str]]
) -> tuple[Turn, ...]:
    """Build turns from (role, content) pairs, dropping unmapped roles and
    collapsing consecutive same-role turns (some datasets split a single
    assistant message across rows). Leading assistant turns are dropped so
    the conversation starts with 'user'."""
    turns: list[Turn] = []
    for role, content in pairs:
        norm = _normalise_role(role)
        if norm is None:
            continue
        content = (content or "").strip()
        if not content:
            continue
        if turns and turns[-1].role == norm:
            # merge consecutive same-role turns
            turns[-1] = Turn(role=norm, content=turns[-1].content + "\n\n" + content)
        else:
            turns.append(Turn(role=norm, content=content))  # type: ignore[arg-type]
    # Drop leading assistant turns
    while turns and turns[0].role != "user":
        turns.pop(0)
    # Drop a trailing user turn so we end (…, user, assistant)
    while turns and turns[-1].role != "assistant":
        turns.pop()
    return tuple(turns)


def _passes_filters(turns: tuple[Turn, ...]) -> bool:
    if len(turns) < MIN_TURNS:
        return False
    total_chars = sum(len(t.content) for t in turns)
    if total_chars < MIN_CHARS:
        return False
    # Must end (user, assistant) for holdout
    if turns[-1].role != "assistant" or turns[-2].role != "user":
        return False
    return True


# ---------------------------------------------------------------------------
# Per-dataset adapters (pure functions of a raw row)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterResult:
    conversation: Conversation | None
    skipped_reason: str = ""


def adapt_wildchat(row: dict[str, Any], *, idx: int) -> AdapterResult:
    """WildChat-1M: row has `conversation` = list of {role, content, language,
    toxic, ...}. Filter English + non-toxic at the conversation level."""
    if row.get("language") not in (None, "English", "english", "en"):
        return AdapterResult(None, f"non-english:{row.get('language')}")
    if row.get("toxic") is True:
        return AdapterResult(None, "toxic")
    conv = row.get("conversation") or []
    pairs = [(m.get("role", ""), m.get("content", "")) for m in conv]
    turns = _turns_from_role_content(pairs)
    if not _passes_filters(turns):
        return AdapterResult(None, "filtered")
    c = Conversation(
        id=f"wildchat-{row.get('conversation_hash', idx)}",
        scenario_type="wildchat",
        turns=turns,
        source_dataset="wildchat",
        source_metadata={
            "model": row.get("model", ""),
            "language": row.get("language", ""),
            "license": "ODC-BY",
        },
    )
    try:
        validate(c)
    except ConversationValidationError as e:
        return AdapterResult(None, f"invalid:{e}")
    return AdapterResult(c)


def adapt_ultrachat(row: dict[str, Any], *, idx: int) -> AdapterResult:
    """UltraChat_200k: row has `messages` = list of {role, content}."""
    msgs = row.get("messages") or []
    pairs = [(m.get("role", ""), m.get("content", "")) for m in msgs]
    turns = _turns_from_role_content(pairs)
    if not _passes_filters(turns):
        return AdapterResult(None, "filtered")
    c = Conversation(
        id=f"ultrachat-{row.get('prompt_id', idx)}",
        scenario_type="ultrachat",
        turns=turns,
        source_dataset="ultrachat",
        source_metadata={"license": "MIT", "synthetic": True},
    )
    try:
        validate(c)
    except ConversationValidationError as e:
        return AdapterResult(None, f"invalid:{e}")
    return AdapterResult(c)


def adapt_oasst_thread(
    thread: list[dict[str, Any]], *, tree_id: str
) -> AdapterResult:
    """OASST2: a thread is a root-to-leaf path already linearised (list of
    message dicts with `role` in {prompter, assistant} and `text`). We
    linearise the tree upstream in `_walk_oasst_trees`."""
    pairs = [(m.get("role", ""), m.get("text", "")) for m in thread]
    turns = _turns_from_role_content(pairs)
    if not _passes_filters(turns):
        return AdapterResult(None, "filtered")
    c = Conversation(
        id=f"oasst2-{tree_id}",
        scenario_type="oasst2",
        turns=turns,
        source_dataset="oasst2",
        source_metadata={"license": "Apache-2.0"},
    )
    try:
        validate(c)
    except ConversationValidationError as e:
        return AdapterResult(None, f"invalid:{e}")
    return AdapterResult(c)


def adapt_mtbench(row: dict[str, Any], *, idx: int) -> AdapterResult:
    """MT-Bench: multi-turn but short (2 user turns). row format varies by
    mirror; we accept a `turns` list of user strings + a `reference`/answer
    structure, OR a pre-built messages list. Here we handle the common
    `{"prompt": [u1, u2], "reference": [a1, a2]}`-style shape."""
    if "messages" in row:
        pairs = [(m.get("role", ""), m.get("content", "")) for m in row["messages"]]
    elif "prompt" in row and "reference" in row:
        # interleave user prompts and reference answers
        pairs = []
        for u, a in zip(row["prompt"], row["reference"]):
            pairs.append(("user", u))
            pairs.append(("assistant", a))
    else:
        return AdapterResult(None, "unrecognised-mtbench-shape")
    turns = _turns_from_role_content(pairs)
    if not _passes_filters(turns):
        return AdapterResult(None, "filtered")
    c = Conversation(
        id=f"mtbench-{row.get('question_id', idx)}",
        scenario_type="mtbench",
        turns=turns,
        source_dataset="mtbench",
        source_metadata={"license": "CC-BY-4.0", "category": row.get("category", "")},
    )
    try:
        validate(c)
    except ConversationValidationError as e:
        return AdapterResult(None, f"invalid:{e}")
    return AdapterResult(c)


ADAPTERS: dict[str, Callable] = {
    "wildchat": adapt_wildchat,
    "ultrachat": adapt_ultrachat,
    "mtbench": adapt_mtbench,
    # oasst2 handled separately (tree walk)
}


# ---------------------------------------------------------------------------
# HF pull (network path — only used by the CLI)
# ---------------------------------------------------------------------------


def _walk_oasst_trees(dataset) -> Iterable[tuple[str, list[dict]]]:
    """Reconstruct root-to-best-leaf linear threads from the OASST2 message
    forest. Groups by message_tree_id, builds parent->children, walks from
    the root selecting the highest-ranked (rank==0) child at each step."""
    from collections import defaultdict

    by_tree: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        if row.get("lang") != "en":
            continue
        if row.get("tree_state") not in (None, "ready_for_export"):
            continue
        by_tree[row["message_tree_id"]].append(row)

    for tree_id, msgs in by_tree.items():
        by_id = {m["message_id"]: m for m in msgs}
        children: dict[str | None, list[dict]] = {}
        root = None
        for m in msgs:
            children.setdefault(m.get("parent_id"), []).append(m)
            if m.get("parent_id") is None:
                root = m
        if root is None:
            continue
        # Walk root -> best child
        thread: list[dict] = []
        node = root
        while node is not None:
            thread.append(node)
            kids = children.get(node["message_id"], [])
            if not kids:
                break
            # pick rank 0 if available, else first
            kids_sorted = sorted(
                kids, key=lambda k: (k.get("rank") is None, k.get("rank") or 0)
            )
            node = kids_sorted[0]
        yield tree_id, thread


def pull_and_ingest(
    dataset_name: str,
    out_path,
    *,
    limit: int | None = None,
    hf_split: str = "train",
) -> int:
    """Pull a dataset from HF, adapt + filter, write structured JSONL.
    Returns the number of conversations written. Requires `datasets`."""
    from datasets import load_dataset  # noqa: PLC0415

    kept: list[Conversation] = []
    n_seen = 0
    n_skipped = 0

    if dataset_name == "wildchat":
        ds = load_dataset("allenai/WildChat-1M", split=hf_split, streaming=True)
        for i, row in enumerate(ds):
            n_seen += 1
            res = adapt_wildchat(row, idx=i)
            if res.conversation:
                kept.append(res.conversation)
            else:
                n_skipped += 1
            if limit and len(kept) >= limit:
                break
    elif dataset_name == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for i, row in enumerate(ds):
            n_seen += 1
            res = adapt_ultrachat(row, idx=i)
            if res.conversation:
                kept.append(res.conversation)
            else:
                n_skipped += 1
            if limit and len(kept) >= limit:
                break
    elif dataset_name == "oasst2":
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        for tree_id, thread in _walk_oasst_trees(ds):
            n_seen += 1
            res = adapt_oasst_thread(thread, tree_id=tree_id)
            if res.conversation:
                kept.append(res.conversation)
            else:
                n_skipped += 1
            if limit and len(kept) >= limit:
                break
    elif dataset_name == "mtbench":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        for i, row in enumerate(ds):
            n_seen += 1
            res = adapt_mtbench(row, idx=i)
            if res.conversation:
                kept.append(res.conversation)
            else:
                n_skipped += 1
            if limit and len(kept) >= limit:
                break
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    write_jsonl(kept, out_path)
    logger.info(
        "%s: seen=%d kept=%d skipped=%d -> %s",
        dataset_name,
        n_seen,
        len(kept),
        n_skipped,
        out_path,
    )
    return len(kept)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--dataset",
        required=True,
        choices=["wildchat", "ultrachat", "oasst2", "mtbench"],
    )
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="Max conversations to keep")
    args = p.parse_args()
    pull_and_ingest(args.dataset, args.out, limit=args.limit)


if __name__ == "__main__":
    main()


__all__ = [
    "ADAPTERS",
    "AdapterResult",
    "MIN_CHARS",
    "MIN_TURNS",
    "adapt_mtbench",
    "adapt_oasst_thread",
    "adapt_ultrachat",
    "adapt_wildchat",
    "pull_and_ingest",
]
