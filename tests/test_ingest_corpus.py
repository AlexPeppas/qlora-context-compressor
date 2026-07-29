"""Tests for compressor.eval.ingest_corpus — public-dataset adapters.

Fixture-based, no network. Validates that each adapter:
  * converts raw dataset rows into valid structured Conversations
  * applies the shared filters (min turns, min chars, ends user->assistant)
  * normalises roles and merges consecutive same-role turns
  * records license + source provenance
"""
from __future__ import annotations

from compressor.eval.ingest_corpus import (
    MIN_CHARS,
    adapt_mtbench,
    adapt_oasst_thread,
    adapt_ultrachat,
    adapt_wildchat,
)


def _long(text: str, n: int = 900) -> str:
    """Pad content so the conversation clears MIN_CHARS."""
    return (text + " ") * n


# ---------------------------------------------------------------------------
# WildChat
# ---------------------------------------------------------------------------


def test_wildchat_basic_accept():
    row = {
        "conversation_hash": "abc",
        "language": "English",
        "toxic": False,
        "model": "gpt-4",
        "conversation": [
            {"role": "user", "content": _long("u1")},
            {"role": "assistant", "content": _long("a1")},
            {"role": "user", "content": _long("u2")},
            {"role": "assistant", "content": _long("a2")},
        ],
    }
    res = adapt_wildchat(row, idx=0)
    assert res.conversation is not None
    c = res.conversation
    assert c.id == "wildchat-abc"
    assert c.num_turns == 4
    assert c.source_dataset == "wildchat"
    assert c.source_metadata["license"] == "ODC-BY"
    # holdout works
    prior, last_user, held = c.with_holdout()
    assert last_user.role == "user" and held.role == "assistant"


def test_wildchat_rejects_non_english():
    row = {
        "language": "Chinese",
        "toxic": False,
        "conversation": [
            {"role": "user", "content": _long("u1")},
            {"role": "assistant", "content": _long("a1")},
            {"role": "user", "content": _long("u2")},
            {"role": "assistant", "content": _long("a2")},
        ],
    }
    res = adapt_wildchat(row, idx=0)
    assert res.conversation is None
    assert "non-english" in res.skipped_reason


def test_wildchat_rejects_toxic():
    row = {
        "language": "English",
        "toxic": True,
        "conversation": [
            {"role": "user", "content": _long("u1")},
            {"role": "assistant", "content": _long("a1")},
            {"role": "user", "content": _long("u2")},
            {"role": "assistant", "content": _long("a2")},
        ],
    }
    res = adapt_wildchat(row, idx=0)
    assert res.conversation is None
    assert res.skipped_reason == "toxic"


def test_wildchat_rejects_too_short():
    row = {
        "language": "English",
        "toxic": False,
        "conversation": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "later"},
        ],
    }
    res = adapt_wildchat(row, idx=0)
    assert res.conversation is None
    assert res.skipped_reason == "filtered"


def test_wildchat_drops_trailing_user_turn():
    # 5 turns ending on user -> should trim to 4 ending on assistant
    row = {
        "conversation_hash": "xyz",
        "language": "English",
        "toxic": False,
        "conversation": [
            {"role": "user", "content": _long("u1")},
            {"role": "assistant", "content": _long("a1")},
            {"role": "user", "content": _long("u2")},
            {"role": "assistant", "content": _long("a2")},
            {"role": "user", "content": _long("u3")},  # trailing, dropped
        ],
    }
    res = adapt_wildchat(row, idx=0)
    assert res.conversation is not None
    assert res.conversation.num_turns == 4
    assert res.conversation.turns[-1].role == "assistant"


# ---------------------------------------------------------------------------
# UltraChat
# ---------------------------------------------------------------------------


def test_ultrachat_basic_accept():
    row = {
        "prompt_id": "p123",
        "messages": [
            {"role": "user", "content": _long("u1")},
            {"role": "assistant", "content": _long("a1")},
            {"role": "user", "content": _long("u2")},
            {"role": "assistant", "content": _long("a2")},
        ],
    }
    res = adapt_ultrachat(row, idx=0)
    assert res.conversation is not None
    assert res.conversation.source_metadata["synthetic"] is True
    assert res.conversation.source_metadata["license"] == "MIT"


# ---------------------------------------------------------------------------
# OASST2
# ---------------------------------------------------------------------------


def test_oasst_thread_accept():
    thread = [
        {"role": "prompter", "text": _long("u1")},
        {"role": "assistant", "text": _long("a1")},
        {"role": "prompter", "text": _long("u2")},
        {"role": "assistant", "text": _long("a2")},
    ]
    res = adapt_oasst_thread(thread, tree_id="t1")
    assert res.conversation is not None
    assert res.conversation.id == "oasst2-t1"
    assert res.conversation.source_metadata["license"] == "Apache-2.0"
    # prompter mapped to user
    assert res.conversation.turns[0].role == "user"


# ---------------------------------------------------------------------------
# MT-Bench
# ---------------------------------------------------------------------------


def test_mtbench_prompt_reference_shape():
    row = {
        "question_id": 81,
        "category": "writing",
        "prompt": [_long("u1"), _long("u2")],
        "reference": [_long("a1"), _long("a2")],
    }
    res = adapt_mtbench(row, idx=0)
    assert res.conversation is not None
    assert res.conversation.num_turns == 4
    assert res.conversation.id == "mtbench-81"


def test_mtbench_unrecognised_shape():
    row = {"question_id": 1, "foo": "bar"}
    res = adapt_mtbench(row, idx=0)
    assert res.conversation is None
    assert res.skipped_reason == "unrecognised-mtbench-shape"


# ---------------------------------------------------------------------------
# Role normalisation + merging
# ---------------------------------------------------------------------------


def test_consecutive_same_role_merged():
    # WildChat sometimes splits an assistant message across entries
    row = {
        "conversation_hash": "m",
        "language": "English",
        "toxic": False,
        "conversation": [
            {"role": "user", "content": _long("u1")},
            {"role": "assistant", "content": _long("a1a")},
            {"role": "assistant", "content": _long("a1b")},  # merge with prev
            {"role": "user", "content": _long("u2")},
            {"role": "assistant", "content": _long("a2")},
        ],
    }
    res = adapt_wildchat(row, idx=0)
    assert res.conversation is not None
    # merged -> user, assistant, user, assistant = 4 turns
    assert res.conversation.num_turns == 4
    assert "a1a" in res.conversation.turns[1].content
    assert "a1b" in res.conversation.turns[1].content


def test_system_turns_dropped():
    row = {
        "prompt_id": "s",
        "messages": [
            {"role": "system", "content": _long("sys")},  # dropped
            {"role": "user", "content": _long("u1")},
            {"role": "assistant", "content": _long("a1")},
            {"role": "user", "content": _long("u2")},
            {"role": "assistant", "content": _long("a2")},
        ],
    }
    res = adapt_ultrachat(row, idx=0)
    assert res.conversation is not None
    assert res.conversation.turns[0].role == "user"
    assert res.conversation.num_turns == 4
