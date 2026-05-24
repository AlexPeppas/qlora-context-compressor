"""Tests for compressor.eval.conversations.

Covers:
  * structured-schema invariants
  * legacy flat-string parsing
  * holdout for M_downstream
  * round-trip (flatten -> parse)
  * defensive failures the rubber-duck critique flagged
"""
from __future__ import annotations

import json

import pytest

from compressor.eval.conversations import (
    Conversation,
    ConversationValidationError,
    Turn,
    load_jsonl,
    parse_legacy_flat_string,
    validate,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Validation invariants
# ---------------------------------------------------------------------------


def _make_conv(turns: list[tuple[str, str]], cid: str = "test") -> Conversation:
    return Conversation(
        id=cid,
        scenario_type="x",
        turns=tuple(Turn(role=r, content=c) for r, c in turns),  # type: ignore[arg-type]
    )


def test_validate_minimal_pair():
    c = _make_conv([("user", "hi"), ("assistant", "hello")])
    validate(c)  # should not raise


def test_validate_rejects_empty_id():
    c = Conversation(id="", scenario_type="x", turns=(Turn("user", "hi"),))
    with pytest.raises(ConversationValidationError, match="empty id"):
        validate(c)


def test_validate_rejects_single_turn():
    c = _make_conv([("user", "hi")])
    with pytest.raises(ConversationValidationError, match="at least 2 turns"):
        validate(c)


def test_validate_rejects_assistant_first():
    c = _make_conv([("assistant", "hi"), ("user", "what")])
    with pytest.raises(ConversationValidationError, match="first turn must be 'user'"):
        validate(c)


def test_validate_rejects_double_user():
    c = _make_conv([("user", "hi"), ("user", "still me"), ("assistant", "ok")])
    with pytest.raises(ConversationValidationError, match="breaks alternation"):
        validate(c)


def test_validate_rejects_empty_content():
    c = _make_conv([("user", "hi"), ("assistant", "   ")])
    with pytest.raises(ConversationValidationError, match="empty content"):
        validate(c)


# ---------------------------------------------------------------------------
# Holdout for M_downstream
# ---------------------------------------------------------------------------


def test_holdout_returns_prior_plus_held_pair():
    c = _make_conv(
        [
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
        ]
    )
    prior, last_user, held = c.with_holdout()
    assert prior.num_turns == 2
    assert prior.turns[0].content == "u1"
    assert prior.turns[1].content == "a1"
    assert last_user.content == "u2"
    assert held.content == "a2"


def test_holdout_preserves_metadata():
    c = Conversation(
        id="x",
        scenario_type="coding",
        domain_hint="hint",
        source_dataset="synthetic_ood",
        source_metadata={"k": "v"},
        turns=tuple(
            [Turn("user", "u1"), Turn("assistant", "a1"), Turn("user", "u2"), Turn("assistant", "a2")]
        ),
    )
    prior, _, _ = c.with_holdout()
    assert prior.scenario_type == "coding"
    assert prior.domain_hint == "hint"
    assert prior.source_dataset == "synthetic_ood"
    assert prior.source_metadata == {"k": "v"}


def test_holdout_rejects_too_short():
    c = _make_conv([("user", "hi"), ("assistant", "hello")])
    with pytest.raises(ValueError, match="need at least 3"):
        c.with_holdout()


def test_holdout_rejects_wrong_ending():
    # Ends with user, not assistant -- cannot hold out the LAST assistant turn
    # because the conversation doesn't END with an assistant turn following a user turn.
    c = _make_conv(
        [("user", "u1"), ("assistant", "a1"), ("user", "u2")]
    )
    with pytest.raises(ValueError, match="does not end with"):
        c.with_holdout()


# ---------------------------------------------------------------------------
# Legacy parsing
# ---------------------------------------------------------------------------


def test_parse_legacy_basic():
    text = "[USER]: hi\n\n[ASSISTANT]: hello"
    turns = parse_legacy_flat_string(text)
    assert len(turns) == 2
    assert turns[0] == Turn("user", "hi")
    assert turns[1] == Turn("assistant", "hello")


def test_parse_legacy_preserves_code_blocks():
    text = (
        "[USER]: write me a function\n\n"
        "[ASSISTANT]: Sure:\n```python\ndef foo():\n    return 42\n```"
    )
    turns = parse_legacy_flat_string(text)
    assert len(turns) == 2
    assert "```python" in turns[1].content
    assert "return 42" in turns[1].content


def test_parse_legacy_role_marker_inside_code_block_breaks_alternation():
    """Regression test for the rubber-duck flagged failure mode: if a code
    block contains a literal `[USER]:` or `[ASSISTANT]:` string, the parser
    will mis-split. We detect this because alternation is violated."""
    text = (
        "[USER]: how do I parse chat logs?\n\n"
        "[ASSISTANT]: Look for `[USER]:` markers like this:\n"
        "```python\n"
        "if line.startswith('[USER]:'): pass\n"
        "```\n"
        "[ASSISTANT]: oops a stray marker"
    )
    # The mis-split would produce: user, assistant, assistant, assistant
    # which violates alternation -> validation raises.
    with pytest.raises(ConversationValidationError):
        parse_legacy_flat_string(text)


def test_parse_legacy_trims_content():
    text = "[USER]:    spaces around    \n\n[ASSISTANT]:\ttabs and newlines\n"
    turns = parse_legacy_flat_string(text)
    assert turns[0].content == "spaces around"
    assert turns[1].content == "tabs and newlines"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_flatten_then_parse_preserves_content():
    c = _make_conv(
        [
            ("user", "first user turn"),
            ("assistant", "first asst turn"),
            ("user", "second user turn with `code`"),
            ("assistant", "second asst turn:\n```\nx = 1\n```"),
        ]
    )
    flat = c.flatten()
    turns2 = parse_legacy_flat_string(flat, conv_id="rt")
    assert len(turns2) == len(c.turns)
    for orig, rt in zip(c.turns, turns2):
        assert orig.role == rt.role
        assert orig.content == rt.content


# ---------------------------------------------------------------------------
# I/O round-trip
# ---------------------------------------------------------------------------


def test_jsonl_write_then_load_roundtrip(tmp_path):
    convs = [
        Conversation(
            id="c1",
            scenario_type="coding",
            domain_hint="hint",
            source_dataset="synthetic",
            source_metadata={"foo": "bar"},
            turns=(Turn("user", "hi"), Turn("assistant", "hello")),
        ),
        Conversation(
            id="c2",
            scenario_type="research",
            turns=tuple(
                [Turn("user", "u1"), Turn("assistant", "a1"), Turn("user", "u2"), Turn("assistant", "a2")]
            ),
        ),
    ]
    path = tmp_path / "out.jsonl"
    write_jsonl(convs, path)
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].id == "c1"
    assert loaded[0].source_metadata == {"foo": "bar"}
    assert loaded[1].num_turns == 4


def test_jsonl_rejects_invalid_on_load(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "bad", "turns": [{"role": "assistant", "content": "wrong"}]})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConversationValidationError):
        load_jsonl(path)


# ---------------------------------------------------------------------------
# Sanity check: the real migrated corpus loads + every conv supports holdout
# ---------------------------------------------------------------------------


def test_real_migrated_corpus_is_valid(tmp_path):
    """The 6 OOD conversations actually used by the bake-off harness."""
    import os
    import pathlib

    real = pathlib.Path(__file__).resolve().parents[1] / "data" / "bakeoff_conversations.jsonl"
    if not real.exists():
        pytest.skip("real bake-off corpus not present")

    convs = load_jsonl(real)
    assert len(convs) >= 6
    for c in convs:
        # Every conv must support holdout (3+ turns, ends user->assistant)
        prior, last_user, held = c.with_holdout()
        assert prior.num_turns >= 2
        assert last_user.role == "user"
        assert held.role == "assistant"
        assert last_user.content
        assert held.content
