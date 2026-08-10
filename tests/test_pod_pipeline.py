from __future__ import annotations

import json

import pytest

from compressor.eval.conversations import Conversation, Turn
from compressor.eval.pod_pipeline import (
    JsonlShard,
    RunPaths,
    build_input_tasks,
    expected_sources,
    merge_shards,
)


def _conversation() -> Conversation:
    return Conversation(
        id="public-1",
        scenario_type="test",
        turns=(
            Turn("user", "first question"),
            Turn("assistant", "first answer"),
            Turn("user", "final question"),
            Turn("assistant", "held answer"),
        ),
        source_dataset="fixture",
    )


def _row(source: str, mode: str, tier: str) -> dict:
    return {
        "conversation_id": "public-1",
        "scenario_type": "test",
        "turn_age": tier,
        "target_ratio": {"recent": 3, "mid": 5, "old": 10}[tier],
        "source": source,
        "compressed": f"{source}-{tier}",
        "input_chars": 100,
        "output_chars": 20,
        "achieved_ratio": 5.0,
        "gen_seconds": 0.1,
        "input_mode": mode,
    }


def test_build_input_tasks_separates_full_and_holdout_inputs():
    tasks = build_input_tasks([_conversation()])

    assert len(tasks["standard"]) == 3
    assert len(tasks["downstream"]) == 3
    assert "held answer" in tasks["standard"][0].conversation
    assert "final question" not in tasks["downstream"][0].conversation
    assert tasks["downstream"][0].downstream == {
        "last_user_turn": "final question",
        "held_assistant_turn": "held answer",
        "prior_num_turns": 2,
        "prior_chars": len("[USER]: first question\n\n[ASSISTANT]: first answer"),
    }


def test_jsonl_shard_resumes_completed_rows(tmp_path):
    path = tmp_path / "source.standard.jsonl"
    task = build_input_tasks([_conversation()])["standard"][0]

    with JsonlShard(path, "source", "standard") as shard:
        assert shard.pending(task)
        shard.append(_row("source", "standard", "recent"))

    with JsonlShard(path, "source", "standard") as shard:
        assert not shard.pending(task)


def test_jsonl_shard_discards_torn_final_line(tmp_path):
    path = tmp_path / "source.standard.jsonl"
    good = json.dumps(_row("source", "standard", "recent"))
    path.write_text(good + "\n" + '{"conversation_id":', encoding="utf-8")

    with JsonlShard(path, "source", "standard"):
        pass

    assert path.read_text(encoding="utf-8") == good + "\n"


def test_merge_shards_checks_completeness_and_orders_rows(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    sources = {"base-qwen", "tfix375"}
    for mode in ("standard", "downstream"):
        for source in reversed(sorted(sources)):
            shard = paths.shards / f"{source}.{mode}.jsonl"
            with shard.open("w", encoding="utf-8") as fh:
                for tier in ("old", "recent", "mid"):
                    fh.write(json.dumps(_row(source, mode, tier)) + "\n")

    outputs = merge_shards(paths, expected=sources, conversation_count=1)

    assert outputs["standard"]["rows"] == 6
    rows = [
        json.loads(line)
        for line in (paths.outputs / "compressions_standard.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["turn_age"], row["source"]) for row in rows] == [
        ("recent", "base-qwen"),
        ("recent", "tfix375"),
        ("mid", "base-qwen"),
        ("mid", "tfix375"),
        ("old", "base-qwen"),
        ("old", "tfix375"),
    ]


def test_merge_shards_rejects_missing_rows(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    shard = paths.shards / "base-qwen.standard.jsonl"
    shard.write_text(
        json.dumps(_row("base-qwen", "standard", "recent")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="incomplete standard output"):
        merge_shards(paths, expected={"base-qwen"}, conversation_count=1)


def test_expected_sources_includes_seeded_ours_variants():
    assert expected_sources(
        ("frontier", "practical"),
        ("base", "ours", "llmlingua2"),
        "tfix375",
        (11, 22, 33),
    ) == {
        "frontier-gpt54",
        "practical-gpt4o-mini",
        "base-qwen",
        "tfix375",
        "tfix375-s11",
        "tfix375-s22",
        "tfix375-s33",
        "llmlingua2",
    }
