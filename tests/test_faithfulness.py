"""Tests for compressor.eval.faithfulness.

Uses MockJudgeClient — no real API calls. Validates:
  * Stage 1 extraction signature + cache-key invariance
  * Stage 2 evidence validation downgrades unsupported claims
  * Aggregation math (present=1, partial=0.5, false=0)
  * Error paths (missing item ids, duplicate decisions)
  * Provenance fields propagate correctly
"""
from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from compressor.eval.faithfulness import (
    RUBRIC_VERSION,
    CoverageDecision,
    CoverageReportV1,
    CriticalItem,
    ExtractedItemsV1,
    check_coverage,
    evaluate,
    extract_critical_items,
    faithfulness_score,
)
from compressor.eval.llm_client import MockJudgeClient, cache_key, schema_hash_of
from compressor.eval.prompts import load_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _make_items(*specs: tuple[int, str, str, str]) -> list[CriticalItem]:
    """specs: (id, type, summary, verbatim_indicator)"""
    return [
        CriticalItem(id=i, type=t, summary=s, verbatim_indicator=v)  # type: ignore[arg-type]
        for i, t, s, v in specs
    ]


def _make_decisions(*specs: tuple[int, str, str]) -> list[CoverageDecision]:
    """specs: (id, present, evidence)"""
    return [
        CoverageDecision(id=i, present=p, evidence=e)  # type: ignore[arg-type]
        for i, p, e in specs
    ]


# ---------------------------------------------------------------------------
# Prompt loading + hashing
# ---------------------------------------------------------------------------


def test_prompt_loads_and_hashes_stably():
    p1 = load_prompt("faithfulness_stage1_v1")
    p2 = load_prompt("faithfulness_stage1_v1")
    assert p1.content_hash == p2.content_hash
    assert p1.system_template  # non-empty
    assert "{source}" in p1.user_template


def test_prompt_render_substitutes_placeholders():
    p = load_prompt("faithfulness_stage1_v1")
    system, user = p.render(source="USER: hi\nASSISTANT: hello")
    assert "{source}" not in user
    assert "USER: hi" in user


def test_stage2_prompt_has_both_placeholders():
    p = load_prompt("faithfulness_stage2_v1")
    assert "{items_json}" in p.user_template
    assert "{compression}" in p.user_template


def test_curator_blocks_are_stripped_before_hash(tmp_path, monkeypatch):
    """PKM curator auto-injection must not alter the rendered prompt or the
    content hash. Regression for the 2026-07-04 incident where an Obsidian
    curator tool injected wiki-link blocks into prompt files."""
    from compressor.eval import prompts as prompts_mod

    clean = "[SYSTEM]\nyou are a judge\n[USER]\ngrade {x}\n"
    injected = (
        clean
        + "\n%% curator:start %%\n## Related\n- [[faithfulness]]  <!-- score=0.7 -->\n%% curator:end %%\n"
    )

    clean_file = tmp_path / "clean.md"
    injected_file = tmp_path / "injected.md"
    clean_file.write_text(clean, encoding="utf-8")
    injected_file.write_text(injected, encoding="utf-8")

    monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", tmp_path)

    clean_tpl = prompts_mod.load_prompt("clean")
    injected_tpl = prompts_mod.load_prompt("injected")

    # The injected block must not appear in the rendered prompt
    _, user = injected_tpl.render(x="thing")
    assert "curator" not in user
    assert "faithfulness" not in user
    # Hash must match the clean version (block fully stripped)
    assert injected_tpl.content_hash == clean_tpl.content_hash


# ---------------------------------------------------------------------------
# Stage 1: extraction
# ---------------------------------------------------------------------------


def test_extract_critical_items_returns_pydantic_items():
    judge = MockJudgeClient()
    prompt = load_prompt("faithfulness_stage1_v1")
    source = "test source"
    _, user = prompt.render(source=source)

    canned = ExtractedItemsV1(
        items=[
            CriticalItem(id=1, type="decision", summary="use redis", verbatim_indicator="redis"),
            CriticalItem(id=2, type="number", summary="port 5432", verbatim_indicator="5432"),
        ]
    )
    judge.register_response("faithfulness_stage1_v1", user, canned)

    items, result = extract_critical_items(source, judge)
    assert len(items) == 2
    assert items[0].id == 1
    assert items[0].type == "decision"
    assert result.provenance.prompt_name == "faithfulness_stage1_v1"
    assert result.provenance.judge_name == "mock-judge"


def test_extract_critical_items_records_prompt_hash():
    judge = MockJudgeClient()
    prompt = load_prompt("faithfulness_stage1_v1")
    source = "test source"
    _, user = prompt.render(source=source)

    canned = ExtractedItemsV1(
        items=[CriticalItem(id=1, type="decision", summary="x", verbatim_indicator="y")]
    )
    judge.register_response("faithfulness_stage1_v1", user, canned)

    _, result = extract_critical_items(source, judge)
    assert result.provenance.prompt_hash == prompt.content_hash


# ---------------------------------------------------------------------------
# Stage 2: coverage check + evidence validation
# ---------------------------------------------------------------------------


def test_check_coverage_passes_through_valid_evidence():
    judge = MockJudgeClient()
    items = _make_items(
        (1, "number", "port 5432", "5432"),
        (2, "decision", "use redis", "redis"),
    )
    compression = "We chose redis as the cache, listening on port 5432."

    prompt = load_prompt("faithfulness_stage2_v1")
    import json as _json

    items_json = _json.dumps([it.model_dump() for it in items], ensure_ascii=False)
    _, user = prompt.render(items_json=items_json, compression=compression)

    canned = CoverageReportV1(
        decisions=_make_decisions(
            (1, "present", "5432"),
            (2, "present", "redis"),
        )
    )
    judge.register_response("faithfulness_stage2_v1", user, canned)

    decisions, _ = check_coverage(items, compression, judge)
    assert decisions[0].present == "present"
    assert decisions[1].present == "present"


def test_check_coverage_accepts_long_verbatim_evidence():
    """Provider tool output may exceed the requested 30-char evidence span.
    A long span is still valid when it is an exact compression substring."""
    judge = MockJudgeClient()
    items = _make_items((1, "entity", "NLP applications", "Natural language"))
    evidence = (
        "Natural language processing supports classification, summarization, "
        "translation, and recommendation systems"
    )
    compression = f"Key applications: {evidence}."
    prompt = load_prompt("faithfulness_stage2_v1")
    items_json = json.dumps([it.model_dump() for it in items], ensure_ascii=False)
    _, user = prompt.render(items_json=items_json, compression=compression)
    judge.register_response(
        "faithfulness_stage2_v1",
        user,
        CoverageReportV1(
            decisions=[
                CoverageDecision(id=1, present="present", evidence=evidence)
            ]
        ),
    )

    decisions, _ = check_coverage(items, compression, judge)

    assert decisions[0].present == "present"
    assert decisions[0].evidence == evidence


@pytest.mark.parametrize(
    "encoded",
    [
        '[{"id": 1, "present": "false", "evidence": ""}]',
        json.dumps(
            json.dumps(
                json.dumps('[{"id": 1, "present": "false", "evidence": ""}]')
            )
        ),
        '```json\n[{"id": 1, "present": "false", "evidence": ""}]\n```',
    ],
)
def test_coverage_report_decodes_stringified_decisions(encoded):
    report = CoverageReportV1.model_validate({"decisions": encoded})

    assert report.decisions == [
        CoverageDecision(id=1, present="false", evidence="")
    ]


def test_coverage_report_rejects_malformed_stringified_decisions():
    with pytest.raises(ValidationError):
        CoverageReportV1.model_validate({"decisions": "[not valid JSON"})


def test_coverage_report_repairs_missing_present_keys_from_captured_payload():
    captured_shape = (
        '[{"id": 1, "present": "present", "evidence": "outline"},'
        '{"id": 2, "partial", "evidence": "challenges/lessons"},'
        '{"id": 3, "present": "partial", "evidence": "privacy/security"},'
        '{"id": 4, "false", "evidence": ""}]'
    )

    report = CoverageReportV1.model_validate({"decisions": captured_shape})

    assert [(decision.id, decision.present) for decision in report.decisions] == [
        (1, "present"),
        (2, "partial"),
        (3, "partial"),
        (4, "false"),
    ]


def test_coverage_report_does_not_repair_unknown_standalone_values():
    malformed = '[{"id": 1, "maybe", "evidence": ""}]'

    with pytest.raises(ValidationError):
        CoverageReportV1.model_validate({"decisions": malformed})


def test_check_coverage_downgrades_hallucinated_evidence():
    """Judge claims `present` with evidence that isn't actually in the compression
    -> automatically downgraded to `false`. This is the rubber-duck-required
    grounding check."""
    judge = MockJudgeClient()
    items = _make_items(
        (1, "decision", "use redis", "redis"),
        (2, "number", "port 5432", "5432"),
    )
    compression = "We chose redis as the cache."  # NO 5432 mentioned

    prompt = load_prompt("faithfulness_stage2_v1")
    import json as _json

    items_json = _json.dumps([it.model_dump() for it in items], ensure_ascii=False)
    _, user = prompt.render(items_json=items_json, compression=compression)

    canned = CoverageReportV1(
        decisions=_make_decisions(
            (1, "present", "redis"),
            (2, "present", "5432"),  # judge LIES; this string is NOT in compression
        )
    )
    judge.register_response("faithfulness_stage2_v1", user, canned)

    decisions, _ = check_coverage(items, compression, judge)
    assert decisions[0].present == "present"
    assert decisions[1].present == "false"  # downgraded
    assert decisions[1].evidence == ""


def test_check_coverage_is_case_insensitive_for_evidence():
    judge = MockJudgeClient()
    items = _make_items((1, "entity", "Redis", "Redis"))
    compression = "we chose REDIS as the cache."  # different case

    prompt = load_prompt("faithfulness_stage2_v1")
    import json as _json

    items_json = _json.dumps([it.model_dump() for it in items], ensure_ascii=False)
    _, user = prompt.render(items_json=items_json, compression=compression)

    canned = CoverageReportV1(decisions=_make_decisions((1, "present", "redis")))
    judge.register_response("faithfulness_stage2_v1", user, canned)

    decisions, _ = check_coverage(items, compression, judge)
    assert decisions[0].present == "present"


def test_check_coverage_keeps_false_without_evidence():
    judge = MockJudgeClient()
    items = _make_items((1, "number", "port 5432", "5432"))
    compression = "no number here"

    prompt = load_prompt("faithfulness_stage2_v1")
    import json as _json

    items_json = _json.dumps([it.model_dump() for it in items], ensure_ascii=False)
    _, user = prompt.render(items_json=items_json, compression=compression)

    canned = CoverageReportV1(decisions=_make_decisions((1, "false", "")))
    judge.register_response("faithfulness_stage2_v1", user, canned)

    decisions, _ = check_coverage(items, compression, judge)
    assert decisions[0].present == "false"
    assert decisions[0].evidence == ""


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_faithfulness_score_all_present_is_1():
    items = _make_items(
        (1, "decision", "x", "x"),
        (2, "number", "y", "y"),
    )
    decisions = _make_decisions((1, "present", "x"), (2, "present", "y"))
    s = faithfulness_score(items, decisions)
    assert s.score == 1.0
    assert s.n_present == 2 and s.n_partial == 0 and s.n_false == 0


def test_faithfulness_score_all_false_is_0():
    items = _make_items(
        (1, "decision", "x", "x"),
        (2, "number", "y", "y"),
    )
    decisions = _make_decisions((1, "false", ""), (2, "false", ""))
    s = faithfulness_score(items, decisions)
    assert s.score == 0.0


def test_faithfulness_score_partial_counts_half():
    items = _make_items(
        (1, "decision", "x", "x"),
        (2, "number", "y", "y"),
        (3, "code", "z", "z"),
    )
    decisions = _make_decisions(
        (1, "present", "x"), (2, "partial", "y"), (3, "false", "")
    )
    s = faithfulness_score(items, decisions)
    # (1 + 0.5 + 0) / 3 = 0.5
    assert s.score == pytest.approx(0.5)
    assert s.n_present == 1
    assert s.n_partial == 1
    assert s.n_false == 1


def test_faithfulness_score_rejects_duplicate_ids():
    items = _make_items((1, "decision", "x", "x"))
    decisions = _make_decisions((1, "present", "x"), (1, "false", ""))
    with pytest.raises(ValueError, match="duplicate decision ids"):
        faithfulness_score(items, decisions)


def test_faithfulness_score_rejects_id_mismatch():
    items = _make_items((1, "decision", "x", "x"), (2, "number", "y", "y"))
    decisions = _make_decisions((1, "present", "x"), (3, "false", ""))
    with pytest.raises(ValueError, match="do not match"):
        faithfulness_score(items, decisions)


# ---------------------------------------------------------------------------
# End-to-end via evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_runs_both_stages_when_items_not_cached():
    judge = MockJudgeClient()
    source = "USER: hi\nASSISTANT: redis on 5432"
    compression = "We chose redis as the cache, listening on port 5432."

    # Set up Stage 1 mock
    p1 = load_prompt("faithfulness_stage1_v1")
    _, user1 = p1.render(source=source)
    extracted = ExtractedItemsV1(
        items=[
            CriticalItem(id=1, type="entity", summary="redis", verbatim_indicator="redis"),
            CriticalItem(id=2, type="number", summary="5432", verbatim_indicator="5432"),
        ]
    )
    judge.register_response("faithfulness_stage1_v1", user1, extracted)

    # Set up Stage 2 mock
    p2 = load_prompt("faithfulness_stage2_v1")
    import json as _json

    items_json = _json.dumps([it.model_dump() for it in extracted.items], ensure_ascii=False)
    _, user2 = p2.render(items_json=items_json, compression=compression)
    coverage = CoverageReportV1(
        decisions=_make_decisions((1, "present", "redis"), (2, "present", "5432"))
    )
    judge.register_response("faithfulness_stage2_v1", user2, coverage)

    evaluation = evaluate(source, compression, judge)
    assert evaluation.score.score == 1.0
    # Verify both stages were called
    assert len(judge.calls) == 2
    assert judge.calls[0]["prompt_name"] == "faithfulness_stage1_v1"
    assert judge.calls[1]["prompt_name"] == "faithfulness_stage2_v1"


def test_evaluate_skips_stage1_when_items_cached():
    """Verifies the per-source Stage-1-share-across-baselines optimization."""
    judge = MockJudgeClient()
    source = "anything"
    compression = "redis on 5432"

    # Pre-extracted items + a fake Stage 1 result
    items = _make_items(
        (1, "entity", "redis", "redis"),
        (2, "number", "5432", "5432"),
    )
    # Use a real call to populate the JudgeResult shape with a different mock setup
    # — easier than fabricating one by hand
    fake_judge = MockJudgeClient(name="prev-judge")
    p1 = load_prompt("faithfulness_stage1_v1")
    _, user1 = p1.render(source="dummy")
    fake_judge.register_response(
        "faithfulness_stage1_v1",
        user1,
        ExtractedItemsV1(items=items),
    )
    _, stage1_result = extract_critical_items("dummy", fake_judge)

    # Now set up the real judge for Stage 2 only
    p2 = load_prompt("faithfulness_stage2_v1")
    import json as _json

    items_json = _json.dumps([it.model_dump() for it in items], ensure_ascii=False)
    _, user2 = p2.render(items_json=items_json, compression=compression)
    judge.register_response(
        "faithfulness_stage2_v1",
        user2,
        CoverageReportV1(
            decisions=_make_decisions((1, "present", "redis"), (2, "present", "5432"))
        ),
    )

    evaluation = evaluate(
        source, compression, judge, items=items, stage1_result=stage1_result
    )
    assert evaluation.score.score == 1.0
    # Only Stage 2 should have been called on the real judge
    assert len(judge.calls) == 1
    assert judge.calls[0]["prompt_name"] == "faithfulness_stage2_v1"


def test_evaluate_rejects_partial_cache():
    judge = MockJudgeClient()
    items = _make_items((1, "decision", "x", "x"))
    with pytest.raises(ValueError, match="either both"):
        evaluate("src", "comp", judge, items=items, stage1_result=None)


# ---------------------------------------------------------------------------
# Cache key invariance
# ---------------------------------------------------------------------------


def test_cache_key_stable_across_dict_order():
    k1 = cache_key(
        prompt_hash="abc",
        schema_hash="def",
        snapshot_id="gpt-test",
        user_inputs={"a": "1", "b": "2"},
        rubric_version=RUBRIC_VERSION,
        seed=42,
    )
    k2 = cache_key(
        prompt_hash="abc",
        schema_hash="def",
        snapshot_id="gpt-test",
        user_inputs={"b": "2", "a": "1"},
        rubric_version=RUBRIC_VERSION,
        seed=42,
    )
    assert k1 == k2


def test_cache_key_changes_with_rubric_version():
    base = dict(
        prompt_hash="abc",
        schema_hash="def",
        snapshot_id="gpt-test",
        user_inputs={"a": "1"},
        seed=42,
    )
    k1 = cache_key(rubric_version="v1", **base)
    k2 = cache_key(rubric_version="v2", **base)
    assert k1 != k2


def test_cache_key_changes_with_seed():
    base = dict(
        prompt_hash="abc",
        schema_hash="def",
        snapshot_id="gpt-test",
        user_inputs={"a": "1"},
        rubric_version=RUBRIC_VERSION,
    )
    k1 = cache_key(seed=42, **base)
    k2 = cache_key(seed=None, **base)
    assert k1 != k2


def test_schema_hash_is_stable():
    h1 = schema_hash_of(ExtractedItemsV1)
    h2 = schema_hash_of(ExtractedItemsV1)
    assert h1 == h2
    # Different schemas produce different hashes
    h3 = schema_hash_of(CoverageReportV1)
    assert h1 != h3


# ---------------------------------------------------------------------------
# Provenance propagation
# ---------------------------------------------------------------------------


def test_provenance_has_all_required_fields():
    judge = MockJudgeClient()
    prompt = load_prompt("faithfulness_stage1_v1")
    _, user = prompt.render(source="x")
    judge.register_response(
        "faithfulness_stage1_v1",
        user,
        ExtractedItemsV1(
            items=[
                CriticalItem(id=1, type="decision", summary="x", verbatim_indicator="x")
            ]
        ),
    )
    _, result = extract_critical_items("x", judge)
    prov = result.provenance.to_dict()
    required = {
        "judge_name", "model", "snapshot_id", "backend",
        "prompt_hash", "prompt_name", "schema_hash", "schema_name",
        "temperature", "seed", "seed_supported",
    }
    assert required.issubset(prov.keys())
