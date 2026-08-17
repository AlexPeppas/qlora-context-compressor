from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from compressor.eval.llm_client import AnthropicJudgeClient, OpenAIJudgeClient


class _Response(BaseModel):
    answer: str


class _ListResponse(BaseModel):
    decisions: list[dict] = Field(min_length=1)


class _FakeCompletions:
    def __init__(self) -> None:
        self.params = None

    def parse(self, **params):
        self.params = params
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=_Response(answer="ok"),
                        content='{"answer":"ok"}',
                        refusal=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        )


def test_openai_judge_uses_modern_completion_token_parameter():
    completions = _FakeCompletions()
    client = OpenAIJudgeClient(
        name="gpt-primary",
        model="gpt-5.4-2026-03-05",
        snapshot_id="gpt-5.4-2026-03-05",
    )
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    result = client.call(
        "system",
        "user",
        _Response,
        prompt_name="test",
        prompt_hash="hash",
        max_tokens=123,
    )

    assert completions.params["max_completion_tokens"] == 123
    assert "max_tokens" not in completions.params
    assert result.parsed.answer == "ok"
    assert result.extras["accepted_generation_params"]["max_completion_tokens"] == 123


def test_anthropic_judge_repairs_json_stringified_list_field():
    tool_input = {
        "decisions": '[{"id": 1, "present": "false", "evidence": ""}]'
    }
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="_ListResponse",
                input=tool_input,
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
    )
    messages = SimpleNamespace(create=lambda **_params: response)
    client = AnthropicJudgeClient(
        name="claude-secondary",
        model="claude-sonnet-4-6",
    )
    client._client = SimpleNamespace(messages=messages)

    result = client.call(
        "system",
        "user",
        _ListResponse,
        prompt_name="test",
        prompt_hash="hash",
    )

    assert result.parsed.decisions == [
        {"id": 1, "present": "false", "evidence": ""}
    ]
    assert result.extras["repaired_json_string_fields"] == ["decisions"]


def test_anthropic_judge_repairs_double_json_encoded_list_field():
    encoded_once = '[{"id": 1, "present": "false", "evidence": ""}]'
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="_ListResponse",
                input={"decisions": json.dumps(encoded_once)},
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
    )
    client = AnthropicJudgeClient(name="claude-secondary")
    client._client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **_params: response)
    )

    result = client.call(
        "system",
        "user",
        _ListResponse,
        prompt_name="test",
        prompt_hash="hash",
    )

    assert result.parsed.decisions[0]["id"] == 1
    assert result.extras["repaired_json_string_fields"] == ["decisions"]


def test_anthropic_judge_does_not_coerce_normal_string_fields():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="_Response",
                input={"answer": '["still", "a", "string"]'},
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
    )
    client = AnthropicJudgeClient(name="claude-secondary")
    client._client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **_params: response)
    )

    result = client.call(
        "system",
        "user",
        _Response,
        prompt_name="test",
        prompt_hash="hash",
    )

    assert result.parsed.answer == '["still", "a", "string"]'
    assert result.extras["repaired_json_string_fields"] == []


def test_anthropic_judge_records_unrecoverable_tool_payload(tmp_path, monkeypatch):
    failure_log = tmp_path / "failures.jsonl"
    monkeypatch.setenv("JUDGE_FAILURE_LOG", str(failure_log))
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="_ListResponse",
                input={"decisions": "[not valid JSON"},
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
    )
    client = AnthropicJudgeClient(name="claude-secondary")
    client._client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **_params: response)
    )

    with pytest.raises(Exception):
        client.call(
            "system",
            "user",
            _ListResponse,
            prompt_name="faithfulness_stage2_v1",
            prompt_hash="hash",
        )

    records = [
        json.loads(line)
        for line in failure_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert records[-1]["payload"] == {"decisions": "[not valid JSON"}
    assert records[-1]["prompt_name"] == "faithfulness_stage2_v1"


def test_anthropic_judge_retries_invalid_tool_payload_with_correction(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JUDGE_FAILURE_LOG", str(tmp_path / "failures.jsonl"))
    invalid = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="_ListResponse",
                input={
                    "decisions": (
                        '[{"id": 1, "present": "present", '
                        '"evidence": "raw "unescaped" quote"}]'
                    )
                },
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
    )
    valid = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="_ListResponse",
                input={
                    "decisions": [
                        {"id": 1, "present": "present", "evidence": "quote"}
                    ]
                },
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=4),
        stop_reason="tool_use",
    )
    responses = iter([invalid, valid])
    calls = []

    def create(**params):
        calls.append(params)
        return next(responses)

    client = AnthropicJudgeClient(name="claude-secondary")
    client._client = SimpleNamespace(
        messages=SimpleNamespace(create=create)
    )

    result = client.call(
        "system",
        "user",
        _ListResponse,
        prompt_name="faithfulness_stage2_v1",
        prompt_hash="hash",
    )

    assert len(calls) == 2
    assert "STRUCTURED OUTPUT CORRECTION" in calls[1]["messages"][0]["content"]
    assert result.parsed.decisions[0]["present"] == "present"
    assert result.extras["structured_validation_retries"] == 1
    assert result.usage.input_tokens == 22
    assert result.usage.output_tokens == 9
