from __future__ import annotations

import json
from types import SimpleNamespace

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
