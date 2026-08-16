from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from compressor.eval.llm_client import OpenAIJudgeClient


class _Response(BaseModel):
    answer: str


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
