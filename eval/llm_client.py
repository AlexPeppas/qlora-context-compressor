"""LLM judge client used by every Phase B rubric.

Goals:
  * Backend-agnostic (OpenAI + Anthropic + an offline Mock for tests).
  * Structured outputs enforced via Pydantic — judges cannot return
    malformed JSON. OpenAI uses the response_format json_schema path
    via client.beta.chat.completions.parse(); Anthropic uses tool-use.
  * Provenance recorded on every call: prompt hash, schema hash, model
    snapshot, temperature, seed support, token usage, cost. This lets
    the cache key correctly invalidate when ANY of those change, and
    gives the paper appendix exact reproducibility metadata.
  * Retry/backoff with explicit budget (default 3 retries, exponential).
  * No real API calls in the test path — MockJudgeClient returns
    canned responses keyed by (prompt_hash, input_hash).

Cost accounting is best-effort: we log token counts from API responses
where available and let a separate cost-table module convert to dollars.
We do NOT hard-code prices here because they drift.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, Type, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Provenance + result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeProvenance:
    """Reproducibility metadata recorded with every judge call.

    Serialized as JSON and embedded in cached results. The cache key
    derives from a subset of these fields (see `cache_key`).
    """

    judge_name: str  # short label, e.g. "gpt-5.5-primary"
    model: str  # model family, e.g. "gpt-5.5"
    snapshot_id: str  # pinned snapshot, e.g. "gpt-5.5-2026-04-01"
    backend: str  # "openai" | "anthropic" | "mock"
    prompt_hash: str  # sha256 of the prompt template content
    prompt_name: str  # e.g. "faithfulness_stage1_v1"
    schema_hash: str  # sha256 of the Pydantic JSON schema
    schema_name: str  # e.g. "ExtractedItemsV1"
    temperature: float
    seed: int | None  # may be None where backend doesn't support it
    seed_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "model": self.model,
            "snapshot_id": self.snapshot_id,
            "backend": self.backend,
            "prompt_hash": self.prompt_hash,
            "prompt_name": self.prompt_name,
            "schema_hash": self.schema_hash,
            "schema_name": self.schema_name,
            "temperature": self.temperature,
            "seed": self.seed,
            "seed_supported": self.seed_supported,
        }


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0  # may include cached/system tokens depending on backend


@dataclass(frozen=True)
class JudgeResult:
    """Output of a single judge call. `parsed` is the Pydantic-validated
    response; `raw` is the underlying string for debugging / appendix.
    """

    parsed: BaseModel
    raw: str
    provenance: JudgeProvenance
    usage: TokenUsage
    wall_seconds: float
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed.model_dump(),
            "raw": self.raw,
            "provenance": self.provenance.to_dict(),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "wall_seconds": self.wall_seconds,
            "extras": dict(self.extras),
        }


# ---------------------------------------------------------------------------
# Cache key derivation
# ---------------------------------------------------------------------------


def cache_key(
    *,
    prompt_hash: str,
    schema_hash: str,
    snapshot_id: str,
    user_inputs: Mapping[str, str],
    rubric_version: str,
    seed: int | None,
) -> str:
    """Deterministic cache key for a judge call.

    Includes every field that can change the output. Bumping `rubric_version`
    invalidates the entire cache even if prompt + schema didn't change.
    """
    payload = {
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "snapshot_id": snapshot_id,
        "seed": seed,
        "rubric_version": rubric_version,
        # Sort user_inputs to get stable hashes regardless of dict order
        "user_inputs": {k: user_inputs[k] for k in sorted(user_inputs)},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def schema_hash_of(model_cls: Type[BaseModel]) -> str:
    """Stable hash of a Pydantic model's JSON schema. Treats schema as the
    semantic identity of the structured-output contract."""
    schema = model_cls.model_json_schema()
    blob = json.dumps(schema, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _repair_json_encoded_structures(
    payload: Any, errors: Sequence[Mapping[str, Any]]
) -> tuple[Any, list[str]]:
    """Decode JSON-stringified list/dict fields identified by Pydantic.

    Some tool-use responses serialize an array/object as a JSON string even
    though the tool schema declares a structured value. Repair only locations
    where validation explicitly reported a list/dict type mismatch; ordinary
    string fields are never coerced.
    """
    repaired = deepcopy(payload)
    repaired_paths: list[str] = []
    expected_types = {"list_type": list, "dict_type": dict}
    for error in errors:
        expected = expected_types.get(str(error.get("type")))
        location = tuple(error.get("loc") or ())
        if expected is None or not location:
            continue

        parent = repaired
        try:
            for part in location[:-1]:
                parent = parent[part]
            leaf = location[-1]
            value = parent[leaf]
        except (KeyError, IndexError, TypeError):
            continue
        if not isinstance(value, str):
            continue
        decoded: Any = value
        for _ in range(3):
            if not isinstance(decoded, str):
                break
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError:
                break
        if not isinstance(decoded, expected):
            continue
        parent[leaf] = decoded
        repaired_paths.append(".".join(str(part) for part in location))
    return repaired, repaired_paths


def _record_validation_failure(
    *,
    backend: str,
    model: str,
    prompt_name: str,
    schema_name: str,
    payload: Any,
    errors: Sequence[Mapping[str, Any]],
) -> Path:
    """Persist exact malformed structured output without API credentials."""
    path = Path(
        os.getenv(
            "JUDGE_FAILURE_LOG",
            "runs/judge_validation_failures.jsonl",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "backend": backend,
        "model": model,
        "prompt_name": prompt_name,
        "schema_name": schema_name,
        "payload": payload,
        "validation_errors": list(errors),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Protocol + retry helper
# ---------------------------------------------------------------------------


@runtime_checkable
class JudgeClient(Protocol):
    """All judge backends implement this interface.

    Implementations are stateless per call; concurrent use is the caller's
    responsibility. `__init__` may lazily import SDKs so the module can
    load on a machine without that SDK installed.
    """

    name: str  # e.g. "gpt-5.5-primary", used in `JudgeProvenance.judge_name`
    model: str
    snapshot_id: str
    backend: str  # "openai" | "anthropic" | "mock"

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        *,
        prompt_name: str,
        prompt_hash: str,
        seed: int | None = 42,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> JudgeResult:
        """Make one judge call. Returns a JudgeResult with the parsed
        Pydantic response. Implementations MUST retry transient errors
        and raise after exhausting the retry budget."""
        ...


def _exp_backoff_retry(
    fn: Callable[[], Any],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    transient_exc_types: Sequence[type[BaseException]] = (),
) -> Any:
    """Run `fn`, retrying on transient exceptions with exponential backoff.

    Non-transient errors raise immediately. Transient: rate limits, timeouts,
    connection errors. Each SDK exposes its own exception class for these.
    Callers pass the right types; we don't import SDKs at module load.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except tuple(transient_exc_types) as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2**attempt)
            logger.warning(
                "Transient error on attempt %d/%d: %s. Retrying in %.1fs",
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


class OpenAIJudgeClient:
    """OpenAI structured-output via `client.chat.completions.parse()`.

    Uses Pydantic models directly: OpenAI's SDK translates the model into a
    json_schema response_format and returns a parsed instance. We then wrap
    that with provenance + usage info.
    """

    backend = "openai"

    def __init__(
        self,
        name: str = "openai-judge",
        model: str = "gpt-4o-mini",
        snapshot_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        # Snapshot ID is normally the exact dated model string. If the
        # caller passes just the family ("gpt-5.5"), we record that as the
        # snapshot too — fine for testing, but for paper runs you should
        # pass the dated snapshot, e.g. "gpt-5.5-2026-04-15".
        self.snapshot_id = snapshot_id or model
        self._api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from dotenv import load_dotenv  # noqa: PLC0415

            load_dotenv()
        except ImportError:
            pass
        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; either pass api_key=... to "
                "OpenAIJudgeClient or set the env var"
            )
        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI(api_key=api_key)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        *,
        prompt_name: str,
        prompt_hash: str,
        seed: int | None = 42,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> JudgeResult:
        self._ensure_client()
        from openai import APIConnectionError, APITimeoutError, RateLimitError  # noqa: PLC0415

        def _do_call() -> Any:
            try:
                from compressor.baselines._openai_compat import openai_chat_create
            except ImportError:
                from baselines._openai_compat import openai_chat_create

            return openai_chat_create(
                self._client,
                model=self.snapshot_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                max_output_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )

        t0 = time.time()
        completion = _exp_backoff_retry(
            _do_call,
            max_retries=3,
            base_delay=2.0,
            transient_exc_types=(RateLimitError, APITimeoutError, APIConnectionError),
        )
        dt = time.time() - t0

        choice = completion.choices[0]
        parsed = choice.message.parsed
        if parsed is None:
            # Model refused or returned an unparseable response — surface clearly
            raise RuntimeError(
                f"OpenAI judge {self.snapshot_id} returned unparseable response: "
                f"finish_reason={choice.finish_reason}, refusal={choice.message.refusal!r}"
            )
        raw = choice.message.content or ""

        usage = completion.usage
        token_usage = TokenUsage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
        provenance = JudgeProvenance(
            judge_name=self.name,
            model=self.model,
            snapshot_id=self.snapshot_id,
            backend=self.backend,
            prompt_hash=prompt_hash,
            prompt_name=prompt_name,
            schema_hash=schema_hash_of(response_model),
            schema_name=response_model.__name__,
            temperature=temperature,
            seed=seed,
            seed_supported="seed" in getattr(completion, "_compat_params", {}),
        )
        return JudgeResult(
            parsed=parsed,
            raw=raw,
            provenance=provenance,
            usage=token_usage,
            wall_seconds=round(dt, 3),
            extras={
                "finish_reason": choice.finish_reason,
                "accepted_generation_params": getattr(
                    completion, "_compat_params", {}
                ),
            },
        )


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


class AnthropicJudgeClient:
    """Anthropic structured-output via tool-use.

    Claude doesn't support a native json_schema response_format like OpenAI;
    the idiomatic equivalent is to declare a single tool whose `input_schema`
    is the Pydantic schema. We force the model to use that tool, then extract
    the parsed input from the response.
    """

    backend = "anthropic"

    def __init__(
        self,
        name: str = "anthropic-judge",
        model: str = "claude-sonnet-4-6",
        snapshot_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.snapshot_id = snapshot_id or model
        self._api_key = api_key
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from dotenv import load_dotenv  # noqa: PLC0415

            load_dotenv()
        except ImportError:
            pass
        api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; either pass api_key=... to "
                "AnthropicJudgeClient or set the env var"
            )
        from anthropic import Anthropic  # noqa: PLC0415

        self._client = Anthropic(api_key=api_key)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        *,
        prompt_name: str,
        prompt_hash: str,
        seed: int | None = 42,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> JudgeResult:
        self._ensure_client()
        from anthropic import APIConnectionError, APITimeoutError, RateLimitError  # noqa: PLC0415

        # Build the tool spec from the Pydantic schema. Tool name doubles as
        # the rubric stage identifier so the model's "tool_use" intent is
        # unambiguous.
        tool_name = response_model.__name__
        tool_schema = response_model.model_json_schema()
        tool = {
            "name": tool_name,
            "description": f"Return the structured {tool_name} object.",
            "input_schema": tool_schema,
        }

        def _do_call() -> Any:
            # Note: Claude doesn't support a `seed` parameter as of writing.
            # We record seed_supported=False in provenance so reproducibility
            # claims are honest.
            return self._client.messages.create(
                model=self.snapshot_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": user_prompt}],
            )

        t0 = time.time()
        response = _exp_backoff_retry(
            _do_call,
            max_retries=3,
            base_delay=2.0,
            transient_exc_types=(RateLimitError, APITimeoutError, APIConnectionError),
        )
        dt = time.time() - t0

        # Extract the tool_use block
        tool_block = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                tool_block = block
                break
        if tool_block is None:
            raise RuntimeError(
                f"Anthropic judge {self.snapshot_id} did not invoke the {tool_name!r} tool. "
                f"Response content types: {[getattr(b, 'type', None) for b in response.content]}"
            )

        normalized_input = tool_block.input
        repaired_paths: list[str] = []
        try:
            parsed = response_model.model_validate(normalized_input)
        except ValidationError as exc:
            normalized_input, repaired_paths = _repair_json_encoded_structures(
                tool_block.input, exc.errors()
            )
            if not repaired_paths:
                failure_path = _record_validation_failure(
                    backend=self.backend,
                    model=self.snapshot_id,
                    prompt_name=prompt_name,
                    schema_name=response_model.__name__,
                    payload=tool_block.input,
                    errors=exc.errors(),
                )
                logger.error(
                    "Unrecoverable Anthropic tool validation failure recorded at %s",
                    failure_path,
                )
                raise
            logger.warning(
                "Anthropic judge %s returned JSON-stringified tool fields; "
                "decoded and revalidated: %s",
                self.snapshot_id,
                repaired_paths,
            )
            try:
                parsed = response_model.model_validate(normalized_input)
            except ValidationError as repaired_exc:
                failure_path = _record_validation_failure(
                    backend=self.backend,
                    model=self.snapshot_id,
                    prompt_name=prompt_name,
                    schema_name=response_model.__name__,
                    payload=tool_block.input,
                    errors=repaired_exc.errors(),
                )
                logger.error(
                    "Anthropic tool repair still failed; exact payload recorded at %s",
                    failure_path,
                )
                raise
        raw = json.dumps(tool_block.input, ensure_ascii=False)

        usage = response.usage
        token_usage = TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )
        provenance = JudgeProvenance(
            judge_name=self.name,
            model=self.model,
            snapshot_id=self.snapshot_id,
            backend=self.backend,
            prompt_hash=prompt_hash,
            prompt_name=prompt_name,
            schema_hash=schema_hash_of(response_model),
            schema_name=response_model.__name__,
            temperature=temperature,
            seed=seed,
            seed_supported=False,
        )
        return JudgeResult(
            parsed=parsed,
            raw=raw,
            provenance=provenance,
            usage=token_usage,
            wall_seconds=round(dt, 3),
            extras={
                "stop_reason": response.stop_reason,
                "repaired_json_string_fields": repaired_paths,
            },
        )


# ---------------------------------------------------------------------------
# Mock backend — for tests and dry-run inspection
# ---------------------------------------------------------------------------


class MockJudgeClient:
    """Returns canned responses keyed by (prompt_name, content_hash).

    Used in unit tests and as a `--dry-run` validator: lets the full judge
    pipeline execute without any API spend. To use, register responses
    via `register_response()` before calling.
    """

    backend = "mock"

    def __init__(
        self,
        name: str = "mock-judge",
        model: str = "mock",
        snapshot_id: str = "mock-fixed",
    ) -> None:
        self.name = name
        self.model = model
        self.snapshot_id = snapshot_id
        # (prompt_name, sha256(user_prompt)) -> BaseModel
        self._responses: dict[tuple[str, str], BaseModel] = {}
        self.calls: list[dict[str, Any]] = []  # call log for tests

    def register_response(
        self, prompt_name: str, user_prompt: str, response: BaseModel
    ) -> None:
        h = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        self._responses[(prompt_name, h)] = response

    def register_response_by_hash(
        self, prompt_name: str, user_prompt_hash: str, response: BaseModel
    ) -> None:
        """Register a response by pre-computed hash — useful when the
        prompt is large and we don't want to materialize it twice in the test."""
        self._responses[(prompt_name, user_prompt_hash)] = response

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        *,
        prompt_name: str,
        prompt_hash: str,
        seed: int | None = 42,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> JudgeResult:
        user_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        self.calls.append(
            {
                "prompt_name": prompt_name,
                "user_prompt_hash": user_hash,
                "system_len": len(system_prompt),
                "user_len": len(user_prompt),
            }
        )
        response = self._responses.get((prompt_name, user_hash))
        if response is None:
            raise KeyError(
                f"MockJudgeClient has no registered response for "
                f"(prompt_name={prompt_name!r}, user_hash={user_hash[:16]}...). "
                f"Registered keys: "
                f"{[(p, h[:16]) for (p, h) in self._responses]}"
            )
        if not isinstance(response, response_model):
            raise TypeError(
                f"Registered response type {type(response).__name__} does not match "
                f"expected {response_model.__name__}"
            )
        provenance = JudgeProvenance(
            judge_name=self.name,
            model=self.model,
            snapshot_id=self.snapshot_id,
            backend=self.backend,
            prompt_hash=prompt_hash,
            prompt_name=prompt_name,
            schema_hash=schema_hash_of(response_model),
            schema_name=response_model.__name__,
            temperature=temperature,
            seed=seed,
            seed_supported=True,
        )
        return JudgeResult(
            parsed=response,
            raw=response.model_dump_json(),
            provenance=provenance,
            usage=TokenUsage(),  # mock: no real tokens
            wall_seconds=0.0,
        )


__all__ = [
    "AnthropicJudgeClient",
    "JudgeClient",
    "JudgeProvenance",
    "JudgeResult",
    "MockJudgeClient",
    "OpenAIJudgeClient",
    "TokenUsage",
    "cache_key",
    "schema_hash_of",
]
