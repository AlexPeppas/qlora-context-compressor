"""Run the public-corpus compression experiment end-to-end on a GPU pod.

Stages:
  ingest      Pull and normalize public datasets in parallel.
  corpus      Merge, deduplicate, validate, and hash the selected corpus.
  compress    Overlap bounded API work with a VRAM-safe sequential GPU lane.
  merge       Merge resumable per-baseline shards into standard/holdout files.

Example from /workspace/compressor:

    python -m eval.pod_pipeline \
        --run-id pilot-01 \
        --datasets wildchat=3,oasst2=3,ultrachat=3 \
        --adapter checkpoints/qwen2.5-7b-compressor-eosfix/checkpoint-375 \
        --api-baselines frontier,practical \
        --gpu-baselines base,ours,llmlingua2,longllmlingua \
        --ours-seeds 11,22,33

All outputs are append-only or atomically replaced under runs/<run-id>.
Re-running the same command resumes completed rows instead of rotating files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .conversations import Conversation, load_jsonl, validate, write_jsonl
from .ingest_corpus import MIN_CHARS, MIN_TURNS, pull_and_ingest

logger = logging.getLogger("pod_pipeline")
TIERS: tuple[tuple[str, int], ...] = (
    ("recent", 3),
    ("mid", 5),
    ("old", 10),
)
TIER_ORDER = {name: index for index, (name, _) in enumerate(TIERS)}
DATASET_ORDER = ("wildchat", "oasst2", "ultrachat", "mtbench")
DATASET_REPOS = {
    "wildchat": ("allenai/WildChat-1M", "ODC-BY"),
    "oasst2": ("OpenAssistant/oasst2", "Apache-2.0"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "MIT"),
    "mtbench": ("HuggingFaceH4/mt_bench_prompts", "CC-BY-4.0"),
}
FAILURE_LOCK = threading.Lock()


@dataclass(frozen=True)
class InputTask:
    conversation_id: str
    scenario_type: str
    turn_age: str
    target_ratio: int
    conversation: str
    mode: str
    downstream: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunPaths:
    root: Path
    datasets: Path
    shards: Path
    outputs: Path
    logs: Path
    corpus: Path
    manifest: Path
    failures: Path

    @classmethod
    def create(cls, root: Path) -> "RunPaths":
        paths = cls(
            root=root,
            datasets=root / "datasets",
            shards=root / "shards",
            outputs=root / "outputs",
            logs=root / "logs",
            corpus=root / "corpus.jsonl",
            manifest=root / "manifest.json",
            failures=root / "failures.jsonl",
        )
        for directory in (paths.root, paths.datasets, paths.shards, paths.outputs, paths.logs):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


class Manifest:
    """Atomically persisted run provenance and stage state."""

    def __init__(self, path: Path, config: dict[str, Any]) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            previous = dict(self.data.get("config") or {})
            previous_adapter = previous.get("adapter")
            current_adapter = config.get("adapter")
            previous["adapter"] = previous_adapter or current_adapter
            comparable = dict(config)
            comparable["adapter"] = current_adapter or previous_adapter
            if previous != comparable:
                raise RuntimeError(
                    f"{path} belongs to a run with different configuration; "
                    "choose another --run-id or restore the original arguments"
                )
            self.data["config"] = comparable
            self._write()
        else:
            self.data = {
                "schema_version": 1,
                "created_at": _utc_now(),
                "status": "running",
                "config": config,
                "environment": _environment_manifest(),
                "stages": {},
            }
            self._write()

    def stage(self, name: str, status: str, **details: Any) -> None:
        with self._lock:
            stage = self.data["stages"].setdefault(name, {})
            stage.update(details)
            stage["status"] = status
            stage[f"{status}_at"] = _utc_now()
            self.data["updated_at"] = _utc_now()
            self._write()

    def finish(self) -> None:
        with self._lock:
            self.data["status"] = "complete"
            self.data["completed_at"] = _utc_now()
            self._write()

    def _write(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


class JsonlShard:
    """Append-only result shard with row-level resume keys."""

    def __init__(self, path: Path, source: str, mode: str) -> None:
        self.path = path
        self.source = source
        self.mode = mode
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed: set[tuple[str, str]] = set()
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            valid_lines: list[str] = []
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    if line_number == len(lines):
                        logger.warning(
                            "discarding torn final JSONL line from %s", path
                        )
                        path.write_text(
                            "".join(f"{valid}\n" for valid in valid_lines),
                            encoding="utf-8",
                        )
                        break
                    raise RuntimeError(f"invalid JSONL in {path}:{line_number}") from exc
                valid_lines.append(line)
                self.completed.add((row["conversation_id"], row["turn_age"]))
        self._fh = path.open("a", encoding="utf-8")

    def pending(self, task: InputTask) -> bool:
        return (task.conversation_id, task.turn_age) not in self.completed

    def append(self, row: dict[str, Any]) -> None:
        row["input_mode"] = self.mode
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.completed.add((row["conversation_id"], row["turn_age"]))

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlShard":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _environment_manifest() -> dict[str, Any]:
    packages = (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "datasets",
        "llmlingua",
        "openai",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "packages": {name: _package_version(name) for name in packages},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_logging(path: Path, verbose: bool) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(name)s - %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


@contextmanager
def _stage_logging(path: Path):
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(threadName)s %(name)s - %(message)s"
        )
    )
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def _adapter_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    files = [
        candidate
        for candidate in (
            path / "adapter_config.json",
            path / "adapter_model.safetensors",
            path / "adapter_model.bin",
        )
        if candidate.exists()
    ]
    return {
        "path": str(path),
        "files": {
            candidate.name: {
                "bytes": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            }
            for candidate in files
        },
    }


def _parse_dataset_limits(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in filter(None, (part.strip() for part in value.split(","))):
        name, separator, limit_text = item.partition("=")
        if separator != "=" or name not in DATASET_ORDER:
            raise argparse.ArgumentTypeError(
                f"invalid dataset selection {item!r}; expected name=limit"
            )
        limit = int(limit_text)
        if limit <= 0:
            raise argparse.ArgumentTypeError("dataset limits must be positive")
        result[name] = limit
    if not result:
        raise argparse.ArgumentTypeError("at least one dataset is required")
    return result


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def ingest_datasets(
    selections: dict[str, int],
    paths: RunPaths,
    workers: int,
) -> dict[str, dict[str, Any]]:
    """Pull independent HF datasets concurrently into normalized JSONL files."""

    def ingest_one(name: str, limit: int) -> tuple[str, int, Path, str, str, str]:
        out = paths.datasets / f"{name}.jsonl"
        metadata_path = paths.datasets / f"{name}.meta.json"
        repo, license_name = DATASET_REPOS[name]
        from huggingface_hub import HfApi

        if out.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            count = len(load_jsonl(out))
            logger.info("ingest resume: %s already has %d rows", name, count)
            if count != metadata["kept"] or _sha256(out) != metadata["sha256"]:
                raise RuntimeError(f"dataset artifact does not match {metadata_path}")
            return (
                name,
                count,
                out,
                metadata["repository"],
                metadata["revision"],
                metadata["license"],
            )
        revision = HfApi().dataset_info(repo).sha
        count = pull_and_ingest(name, out, limit=limit, revision=revision)
        metadata_path.write_text(
            json.dumps(
                {
                    "repository": repo,
                    "revision": revision,
                    "license": license_name,
                    "requested": limit,
                    "kept": count,
                    "sha256": _sha256(out),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if count < limit:
            logger.warning("%s yielded only %d/%d requested rows", name, count, limit)
        return name, count, out, repo, revision, license_name

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=min(workers, len(selections)),
        thread_name_prefix="dataset",
    ) as pool:
        futures = {
            pool.submit(ingest_one, name, limit): name
            for name, limit in selections.items()
        }
        for future in as_completed(futures):
            name, count, path, repo, revision, license_name = future.result()
            results[name] = {
                "repository": repo,
                "revision": revision,
                "license": license_name,
                "requested": selections[name],
                "kept": count,
                "path": str(path),
                "sha256": _sha256(path),
            }
    return results


def build_corpus(selections: dict[str, int], paths: RunPaths) -> list[Conversation]:
    """Merge dataset files in a fixed order and reject duplicate IDs."""
    merged: list[Conversation] = []
    seen: set[str] = set()
    for name in DATASET_ORDER:
        if name not in selections:
            continue
        dataset_path = paths.datasets / f"{name}.jsonl"
        for conversation in load_jsonl(dataset_path):
            validate(conversation)
            if conversation.id in seen:
                raise RuntimeError(f"duplicate conversation id: {conversation.id}")
            seen.add(conversation.id)
            merged.append(conversation)
    write_jsonl(merged, paths.corpus)
    logger.info("corpus: %d validated conversations -> %s", len(merged), paths.corpus)
    return merged


def build_input_tasks(conversations: Sequence[Conversation]) -> dict[str, list[InputTask]]:
    """Construct full-context and contamination-free holdout compression tasks."""
    tasks: dict[str, list[InputTask]] = {"standard": [], "downstream": []}
    for conversation in conversations:
        standard = conversation.flatten()
        prior, last_user, held_assistant = conversation.with_holdout()
        prior_flat = prior.flatten()
        downstream = {
            "last_user_turn": last_user.content,
            "held_assistant_turn": held_assistant.content,
            "prior_num_turns": prior.num_turns,
            "prior_chars": len(prior_flat),
        }
        for turn_age, ratio in TIERS:
            tasks["standard"].append(
                InputTask(
                    conversation_id=conversation.id,
                    scenario_type=conversation.scenario_type,
                    turn_age=turn_age,
                    target_ratio=ratio,
                    conversation=standard,
                    mode="standard",
                )
            )
            tasks["downstream"].append(
                InputTask(
                    conversation_id=conversation.id,
                    scenario_type=conversation.scenario_type,
                    turn_age=turn_age,
                    target_ratio=ratio,
                    conversation=prior_flat,
                    mode="downstream",
                    downstream=downstream,
                )
            )
    return tasks


def _request_for(task: InputTask):
    try:
        from compressor.baselines import CompressionRequest
    except ImportError:
        from baselines import CompressionRequest
    return CompressionRequest(
        conversation=task.conversation,
        turn_age=task.turn_age,
        target_ratio=task.target_ratio,
        conversation_id=task.conversation_id,
        scenario_type=task.scenario_type,
    )


def _record_result(shard: JsonlShard, task: InputTask, result: Any) -> None:
    row = result.to_jsonl_dict()
    if task.downstream is not None:
        row["downstream"] = task.downstream
    shard.append(row)


def _failure(paths: RunPaths, source: str, task: InputTask, exc: BaseException) -> None:
    row = {
        "timestamp": _utc_now(),
        "source": source,
        "input_mode": task.mode,
        "conversation_id": task.conversation_id,
        "turn_age": task.turn_age,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with FAILURE_LOCK:
        with paths.failures.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_sequential_baseline(
    baseline: Any,
    tasks_by_mode: dict[str, list[InputTask]],
    paths: RunPaths,
) -> int:
    completed = 0
    baseline.load()
    try:
        for mode, tasks in tasks_by_mode.items():
            shard_path = paths.shards / f"{baseline.name}.{mode}.jsonl"
            with JsonlShard(shard_path, baseline.name, mode) as shard:
                pending = [task for task in tasks if shard.pending(task)]
                logger.info("%s/%s: %d pending rows", baseline.name, mode, len(pending))
                for index, task in enumerate(pending, 1):
                    try:
                        result = baseline.compress(_request_for(task))
                        _record_result(shard, task, result)
                    except Exception as exc:
                        _failure(paths, baseline.name, task, exc)
                        raise
                    completed += 1
                    logger.info(
                        "%s/%s [%d/%d] %s %s %.2fx",
                        baseline.name,
                        mode,
                        index,
                        len(pending),
                        task.conversation_id,
                        task.turn_age,
                        result.achieved_ratio,
                    )
    finally:
        baseline.unload()
    return completed


def run_api_lane(
    selected: Sequence[str],
    tasks_by_mode: dict[str, list[InputTask]],
    paths: RunPaths,
    workers: int,
) -> int:
    """Run API models concurrently with bounded request fan-out."""
    try:
        from compressor.baselines.frontier import FrontierBaseline, PracticalAPIBaseline
    except ImportError:
        from baselines.frontier import FrontierBaseline, PracticalAPIBaseline

    factories = {
        "frontier": FrontierBaseline,
        "practical": PracticalAPIBaseline,
    }
    baselines = [factories[name]() for name in selected]
    for baseline in baselines:
        baseline.load()

    work: list[tuple[Any, InputTask, JsonlShard]] = []
    shards: list[JsonlShard] = []
    try:
        for baseline in baselines:
            for mode, tasks in tasks_by_mode.items():
                shard = JsonlShard(
                    paths.shards / f"{baseline.name}.{mode}.jsonl",
                    baseline.name,
                    mode,
                )
                shards.append(shard)
                work.extend(
                    (baseline, task, shard) for task in tasks if shard.pending(task)
                )
        logger.info("api lane: %d pending rows across %d models", len(work), len(baselines))
        completed = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="api") as pool:
            futures: dict[Future, tuple[Any, InputTask, JsonlShard]] = {
                pool.submit(baseline.compress, _request_for(task)): (baseline, task, shard)
                for baseline, task, shard in work
            }
            for future in as_completed(futures):
                baseline, task, shard = futures[future]
                try:
                    result = future.result()
                    _record_result(shard, task, result)
                except Exception as exc:
                    _failure(paths, baseline.name, task, exc)
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                completed += 1
                logger.info(
                    "api [%d/%d] %s/%s %s %s %.2fx",
                    completed,
                    len(work),
                    baseline.name,
                    task.mode,
                    task.conversation_id,
                    task.turn_age,
                    result.achieved_ratio,
                )
        return completed
    finally:
        for shard in shards:
            shard.close()
        for baseline in baselines:
            baseline.unload()


def _run_qwen_family(
    adapter_path: Path,
    adapter_name: str,
    include_base: bool,
    include_ours: bool,
    seeds: Sequence[int],
    tasks_by_mode: dict[str, list[InputTask]],
    paths: RunPaths,
) -> int:
    """Load Qwen once, then run base, greedy LoRA, and sampled LoRA variants."""
    try:
        from compressor.baselines import CompressionResult
        from compressor.baselines._qwen_runtime import (
            TIER_MAX_NEW,
            build_qwen_chat_prompt,
            generate,
            load_base_qwen_4bit,
        )
    except ImportError:
        from baselines import CompressionResult
        from baselines._qwen_runtime import (
            TIER_MAX_NEW,
            build_qwen_chat_prompt,
            generate,
            load_base_qwen_4bit,
        )

    import torch
    from peft import PeftModel

    model, tokenizer = load_base_qwen_4bit()
    if include_ours:
        model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=adapter_name)
        model.set_adapter(adapter_name)
    model.eval()
    variants: list[tuple[str, bool, int | None]] = []
    if include_base:
        variants.append(("base-qwen", False, None))
    if include_ours:
        variants.append((adapter_name, True, None))
        variants.extend((f"{adapter_name}-s{seed}", True, seed) for seed in seeds)

    completed = 0
    try:
        for source, use_adapter, seed in variants:
            for mode, tasks in tasks_by_mode.items():
                with JsonlShard(
                    paths.shards / f"{source}.{mode}.jsonl", source, mode
                ) as shard:
                    pending = [task for task in tasks if shard.pending(task)]
                    logger.info("%s/%s: %d pending rows", source, mode, len(pending))
                    adapter_context = (
                        nullcontext()
                        if use_adapter
                        else model.disable_adapter()
                        if include_ours
                        else nullcontext()
                    )
                    with adapter_context:
                        for index, task in enumerate(pending, 1):
                            prompt = build_qwen_chat_prompt(
                                task.conversation,
                                task.turn_age,
                                task.target_ratio,
                            )
                            sampled = seed is not None
                            try:
                                generated = generate(
                                    model,
                                    tokenizer,
                                    prompt,
                                    TIER_MAX_NEW[task.turn_age],
                                    do_sample=sampled,
                                    seed=seed,
                                    temperature=0.7,
                                    top_p=0.9,
                                )
                                text = generated["text"].strip()
                                result = CompressionResult(
                                    conversation_id=task.conversation_id,
                                    scenario_type=task.scenario_type,
                                    turn_age=task.turn_age,
                                    target_ratio=task.target_ratio,
                                    source=source,
                                    compressed=text,
                                    input_chars=len(task.conversation),
                                    output_chars=len(text),
                                    achieved_ratio=len(task.conversation) / max(len(text), 1),
                                    gen_seconds=round(generated["gen_seconds"], 3),
                                    input_tokens=generated["input_tokens"],
                                    output_tokens=generated["output_tokens"],
                                    max_new_tokens=TIER_MAX_NEW[task.turn_age],
                                    stop_reason=generated["stop_reason"],
                                    stopped_on_eos=generated["stopped_on_eos"],
                                    extras={
                                        "base_model": "Qwen/Qwen2.5-7B-Instruct",
                                        "adapter": str(adapter_path) if use_adapter else None,
                                        "seed": seed,
                                        "temperature": 0.7 if sampled else 0.0,
                                        "top_p": 0.9 if sampled else 1.0,
                                    },
                                )
                                _record_result(shard, task, result)
                            except Exception as exc:
                                _failure(paths, source, task, exc)
                                raise
                            completed += 1
                            logger.info(
                                "%s/%s [%d/%d] %s %s %.2fx",
                                source,
                                mode,
                                index,
                                len(pending),
                                task.conversation_id,
                                task.turn_age,
                                result.achieved_ratio,
                            )
    finally:
        del model
        torch.cuda.empty_cache()
    return completed


def run_gpu_lane(
    selected: Sequence[str],
    adapter_path: Path | None,
    adapter_name: str,
    seeds: Sequence[int],
    tasks_by_mode: dict[str, list[InputTask]],
    paths: RunPaths,
) -> int:
    """Run one GPU-resident model family at a time to avoid VRAM contention."""
    if selected and not _cuda_available():
        raise RuntimeError("GPU baselines requested but torch.cuda.is_available() is false")
    include_base = "base" in selected
    include_ours = "ours" in selected
    if include_ours and adapter_path is None:
        raise RuntimeError("--adapter is required when gpu baseline 'ours' is selected")

    completed = 0
    if include_base or include_ours:
        assert adapter_path is not None or not include_ours
        completed += _run_qwen_family(
            adapter_path or Path("."),
            adapter_name,
            include_base,
            include_ours,
            seeds,
            tasks_by_mode,
            paths,
        )
    if "llmlingua2" in selected:
        try:
            from compressor.baselines.lingua import LLMLingua2Baseline
        except ImportError:
            from baselines.lingua import LLMLingua2Baseline
        completed += _run_sequential_baseline(
            LLMLingua2Baseline(), tasks_by_mode, paths
        )
    if "longllmlingua" in selected:
        try:
            from compressor.baselines.lingua import LongLLMLinguaBaseline
        except ImportError:
            from baselines.lingua import LongLLMLinguaBaseline
        completed += _run_sequential_baseline(
            LongLLMLinguaBaseline(), tasks_by_mode, paths
        )
    return completed


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def expected_sources(
    api_baselines: Sequence[str],
    gpu_baselines: Sequence[str],
    adapter_name: str,
    seeds: Sequence[int],
) -> set[str]:
    sources: set[str] = set()
    if "frontier" in api_baselines:
        sources.add("frontier-gpt54")
    if "practical" in api_baselines:
        sources.add("practical-gpt4o-mini")
    if "base" in gpu_baselines:
        sources.add("base-qwen")
    if "ours" in gpu_baselines:
        sources.add(adapter_name)
        sources.update(f"{adapter_name}-s{seed}" for seed in seeds)
    if "llmlingua2" in gpu_baselines:
        sources.add("llmlingua2")
    if "longllmlingua" in gpu_baselines:
        sources.add("longllmlingua")
    return sources


def merge_shards(
    paths: RunPaths,
    *,
    expected: set[str] | None = None,
    conversation_count: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Produce deterministic mode-level outputs from resumable source shards."""
    outputs: dict[str, dict[str, Any]] = {}
    for mode in ("standard", "downstream"):
        rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        for shard in sorted(paths.shards.glob(f"*.{mode}.jsonl")):
            lines = shard.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    if line_number == len(lines):
                        logger.warning(
                            "ignoring torn final JSONL line during merge: %s", shard
                        )
                        break
                    raise RuntimeError(
                        f"invalid JSONL in {shard}:{line_number}"
                    ) from exc
                key = (row["source"], row["conversation_id"], row["turn_age"])
                if key in rows:
                    raise RuntimeError(f"duplicate result key {key} in {shard}:{line_number}")
                rows[key] = row
        ordered = sorted(
            rows.values(),
            key=lambda row: (
                row["conversation_id"],
                TIER_ORDER[row["turn_age"]],
                row["source"],
            ),
        )
        if expected is not None and conversation_count is not None:
            expected_rows = len(expected) * conversation_count * len(TIERS)
            actual_sources = {row["source"] for row in ordered}
            if len(ordered) != expected_rows or actual_sources != expected:
                raise RuntimeError(
                    f"incomplete {mode} output: rows={len(ordered)}/{expected_rows}, "
                    f"sources={sorted(actual_sources)}, expected={sorted(expected)}"
                )
        out = paths.outputs / f"compressions_{mode}.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for row in ordered:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out)
        outputs[mode] = {
            "rows": len(ordered),
            "path": str(out),
            "sha256": _sha256(out),
        }
        logger.info("merge: %s -> %d rows", out, len(ordered))
    return outputs


def _preflight(
    api_baselines: Sequence[str],
    gpu_baselines: Sequence[str],
    adapter: Path | None,
) -> None:
    if api_baselines and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for API baselines")
    if "ours" in gpu_baselines and (adapter is None or not adapter.exists()):
        raise RuntimeError(f"LoRA adapter does not exist: {adapter}")
    unknown_api = set(api_baselines) - {"frontier", "practical"}
    unknown_gpu = set(gpu_baselines) - {
        "base",
        "ours",
        "llmlingua2",
        "longllmlingua",
    }
    if unknown_api or unknown_gpu:
        raise RuntimeError(
            f"unknown baselines: api={sorted(unknown_api)}, gpu={sorted(unknown_gpu)}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--stages",
        type=_parse_csv,
        default=("ingest", "corpus", "compress", "merge"),
        help="Comma-separated subset of ingest,corpus,compress,merge.",
    )
    parser.add_argument(
        "--datasets",
        type=_parse_dataset_limits,
        default={"wildchat": 15, "oasst2": 8, "ultrachat": 10},
        help="Comma-separated name=accepted-row-count selections.",
    )
    parser.add_argument("--dataset-workers", type=int, default=3)
    parser.add_argument(
        "--api-baselines",
        type=_parse_csv,
        default=("frontier", "practical"),
    )
    parser.add_argument("--api-workers", type=int, default=4)
    parser.add_argument(
        "--gpu-baselines",
        type=_parse_csv,
        default=("base", "ours", "llmlingua2", "longllmlingua"),
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-name", default="tfix375")
    parser.add_argument("--ours-seeds", type=_parse_seeds, default=(11, 22, 33))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    unknown_stages = set(args.stages) - {"ingest", "corpus", "compress", "merge"}
    if unknown_stages:
        raise SystemExit(f"unknown stages: {sorted(unknown_stages)}")
    paths = RunPaths.create(args.run_root / args.run_id)
    _configure_logging(paths.logs / "pipeline.log", args.verbose)
    if "compress" in args.stages:
        _preflight(args.api_baselines, args.gpu_baselines, args.adapter)
    config = {
        "run_id": args.run_id,
        "datasets": args.datasets,
        "filters": {
            "language": "English",
            "minimum_turns": MIN_TURNS,
            "minimum_characters": MIN_CHARS,
            "must_end_user_assistant": True,
        },
        "dataset_workers": args.dataset_workers,
        "api_baselines": list(args.api_baselines),
        "api_workers": args.api_workers,
        "gpu_baselines": list(args.gpu_baselines),
        "adapter": _adapter_fingerprint(args.adapter),
        "adapter_name": args.adapter_name,
        "ours_seeds": list(args.ours_seeds),
        "models": {
            "base_qwen": "Qwen/Qwen2.5-7B-Instruct",
            "frontier": "gpt-5.4-2026-03-05",
            "practical": "gpt-4o-mini-2024-07-18",
            "llmlingua2": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            "longllmlingua": "NousResearch/Llama-2-7b-hf",
        },
        "compression_prompt_sha256": hashlib.sha256(
            (
                "You are a context compressor.\n"
                "Turn age: {turn_age_desc}\n"
                "Target compression: ~1/{ratio}x\n"
                "Output only the compressed text."
            ).encode("utf-8")
        ).hexdigest(),
    }
    manifest = Manifest(paths.manifest, config)
    logger.info("run %s starting stages=%s", args.run_id, ",".join(args.stages))

    conversations: list[Conversation] | None = None
    try:
        if "ingest" in args.stages:
            manifest.stage("ingest", "running")
            started = time.monotonic()
            with _stage_logging(paths.logs / "ingest.log"):
                ingested = ingest_datasets(
                    args.datasets, paths, max(1, args.dataset_workers)
                )
            manifest.stage(
                "ingest",
                "complete",
                datasets=ingested,
                duration_seconds=round(time.monotonic() - started, 3),
            )

        if "corpus" in args.stages:
            manifest.stage("corpus", "running")
            started = time.monotonic()
            with _stage_logging(paths.logs / "corpus.log"):
                conversations = build_corpus(args.datasets, paths)
            manifest.stage(
                "corpus",
                "complete",
                conversations=len(conversations),
                path=str(paths.corpus),
                sha256=_sha256(paths.corpus),
                duration_seconds=round(time.monotonic() - started, 3),
            )

        if "compress" in args.stages:
            if conversations is None:
                conversations = load_jsonl(paths.corpus)
            tasks = build_input_tasks(conversations)
            manifest.stage(
                "compress",
                "running",
                tasks_per_mode=len(tasks["standard"]),
            )
            started = time.monotonic()
            with _stage_logging(paths.logs / "compress.log"):
                api_future: Future[int] | None = None
                with ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="api-lane"
                ) as pool:
                    if args.api_baselines:
                        api_future = pool.submit(
                            run_api_lane,
                            args.api_baselines,
                            tasks,
                            paths,
                            max(1, args.api_workers),
                        )
                    gpu_count = run_gpu_lane(
                        args.gpu_baselines,
                        args.adapter,
                        args.adapter_name,
                        args.ours_seeds,
                        tasks,
                        paths,
                    )
                    api_count = api_future.result() if api_future else 0
            manifest.stage(
                "compress",
                "complete",
                new_api_rows=api_count,
                new_gpu_rows=gpu_count,
                duration_seconds=round(time.monotonic() - started, 3),
            )

        if "merge" in args.stages:
            manifest.stage("merge", "running")
            started = time.monotonic()
            if conversations is None:
                conversations = load_jsonl(paths.corpus)
            with _stage_logging(paths.logs / "merge.log"):
                outputs = merge_shards(
                    paths,
                    expected=expected_sources(
                        args.api_baselines,
                        args.gpu_baselines,
                        args.adapter_name,
                        args.ours_seeds,
                    ),
                    conversation_count=len(conversations),
                )
            manifest.stage(
                "merge",
                "complete",
                outputs=outputs,
                duration_seconds=round(time.monotonic() - started, 3),
            )

        manifest.finish()
        logger.info("run %s complete -> %s", args.run_id, paths.root)
    except Exception:
        manifest.stage("pipeline", "failed")
        logger.exception("run %s failed; resume with the same command", args.run_id)
        raise


if __name__ == "__main__":
    main()
