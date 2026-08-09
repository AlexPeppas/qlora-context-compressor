"""Append-only judge-result cache for the pilot / full bake-off.

Caches two kinds of judge work so an interrupted or re-run pilot skips
completed API calls:

  * faithfulness Stage 1 (critical-item extraction) — keyed by
    (conversation_id, judge_name). Shared across all systems/tiers of a
    conversation, so this is the biggest saver.
  * faithfulness per-row score — keyed by
    (conversation_id, system, tier, judge_name).

The cache is a JSONL file; each line is one cached record. On load we index
by key (last write wins). We DON'T include the full rubric/prompt/schema hash
in the surface key here because the whole cache file is scoped to one rubric
version by convention (delete the file to invalidate). For paper-grade
provenance the per-record `provenance` dict is stored so we can audit exactly
which model/prompt produced each cached value.

Design: deliberately simple and dependency-free (stdlib json only). For the
pilot's scale (hundreds of rows) a flat JSONL + in-memory dict is ample.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JudgeCache:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._stage1: dict[tuple[str, str], dict] = {}
        self._scores: dict[tuple[str, str, str, str], float] = {}
        self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        n = 0
        for line in self.path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "stage1":
                self._stage1[(rec["conversation_id"], rec["judge"])] = rec
            elif kind == "score":
                self._scores[
                    (rec["conversation_id"], rec["system"], rec["tier"], rec["judge"])
                ] = rec["score"]
            n += 1
        logger.info(
            "Loaded judge cache: %d stage1, %d scores from %s",
            len(self._stage1), len(self._scores), self.path,
        )

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- faithfulness Stage 1 ---------------------------------------------

    def get_faithfulness_stage1(
        self, conversation_id: str, judge_name: str
    ) -> tuple[list, Any] | None:
        """Returns (items, stage1_result) or None. Reconstructs the item list
        and a lightweight JudgeResult stand-in from the cached record."""
        rec = self._stage1.get((conversation_id, judge_name))
        if rec is None:
            return None
        from .faithfulness import CriticalItem
        from .llm_client import (
            JudgeProvenance,
            JudgeResult,
            TokenUsage,
        )
        from .faithfulness import ExtractedItemsV1

        items = [CriticalItem(**it) for it in rec["items"]]
        prov = rec.get("provenance", {})
        result = JudgeResult(
            parsed=ExtractedItemsV1(items=items),
            raw=rec.get("raw", ""),
            provenance=JudgeProvenance(
                judge_name=prov.get("judge_name", judge_name),
                model=prov.get("model", ""),
                snapshot_id=prov.get("snapshot_id", ""),
                backend=prov.get("backend", "cache"),
                prompt_hash=prov.get("prompt_hash", ""),
                prompt_name=prov.get("prompt_name", ""),
                schema_hash=prov.get("schema_hash", ""),
                schema_name=prov.get("schema_name", ""),
                temperature=prov.get("temperature", 0.0),
                seed=prov.get("seed"),
                seed_supported=prov.get("seed_supported", False),
            ),
            usage=TokenUsage(),
            wall_seconds=0.0,
        )
        return items, result

    def put_faithfulness_stage1(
        self, conversation_id: str, judge_name: str, items: list, result: Any
    ) -> None:
        rec = {
            "kind": "stage1",
            "conversation_id": conversation_id,
            "judge": judge_name,
            "items": [it.model_dump() for it in items],
            "raw": result.raw,
            "provenance": result.provenance.to_dict(),
        }
        self._stage1[(conversation_id, judge_name)] = rec
        self._append(rec)

    # -- faithfulness score ------------------------------------------------

    def get_faithfulness_score(
        self, conversation_id: str, system: str, tier: str, judge_name: str
    ) -> float | None:
        return self._scores.get((conversation_id, system, tier, judge_name))

    def put_faithfulness_score(
        self,
        conversation_id: str,
        system: str,
        tier: str,
        judge_name: str,
        evaluation: Any,
    ) -> None:
        score = evaluation.score.score
        self._scores[(conversation_id, system, tier, judge_name)] = score
        rec = {
            "kind": "score",
            "conversation_id": conversation_id,
            "system": system,
            "tier": tier,
            "judge": judge_name,
            "score": score,
            "detail": evaluation.score.to_dict(),
        }
        self._append(rec)


__all__ = ["JudgeCache"]
