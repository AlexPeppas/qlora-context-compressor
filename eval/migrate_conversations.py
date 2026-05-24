"""Migrate data/bakeoff_conversations.jsonl from legacy flat-string format
to structured `turns: [{role, content}]` format.

One-off ETL. Idempotent: if the input file is already in structured format
(detected by presence of `turns` key), exits without modification.

Usage:
    python -m compressor.eval.migrate_conversations \\
        --in data/bakeoff_conversations.jsonl \\
        --out data/bakeoff_conversations_v2.jsonl

After verifying the v2 file looks correct, manually replace v1:
    mv data/bakeoff_conversations.jsonl data/bakeoff_conversations_v1.jsonl.bak
    mv data/bakeoff_conversations_v2.jsonl data/bakeoff_conversations.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .conversations import (
    Conversation,
    iter_legacy_jsonl,
    parse_legacy_flat_string,
    validate,
    write_jsonl,
)

logger = logging.getLogger("migrate_conversations")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)


def migrate_one(row: dict) -> Conversation:
    """Convert a single legacy row dict into a structured Conversation."""
    if "turns" in row:
        # Already migrated — pass through
        from .conversations import _from_dict  # noqa: PLC0415

        return _from_dict(row)

    if "conversation" not in row:
        raise ValueError(
            f"row {row.get('id', '<?>')!r}: neither 'turns' nor 'conversation' key found"
        )

    turns = parse_legacy_flat_string(row["conversation"], conv_id=row.get("id", "<?>"))
    conv = Conversation(
        id=row["id"],
        scenario_type=row.get("scenario_type", ""),
        domain_hint=row.get("domain_hint", ""),
        turns=turns,
        source_dataset="synthetic_ood",  # all current convs are our synthetic OOD set
        source_metadata={"migrated_from": "legacy_flat_string"},
    )
    validate(conv)
    return conv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    args = p.parse_args()

    converted: list[Conversation] = []
    n_already = 0
    for row in iter_legacy_jsonl(args.in_path):
        if "turns" in row:
            n_already += 1
        conv = migrate_one(row)
        converted.append(conv)
        logger.info(
            "%s: %d turns (%d chars total)",
            conv.id,
            conv.num_turns,
            sum(len(t.content) for t in conv.turns),
        )

    write_jsonl(converted, args.out_path)
    logger.info(
        "Wrote %d conversations to %s (%d were already migrated, %d converted)",
        len(converted),
        args.out_path,
        n_already,
        len(converted) - n_already,
    )


if __name__ == "__main__":
    main()
