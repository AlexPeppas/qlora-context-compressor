"""Run GPU baseline compressors with the last-assistant-turn held out.

Produces `data/bakeoff_downstream_compressions.jsonl`: a parallel JSONL to
the standard bake-off output, but where each compression was generated
WITHOUT the held-out assistant turn in the input. This is the
contamination-free input for M_downstream judging (rubber-duck Phase B
blocker #1).

Required by Phase B.0.5. Run on a GPU pod after Phase B.0 migration:

    python -m compressor.eval.run_downstream_bakeoff \\
        --conversations data/bakeoff_conversations.jsonl \\
        --adapter checkpoints/qwen2.5-7b-compressor-eosfix/checkpoint-375 \\
        --adapter-name tfix375 \\
        --include-base \\
        --include-llmlingua2 \\
        --out data/bakeoff_downstream_compressions.jsonl

LLMLingua / LongLLMLingua / API baselines are out of scope for this runner
because their integration is independent of the GPU base model. They have
their own runners (see `run_lingua_bakeoff.py`, `run_api_bakeoff.py` --
written when needed). Keeping the GPU and non-GPU paths separate avoids
loading 4-bit Qwen unnecessarily when running CPU baselines.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .conversations import Conversation, load_jsonl

logger = logging.getLogger("downstream_bakeoff")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)

# Same per-tier caps as legacy bake-off (defined in baselines._qwen_runtime
# but imported lazily so the CLI starts fast on machines without torch).


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--conversations",
        type=Path,
        default=Path("data/bakeoff_conversations.jsonl"),
        help="Structured-turn conversations file (Phase B.0 schema).",
    )
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter directory (e.g., checkpoints/.../checkpoint-375). Required when --include-lora is set.",
    )
    p.add_argument(
        "--adapter-name",
        type=str,
        default="tfix375",
        help="Short name for the LoRA adapter in result `source` field.",
    )
    p.add_argument(
        "--include-base",
        action="store_true",
        help="Run base Qwen2.5-7B-Instruct (no adapter).",
    )
    p.add_argument(
        "--include-lora",
        action="store_true",
        help="Run the LoRA-adapted Qwen. Requires --adapter.",
    )
    p.add_argument(
        "--include-llmlingua2",
        action="store_true",
        help="Run LLMLingua-2 extractive (small GPU footprint, can share VRAM).",
    )
    p.add_argument(
        "--include-longllmlingua",
        action="store_true",
        help="Run LongLLMLingua (heavier: loads its own Llama-2-7B small LM).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/bakeoff_downstream_compressions.jsonl"),
        help="Output JSONL path.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned (baseline, conversation, tier) combinations without running.",
    )
    args = p.parse_args()

    convs = load_jsonl(args.conversations)
    logger.info("Loaded %d conversations from %s", len(convs), args.conversations)

    # Build the holdout inputs first so we never accidentally pass full
    # conversations into the compressors.
    holdout_inputs: list[tuple[Conversation, str, str]] = []
    for c in convs:
        prior, last_user, held = c.with_holdout()
        # The compressor's input is the flattened PRIOR turns ONLY.
        # last_user is preserved for the M_downstream continuation step
        # (recorded in the result so the judge step doesn't need to re-derive).
        # held is preserved as ground truth for the judge.
        holdout_inputs.append((prior, last_user.content, held.content))
        logger.info(
            "%s: prior=%d turns / %d chars, last_user=%d chars, held_asst=%d chars",
            c.id,
            prior.num_turns,
            sum(len(t.content) for t in prior.turns),
            len(last_user.content),
            len(held.content),
        )

    if args.include_lora and args.adapter is None:
        p.error("--include-lora requires --adapter")

    # Lazy imports so the CLI is usable on the laptop for --dry-run inspection
    if not args.dry_run:
        from compressor.baselines import CompressionRequest
        from compressor.baselines._qwen_runtime import TIERS

        baselines = []
        if args.include_base:
            from compressor.baselines.base_qwen import BaseQwenBaseline

            baselines.append(BaseQwenBaseline())
        if args.include_lora:
            from compressor.baselines.qwen_lora import QwenLoRABaseline

            baselines.append(QwenLoRABaseline(args.adapter, args.adapter_name))
        if args.include_llmlingua2:
            from compressor.baselines.lingua import LLMLingua2Baseline

            baselines.append(LLMLingua2Baseline())
        if args.include_longllmlingua:
            from compressor.baselines.lingua import LongLLMLinguaBaseline

            baselines.append(LongLLMLinguaBaseline())

        if not baselines:
            p.error("No baselines selected (use --include-base / --include-lora / etc.)")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Append mode lets us run multiple baselines in separate invocations
        # (e.g., GPU baselines on pod, API baselines locally) and accumulate.
        # We rotate the file on first writer per session for safety:
        if args.out.exists():
            backup = args.out.with_suffix(args.out.suffix + f".bak-{int(time.time())}")
            logger.warning("Output file exists; backing up to %s", backup)
            args.out.rename(backup)

        n_total = len(baselines) * len(convs) * len(TIERS)
        done = 0
        overall_t0 = time.time()

        with args.out.open("w", encoding="utf-8") as fh:
            for baseline in baselines:
                baseline.load()
                try:
                    for (prior, last_user_content, held_content), conv in zip(
                        holdout_inputs, convs
                    ):
                        prior_flat = prior.flatten()
                        for turn_age, ratio, _max_new in TIERS:
                            done += 1
                            req = CompressionRequest(
                                conversation=prior_flat,
                                turn_age=turn_age,
                                target_ratio=ratio,
                                conversation_id=conv.id,
                                scenario_type=conv.scenario_type,
                            )
                            res = baseline.compress(req)
                            row = res.to_jsonl_dict()
                            # Extra fields specific to downstream-input results:
                            # carry the holdout pair so the judge runner doesn't
                            # need to re-derive them from the source corpus.
                            row["downstream"] = {
                                "last_user_turn": last_user_content,
                                "held_assistant_turn": held_content,
                                "prior_num_turns": prior.num_turns,
                                "prior_chars": len(prior_flat),
                            }
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            fh.flush()
                            logger.info(
                                "[%d/%d] %s  age=%-6s  src=%-22s  in=%d  out=%d  ratio=%.1fx  %.1fs",
                                done,
                                n_total,
                                conv.id,
                                turn_age,
                                baseline.name,
                                res.input_chars,
                                res.output_chars,
                                res.achieved_ratio,
                                res.gen_seconds,
                            )
                finally:
                    baseline.unload()

        overall_dt = time.time() - overall_t0
        logger.info(
            "All %d generations done in %.1fs (-> %s)", n_total, overall_dt, args.out
        )
    else:
        # Dry run: print the planned (baseline, conversation, tier) combinations
        from compressor.baselines._qwen_runtime import TIERS

        planned_baselines = []
        if args.include_base:
            planned_baselines.append("base-qwen")
        if args.include_lora:
            planned_baselines.append(args.adapter_name)
        if args.include_llmlingua2:
            planned_baselines.append("llmlingua2")
        if args.include_longllmlingua:
            planned_baselines.append("longllmlingua")
        n = len(planned_baselines) * len(convs) * len(TIERS)
        logger.info(
            "[dry-run] %d generations planned: %d baselines x %d conversations x %d tiers",
            n,
            len(planned_baselines),
            len(convs),
            len(TIERS),
        )
        for b in planned_baselines:
            logger.info("  baseline: %s", b)


if __name__ == "__main__":
    main()
