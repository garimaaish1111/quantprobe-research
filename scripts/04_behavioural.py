"""Phase 5 - E5, the behavioural control.

Measures whether the quantized model is actually worse *at the task*, so that
a probe result can be interpreted. If INT4 accuracy is unchanged but probe
AUROC drops, the detector broke, not the model. That is the sharpest possible
result.

Two independent behavioural measures, both by likelihood scoring rather than
generation (see src/eval/behavioural.py for why):

  * TruthfulQA MC1  - pick the true answer among distractors
  * HaluEval QA     - score the true answer above the hallucinated one

Needs the model, so it needs CUDA for the INT8/INT4 arms.

Usage
-----
    python scripts/04_behavioural.py --config configs/model_llama1b.yaml --limit 50
    python scripts/04_behavioural.py --config configs/model_llama1b.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths  # noqa: E402
from src.config import load_config, require  # noqa: E402
from src.data.loaders import DataError, load_source  # noqa: E402
from src.eval.behavioural import (  # noqa: E402
    BehaviouralError,
    binomial_ci,
    judge_records,
    judged_discrimination,
    length_confound_report,
    multiple_choice_accuracy,
    score_records,
)
from src.logging_utils import provenance, run_id, setup_logging, write_json  # noqa: E402
from src.models.loader import ModelLoadError, load_model_and_tokenizer  # noqa: E402
from src.seeding import set_seed  # noqa: E402

RULE = "=" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantProbe Phase 5 - E5 behavioural control")
    parser.add_argument("--config", required=True)
    parser.add_argument("--precisions", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None,
                        help="questions/pairs per task. Use for smoke runs.")
    parser.add_argument("--halueval-limit", type=int, default=1500,
                        help="HaluEval source rows (each yields 2 scorings). "
                             "10k rows x 2 x 3 arms is 60k forward passes for "
                             "no extra resolution.")
    parser.add_argument("--include-knowledge", action="store_true",
                        help="give HaluEval's passage to the model. Off by default: "
                             "closed-book is what 'hallucinate' means here.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging(log_dir=paths.logs_dir(), run_name=run_id("behavioural"), level=logging.INFO)

    cfg = load_config(args.config)
    require(cfg, "seed", "model.hf_id", "model.short_name", "quantization", "precisions")
    seed_record = set_seed(cfg.seed)

    model_short = cfg.model.short_name
    precisions = args.precisions or list(cfg.precisions)
    cache_root = Path(cfg.paths.cache_root)

    log.info(RULE)
    log.info("QuantProbe Phase 5 - E5 behavioural control")
    log.info(RULE)
    log.info("model      : %s (%s)", cfg.model.hf_id, model_short)
    log.info("precisions : %s", precisions)
    log.info("knowledge  : %s", "included" if args.include_knowledge else "closed-book")

    # ------------------------------------------------------------- datasets
    try:
        mc1 = load_source("truthfulqa_mc1", cache_root)
        halu = load_source("halueval_qa", cache_root, limit=args.halueval_limit)
    except DataError as exc:
        log.error("dataset load failed: %s", exc)
        return 1

    if args.limit:
        keep_q = {r.meta["question_index"] for r in mc1}
        keep_q = set(sorted(keep_q)[: args.limit])
        mc1 = [r for r in mc1 if r.meta["question_index"] in keep_q]
        keep_r = {r.meta["row_index"] for r in halu}
        keep_r = set(sorted(keep_r)[: args.limit])
        halu = [r for r in halu if r.meta["row_index"] in keep_r]
        log.warning("LIMIT active: %d MC1 rows, %d HaluEval rows", len(mc1), len(halu))

    log.info("truthfulqa_mc1 : %d candidate answers across %d questions",
             len(mc1), len({r.meta["question_index"] for r in mc1}))
    log.info("halueval_qa    : %d answers across %d question pairs",
             len(halu), len({r.meta["row_index"] for r in halu}))

    # ---------------------------------------------------------- per precision
    results: dict[str, dict] = {}
    for precision in precisions:
        log.info("")
        log.info(RULE)
        log.info("PRECISION: %s", precision)
        log.info(RULE)

        model = tokenizer = None
        try:
            model, tokenizer, model_meta = load_model_and_tokenizer(cfg, precision, args.device)
            log.info("loaded %s | %s", model_meta["architecture"],
                     model_meta["effective_load_dtype"])
            log.info("param dtype counts: %s", model_meta["param_dtype_counts"])

            log.info("scoring TruthfulQA MC1 ...")
            mc1_scored = score_records(
                model, tokenizer, mc1, batch_size=args.batch_size,
                max_length=args.max_length,
            )
            mc1_acc = multiple_choice_accuracy(mc1_scored)

            # Length diagnostic on MC1. If "always pick the shortest" scores
            # as well as the model, length is being measured, not knowledge.
            mc1_len = length_confound_report(mc1_scored, "question_index")

            # HaluEval by yes/no JUDGEMENT, not by answer likelihood. Its true
            # answers are extractive spans and its false ones are fluent
            # sentences, so likelihood scoring measures length: summed logprob
            # gave 1.00 accuracy and mean logprob gave 0.00 on the same data.
            log.info("judging HaluEval QA (yes/no, length-immune) ...")
            halu_scored = judge_records(
                model, tokenizer, halu, batch_size=args.batch_size,
                max_length=args.max_length, include_knowledge=args.include_knowledge,
            )
            halu_acc = judged_discrimination(halu_scored)

            n = mc1_acc["n_questions"]
            for metric in ("accuracy_logprob_sum", "accuracy_logprob_mean"):
                mc1_acc[metric + "_ci"] = list(binomial_ci(int(round(mc1_acc[metric] * n)), n))
            halu_acc["accuracy_ci"] = list(
                binomial_ci(halu_acc["n_correct"], halu_acc["n_pairs"])
            )

            results[precision] = {
                "truthfulqa_mc1": mc1_acc,
                "truthfulqa_mc1_length_confound": mc1_len,
                "halueval_qa": halu_acc,
                "model": model_meta,
            }

            log.info("")
            log.info("  TruthfulQA MC1  acc(sum)  %.4f  [%.4f, %.4f]   chance %.3f  n=%d",
                     mc1_acc["accuracy_logprob_sum"], *mc1_acc["accuracy_logprob_sum_ci"],
                     mc1_acc["chance"], mc1_acc["n_questions"])
            log.info("  TruthfulQA MC1  acc(mean) %.4f  [%.4f, %.4f]",
                     mc1_acc["accuracy_logprob_mean"], *mc1_acc["accuracy_logprob_mean_ci"])
            log.info("  HaluEval QA     judge     %.4f  [%.4f, %.4f]   chance 0.500  n=%d",
                     halu_acc["accuracy"], *halu_acc["accuracy_ci"], halu_acc["n_pairs"])
            if mc1_len.get("n"):
                log.info("  [diagnostic] MC1 'always shortest' %.4f - 'always longest' %.4f",
                         mc1_len["accuracy_if_always_shortest"],
                         mc1_len["accuracy_if_always_longest"])
                if mc1_len["accuracy_if_always_shortest"] >= mc1_acc["accuracy_logprob_sum"]:
                    log.warning("  !! picking the shortest answer scores as well as the model "
                                "- MC1 is measuring length here, not knowledge")

        except (ModelLoadError, BehaviouralError) as exc:
            log.error("precision %s FAILED:\n%s", precision, exc)
            return 1
        finally:
            del model, tokenizer
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # --------------------------------- probe vs behaviour, side by side
    probe_path = paths.metrics_dir() / f"e1_per_layer_auroc_{model_short}.json"
    probe_best: dict[str, float] = {}
    if probe_path.is_file():
        import json

        e1 = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_best = {p: e1["best_layer"][p]["auroc_mean"] for p in e1["best_layer"]}

    log.info("")
    log.info(RULE)
    log.info("E5 - TASK ACCURACY NEXT TO PROBE AUROC  (the Phase 5 done-signal)")
    log.info(RULE)
    log.info("  %-6s %14s %14s %14s", "prec", "MC1 acc", "HaluEval acc", "probe AUROC")
    for precision in precisions:
        r = results[precision]
        log.info("  %-6s %14.4f %14.4f %14s",
                 precision,
                 r["truthfulqa_mc1"]["accuracy_logprob_sum"],
                 r["halueval_qa"]["accuracy"],
                 f"{probe_best[precision]:.4f}" if precision in probe_best else "n/a")
    log.info("")
    if probe_best:
        log.info("Read it this way: if task accuracy holds but probe AUROC drops, the")
        log.info("DETECTOR broke. If both hold, quantization is benign here. If task")
        log.info("accuracy drops too, the model itself degraded and the probe result")
        log.info("is a corollary rather than a finding about probes.")
    else:
        log.warning("no E1 probe results found at %s - run scripts/02_probe.py "
                    "to complete the comparison", probe_path)

    metrics_path = write_json(
        paths.metrics_dir() / f"e5_behavioural_{model_short}.json",
        provenance({
            "phase": "5-behavioural",
            "experiment": "E5",
            "model_short_name": model_short,
            "precisions": precisions,
            "seeds": seed_record,
            "scoring": "likelihood (no generation)",
            "include_knowledge": bool(args.include_knowledge),
            "halueval_source_rows": args.halueval_limit,
            "results": {p: {k: v for k, v in results[p].items() if k != "model"}
                        for p in results},
            "probe_best_auroc": probe_best,
        }),
    )
    log.info("")
    log.info("metrics -> %s", metrics_path)
    log.info(RULE)
    log.info("PHASE 5 COMPLETE")
    log.info(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
