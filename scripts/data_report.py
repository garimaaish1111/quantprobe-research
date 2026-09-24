"""Phase 1 done-signal: load all three datasets and prove they are sane.

Prints, for each dataset: row counts, label balance per topic, and five
sample rows. Then builds the topic folds and verifies no topic appears on
both sides of any split.

Deliberately unnumbered. The numbered scripts (00_smoke_test, 01_extract,
02_probe, 03_analyze) are pipeline stages; this is a verification utility
that produces no artefact the pipeline consumes.

Usage
-----
    python scripts/data_report.py
    python scripts/data_report.py --no-cache --halueval-limit 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.loaders import DataError, load_source  # noqa: E402
from src.data.schema import LABEL_TRUE, summarize  # noqa: E402
from src.data.splits import fold_summary, make_folds  # noqa: E402
from src.logging_utils import provenance, run_id, setup_logging, write_json  # noqa: E402
from src.seeding import set_seed  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantProbe Phase 1 data report")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--no-cache", action="store_true", help="ignore the JSONL cache")
    parser.add_argument("--halueval-limit", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args()


def print_samples(records, n: int, log) -> None:
    """Five rows, printed in full. Always look at the actual data."""
    for rec in records[:n]:
        verdict = "TRUE " if rec.label == LABEL_TRUE else "FALSE"
        log.info("  [%s] label=%d (%s)  topic=%s", rec.id, rec.label, verdict, rec.topic)
        if rec.prompt:
            log.info("        prompt   : %s", _clip(rec.prompt))
        log.info("        statement: %s", _clip(rec.statement))
        log.info("        text()   : %s", _clip(rec.text()))
        if rec.meta:
            for key, value in rec.meta.items():
                log.info("        meta.%-8s: %s", key, _clip(str(value)))


def _clip(text: str, width: int = 150) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def print_balance(records, log) -> dict:
    stats = summarize(records)
    log.info(
        "  n=%d | true=%d false=%d | positive rate=%.3f | %d topics",
        stats["n"], stats["n_true"], stats["n_false"], stats["positive_rate"], stats["n_topics"],
    )
    log.info("  %-28s %8s %8s %8s %9s", "topic", "n", "true", "false", "pos.rate")
    log.info("  %s", THIN[:68])
    for topic, row in stats["per_topic"].items():
        log.info(
            "  %-28s %8d %8d %8d %9.3f",
            topic, row["n"], row["true"], row["false"], row["true"] / row["n"],
        )
    return stats


def main() -> int:
    args = parse_args()
    log = setup_logging(log_dir=paths.logs_dir(), run_name=run_id("data"), level=logging.INFO)

    cfg = load_config(args.config)
    seed_record = set_seed(cfg.seed)
    log.info("seed: %s", seed_record["master_seed"])

    cache_dir = None if args.no_cache else paths.cache_dir()
    log.info("dataset cache: %s", cache_dir or "DISABLED")

    plan = [
        ("true_false", {}),
        ("truthfulqa_mc1", {}),
        ("halueval_qa", {"limit": args.halueval_limit}),
    ]

    report: dict = {"datasets": {}}
    loaded: dict = {}

    for name, kwargs in plan:
        log.info("")
        log.info(RULE)
        log.info("DATASET: %s", name)
        log.info(RULE)
        try:
            records = load_source(name, cache_dir, **kwargs)
        except DataError as exc:
            log.error("FAILED to load %s: %s", name, exc)
            return 1

        loaded[name] = records
        log.info("")
        log.info("label balance per topic (1 = TRUE):")
        stats = print_balance(records, log)
        log.info("")
        log.info("first %d rows:", args.samples)
        print_samples(records, args.samples, log)
        report["datasets"][name] = stats

    # ------------------------------------------------------------- splitting
    log.info("")
    log.info(RULE)
    log.info("TOPIC-BASED SPLITS (probe dataset: true_false)")
    log.info(RULE)

    probe_records = loaded["true_false"]
    for scheme, n_splits in (("leave_one_topic_out", None), ("group_kfold", 5)):
        folds = make_folds(
            probe_records, scheme=scheme, n_splits=n_splits or 5
        )
        log.info("")
        log.info("scheme=%s -> %d folds", scheme, len(folds))
        log.info("  %-22s %9s %9s %14s %13s", "held out", "n_train", "n_test", "train pos.rate", "test pos.rate")
        log.info("  %s", THIN[:74])
        for row in fold_summary(probe_records, folds):
            log.info(
                "  %-22s %9d %9d %14.3f %13.3f",
                row["held_out"], row["n_train"], row["n_test"],
                row["train_positive_rate"], row["test_positive_rate"],
            )
        report[f"folds_{scheme}"] = fold_summary(probe_records, folds)

    # make_folds() already asserted no topic spans both sides of a fold.
    log.info("")
    log.info("fold verification passed: no topic appears on both sides of any split")

    sidecar = write_json(
        paths.metrics_dir() / "phase1_data_report.json",
        provenance({"phase": "1-data", "seeds": seed_record, **report}),
    )
    log.info("report written to %s", sidecar)

    log.info("")
    log.info(RULE)
    log.info("PHASE 1 DATA CHECKS PASSED")
    log.info(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
