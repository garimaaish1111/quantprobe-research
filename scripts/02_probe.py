"""Phase 3 - Stage B + E1.

Fits one probe per (precision, layer, fold) over the cached hidden states and
plots AUROC against layer, one curve per precision, with mean +/- std bands.

Reads ONLY from the extraction cache. No model is loaded, nothing touches a
GPU, and nothing here re-derives a hidden state.

Usage
-----
    python scripts/02_probe.py --config configs/experiment_e1.yaml
    python scripts/02_probe.py --config configs/experiment_e1.yaml --layers 0 8 16
    python scripts/02_probe.py --config configs/experiment_e1.yaml --jobs 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths  # noqa: E402
from src.config import load_config, require  # noqa: E402
from src.data.loaders import load_source  # noqa: E402
from src.data.splits import fold_summary, make_folds  # noqa: E402
from src.extract.hidden import ExtractionError, extraction_dir, load_extraction  # noqa: E402
from src.logging_utils import provenance, run_id, setup_logging, write_json  # noqa: E402
from src.probes.linear import (  # noqa: E402
    ProbeError,
    aggregate,
    align_records,
    best_layer,
    leakage_warnings,
    mean_direction,
    probe_layer,
)
from src.seeding import set_seed  # noqa: E402
from src.viz.plots import plot_e1_auroc_by_layer, plot_probe_comparison  # noqa: E402

RULE = "=" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantProbe Phase 3 - probes + E1")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment_e1.yaml"))
    parser.add_argument("--model", default=None, help="override experiment.model_short_name")
    parser.add_argument("--dataset", default="true_false")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="subset, for a smoke run")
    parser.add_argument("--precisions", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=1, help="parallel workers over layers")
    parser.add_argument(
        "--split",
        default="leave_one_topic_out",
        choices=["leave_one_topic_out", "group_kfold"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging(log_dir=paths.logs_dir(), run_name=run_id("probe"), level=logging.INFO)

    cfg = load_config(args.config)
    require(cfg, "seed", "probe", "experiment", "paths.cache_root")
    seed_record = set_seed(cfg.seed)

    cache_root = Path(cfg.paths.cache_root)
    model_short = args.model or cfg.experiment.model_short_name
    precisions = args.precisions or list(cfg.experiment.precisions)

    log.info(RULE)
    log.info("QuantProbe Phase 3 - Stage B + E1")
    log.info(RULE)
    log.info("model      : %s", model_short)
    log.info("precisions : %s", precisions)
    log.info("split      : %s", args.split)
    log.info("seed       : %s", seed_record["master_seed"])

    # ------------------------------------------------------------- load data
    records = load_source(args.dataset, cache_root)

    arms: dict[str, dict] = {}
    for precision in precisions:
        directory = extraction_dir(cache_root, model_short, precision)
        try:
            arms[precision] = load_extraction(directory)
        except ExtractionError as exc:
            log.error("cannot load %s arm: %s", precision, exc)
            return 1
        log.info("%-5s %s from %s", precision, arms[precision]["hidden"].shape, directory)

    # ------------------------------------------------- align by id, not index
    try:
        idx = align_records(arms[precisions[0]]["ids"], records)
    except ProbeError as exc:
        log.error("alignment failed: %s", exc)
        return 1

    ordered = [records[i] for i in idx]
    labels = np.array([r.label for r in ordered], dtype=np.int64)

    for precision, data in arms.items():
        if not np.array_equal(data["labels"], labels):
            log.error(
                "labels in the %s arm do not match the dataset. The cache and the "
                "dataset have diverged - re-extract.", precision,
            )
            return 1
        if data["ids"] != arms[precisions[0]]["ids"]:
            log.error("arm %s has a different row order to %s", precision, precisions[0])
            return 1
    log.info("row alignment verified across all arms (%d rows, by id)", len(ordered))

    # ---------------------------------------------------------------- folds
    folds = make_folds(ordered, scheme=args.split, n_splits=int(cfg.probe.cv_folds))
    log.info("%d folds: %s", len(folds), [f[0] for f in folds])
    for row in fold_summary(ordered, folds):
        log.info(
            "  held out %-12s train %5d  test %5d  test pos.rate %.3f",
            row["held_out"], row["n_train"], row["n_test"], row["test_positive_rate"],
        )

    n_layers = arms[precisions[0]]["hidden"].shape[1]
    layers = args.layers if args.layers else list(range(n_layers))
    log.info("probing %d layers x %d precisions x %d folds x 2 probe types = %d fits",
             len(layers), len(precisions), len(folds), len(layers) * len(precisions) * len(folds) * 2)

    # --------------------------------------------------------------- fitting
    started = time.time()
    all_results = []

    def work(precision: str, layer: int):
        features = np.asarray(arms[precision]["hidden"][:, layer, :])
        return probe_layer(
            features=features,
            labels=labels,
            folds=folds,
            precision=precision,
            layer=layer,
            seed=int(cfg.probe.random_state),
            C=float(cfg.probe.C),
            max_iter=int(cfg.probe.max_iter),
        )

    tasks = [(p, L) for p in precisions for L in layers]

    if args.jobs > 1:
        from joblib import Parallel, delayed

        # verbose=10 so joblib reports task completions. With verbose=0 the
        # run prints NOTHING between start and finish, which on a job that
        # takes minutes is indistinguishable from a hang.
        log.info("fitting %d tasks with %d parallel workers", len(tasks), args.jobs)
        chunks = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(work)(p, L) for p, L in tasks
        )
        for chunk in chunks:
            all_results.extend(chunk)
    else:
        for n, (precision, layer) in enumerate(tasks, start=1):
            all_results.extend(work(precision, layer))
            if n % 5 == 0 or n == len(tasks):
                elapsed = time.time() - started
                log.info("  %3d/%3d (precision=%s layer=%2d) | %.0fs elapsed | ~%.0fs left",
                         n, len(tasks), precision, layer, elapsed,
                         elapsed / n * (len(tasks) - n))

    log.info("fitted %d probes in %.0fs", len(all_results), time.time() - started)

    not_converged = sum(1 for r in all_results if not r.converged)
    if not_converged:
        log.warning(
            "%d/%d fits hit max_iter=%s without converging. Their coefficients are "
            "unreliable; raise probe.max_iter or lower probe.C.",
            not_converged, len(all_results), cfg.probe.max_iter,
        )

    # ------------------------------------------------------------ aggregate
    rows = aggregate(all_results)

    log.info("")
    log.info(RULE)
    log.info("E1 - best layer per precision (logistic probe)")
    log.info(RULE)
    log.info("  %-6s %6s %9s %9s %9s", "prec", "layer", "AUROC", "std", "acc")
    for precision in precisions:
        b = best_layer(rows, precision, "logistic")
        log.info("  %-6s %6d %9.4f %9.4f %9.4f",
                 precision, b["layer"], b["auroc_mean"], b["auroc_std"], b["accuracy_mean"])

    log.info("")
    log.info("per-layer AUROC (logistic), mean over folds:")
    log.info("  %5s %10s %10s %10s", "layer", *precisions)
    for layer in layers:
        cells = []
        for precision in precisions:
            match = [r for r in rows
                     if r["precision"] == precision and r["probe_type"] == "logistic"
                     and r["layer"] == layer]
            cells.append(f"{match[0]['auroc_mean']:.4f}" if match else "-")
        log.info("  %5d %10s %10s %10s", layer, *cells)

    # ------------------------------------------------------- honesty checks
    flags = leakage_warnings(rows, threshold=0.98)
    if flags:
        log.warning("")
        log.warning("!! RESULTS THAT LOOK TOO GOOD - treat as leakage until disproved !!")
        for flag in flags:
            log.warning("  %s", flag)

    # ------------------------------------------------------------- figures
    figure = plot_e1_auroc_by_layer(
        rows,
        paths.figures_dir() / f"e1_per_layer_auroc_{model_short}.png",
        probe_type="logistic",
        model_name=model_short,
        n_folds=len(folds),
    )
    comparison = plot_probe_comparison(
        rows,
        paths.figures_dir() / f"e1_probe_comparison_{model_short}.png",
        model_name=model_short,
    )

    # ------------------------------- probe directions, cached for E4 later
    directions_path = paths.metrics_dir() / f"probe_directions_{model_short}.npz"
    directions = {}
    for precision in precisions:
        for layer in layers:
            try:
                directions[f"{precision}_L{layer}"] = mean_direction(
                    all_results, precision, "logistic", layer
                )
            except ProbeError:
                continue
    np.savez_compressed(directions_path, **directions)
    log.info("saved %d probe directions to %s (E4 reads these)", len(directions), directions_path)

    # -------------------------------------------------------------- metrics
    metrics_path = write_json(
        paths.metrics_dir() / f"e1_per_layer_auroc_{model_short}.json",
        provenance({
            "phase": "3-probes",
            "experiment": "E1",
            "model_short_name": model_short,
            "precisions": precisions,
            "split_scheme": args.split,
            "folds": [f[0] for f in folds],
            "seeds": seed_record,
            "probe_config": cfg.probe.to_plain() if hasattr(cfg.probe, "to_plain") else dict(cfg.probe),
            "n_rows": len(ordered),
            "n_layers": n_layers,
            "n_fits": len(all_results),
            "n_not_converged": not_converged,
            "leakage_flags": flags,
            "best_layer": {
                p: {k: v for k, v in best_layer(rows, p, "logistic").items() if k != "per_fold"}
                for p in precisions
            },
            "results": rows,
            "figures": [str(figure), str(comparison)],
        }),
    )

    log.info("")
    log.info("metrics -> %s", metrics_path)
    log.info("figure  -> %s", figure)
    log.info(RULE)
    log.info("PHASE 3 COMPLETE" if not flags else "PHASE 3 COMPLETE - WITH LEAKAGE WARNINGS")
    log.info(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
