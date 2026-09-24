"""Phase 4 - E2 (transfer), E3 (drift), E4 (probe rotation), and the
restated H4 correlation.

Reads only cached arrays. No model, no GPU.

Usage
-----
    python scripts/03_analyze.py --config configs/experiment_e1.yaml
    python scripts/03_analyze.py --config configs/experiment_e1.yaml --jobs 6
    python scripts/03_analyze.py --config configs/experiment_e1.yaml --layers 7 10 11
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths  # noqa: E402
from src.analysis.drift import correlate, drift_by_layer  # noqa: E402
from src.analysis.transfer import (  # noqa: E402
    aggregate_transfer,
    cross_arm_angles,
    within_arm_angles,
    best_transfer_layer,
    matrix_at_layer,
    transfer_loss,
    transfer_matrix_for_layer,
)
from src.config import load_config, require  # noqa: E402
from src.data.loaders import load_source  # noqa: E402
from src.data.splits import make_folds  # noqa: E402
from src.extract.hidden import ExtractionError, extraction_dir, load_extraction  # noqa: E402
from src.logging_utils import provenance, run_id, setup_logging, write_json  # noqa: E402
from src.probes.linear import align_records  # noqa: E402
from src.seeding import set_seed  # noqa: E402
from src.viz.plots import plot_e2_transfer_heatmap, plot_e3_drift, plot_e4_rotation  # noqa: E402

RULE = "=" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantProbe Phase 4 - E2/E3/E4")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment_e1.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default="true_false")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--split", default="leave_one_topic_out",
                        choices=["leave_one_topic_out", "group_kfold"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging(log_dir=paths.logs_dir(), run_name=run_id("analyze"), level=logging.INFO)

    cfg = load_config(args.config)
    require(cfg, "seed", "probe", "experiment", "paths.cache_root")
    seed_record = set_seed(cfg.seed)

    cache_root = Path(cfg.paths.cache_root)
    model_short = args.model or cfg.experiment.model_short_name
    precisions = list(cfg.experiment.precisions)

    log.info(RULE)
    log.info("QuantProbe Phase 4 - E2 transfer, E3 drift, E4 rotation")
    log.info(RULE)
    log.info("model      : %s", model_short)
    log.info("precisions : %s", precisions)

    records = load_source(args.dataset, cache_root)
    arms: dict[str, dict] = {}
    for precision in precisions:
        try:
            arms[precision] = load_extraction(extraction_dir(cache_root, model_short, precision))
        except ExtractionError as exc:
            log.error("cannot load %s: %s", precision, exc)
            return 1

    idx = align_records(arms[precisions[0]]["ids"], records)
    ordered = [records[i] for i in idx]
    labels = np.array([r.label for r in ordered], dtype=np.int64)
    folds = make_folds(ordered, scheme=args.split, n_splits=int(cfg.probe.cv_folds))

    n_layers = arms[precisions[0]]["hidden"].shape[1]
    layers = args.layers if args.layers else list(range(n_layers))
    log.info("rows %d | layers %d | folds %d", len(ordered), len(layers), len(folds))

    # =============================================================== E2
    log.info("")
    log.info(RULE)
    log.info("E2 - cross-precision transfer (%d layers x 9 cells x %d folds)",
             len(layers), len(folds))
    log.info(RULE)

    def e2_layer(layer: int):
        features = {p: np.asarray(arms[p]["hidden"][:, layer, :]) for p in precisions}
        return transfer_matrix_for_layer(
            features, labels, folds, layer,
            seed=int(cfg.probe.random_state),
            C=float(cfg.probe.C), max_iter=int(cfg.probe.max_iter),
        )

    started = time.time()
    raw_transfer: list[dict] = []
    fold_directions: dict[int, dict] = {}
    if args.jobs > 1:
        from joblib import Parallel, delayed

        log.info("fitting %d layers with %d workers", len(layers), args.jobs)
        for L, out in zip(layers, Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(e2_layer)(L) for L in layers
        )):
            raw_transfer.extend(out["rows"])
            fold_directions[L] = out["directions"]
    else:
        for n, L in enumerate(layers, 1):
            out = e2_layer(L)
            raw_transfer.extend(out["rows"])
            fold_directions[L] = out["directions"]
            if n % 4 == 0 or n == len(layers):
                el = time.time() - started
                log.info("  layer %2d (%d/%d) | %.0fs | ~%.0fs left",
                         L, n, len(layers), el, el / n * (len(layers) - n))
    log.info("E2 done in %.0fs", time.time() - started)

    transfer_rows = aggregate_transfer(raw_transfer)
    loss_rows = transfer_loss(transfer_rows)

    peak = best_transfer_layer(transfer_rows, precisions)
    log.info("")
    log.info("strongest matched layer = %d (E2 read at this layer)", peak)
    log.info("")
    log.info("transfer matrix at layer %d  (rows = trained on, cols = tested on)", peak)
    header = "  %-14s" + " %10s" * len(precisions)
    log.info(header, "", *[p.upper() for p in precisions])
    matrix = matrix_at_layer(transfer_rows, peak, precisions)
    for i, train_p in enumerate(precisions):
        log.info("  train %-8s" + " %10.4f" * len(precisions), train_p, *matrix[i])

    log.info("")
    log.info("mean off-diagonal loss vs the native probe, averaged over all layers:")
    for train_p in precisions:
        for test_p in precisions:
            if train_p == test_p:
                continue
            vals = [r["loss"] for r in loss_rows
                    if r["train_precision"] == train_p and r["test_precision"] == test_p]
            log.info("  train %-5s -> test %-5s : %+.4f", train_p, test_p, float(np.mean(vals)))

    loss_matrix = np.array(
        [[next(r["loss"] for r in loss_rows
               if r["layer"] == peak and r["train_precision"] == a and r["test_precision"] == b)
          for b in precisions] for a in precisions]
    )

    # =============================================================== E3
    log.info("")
    log.info(RULE)
    log.info("E3 - representational drift from FP16")
    log.info(RULE)

    drift_rows: list[dict] = []
    reference = arms["fp16"]["hidden"]
    for precision in precisions:
        if precision == "fp16":
            continue
        for row in drift_by_layer(reference, arms[precision]["hidden"], layers):
            row["precision"] = precision
            drift_rows.append(row)

    log.info("  %5s %12s %12s %12s %12s %12s %12s",
             "layer", "rel(int8)", "cos(int8)", "cka(int8)", "rel(int4)", "cos(int4)", "cka(int4)")
    for L in layers:
        cells = []
        for p in ("int8", "int4"):
            m = [r for r in drift_rows if r["precision"] == p and r["layer"] == L]
            cells.extend([m[0]["relative_drift"], m[0]["mean_cosine"], m[0]["linear_cka"]]
                         if m else [float("nan")] * 3)
        log.info("  %5d %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f", L, *cells)

    # =============================================================== E4
    log.info("")
    log.info(RULE)
    log.info("E4 - probe direction rotation")
    log.info(RULE)

    fold_names = [f[0] for f in folds]
    rotation_rows: list[dict] = []

    for L in layers:
        dirs = fold_directions.get(L, {})
        if not dirs:
            continue
        # Noise floor FIRST: how far apart are two probes fitted on the SAME
        # representation, different folds? A 2048-dim probe fitted on ~5000
        # rows is under-determined, so this is not near zero, and any
        # cross-arm angle below it is indistinguishable from sampling noise.
        baseline = within_arm_angles(dirs, "fp16", fold_names)
        baseline_mean = float(np.mean(baseline)) if baseline else float("nan")

        for precision in precisions:
            if precision == "fp16":
                continue
            cross = cross_arm_angles(dirs, "fp16", precision, fold_names)
            if not cross:
                continue
            rotation_rows.append({
                "layer": L,
                "precision": precision,
                "angle_deg": float(np.mean(cross)),
                "angle_std": float(np.std(cross)),
                "baseline_deg": baseline_mean,
                "excess_deg": float(np.mean(cross)) - baseline_mean,
            })

    log.info("  %5s %12s %12s %12s %12s", "layer", "INT8 ang", "INT4 ang",
             "FP16 noise", "INT4 excess")
    log.info("  (FP16 noise = two FP16 probes on different folds; anything below it is not rotation)")
    for L in layers:
        i8 = [r for r in rotation_rows if r["precision"] == "int8" and r["layer"] == L]
        i4 = [r for r in rotation_rows if r["precision"] == "int4" and r["layer"] == L]
        if not i4:
            continue
        log.info("  %5d %12.2f %12.2f %12.2f %12.2f", L,
                 i8[0]["angle_deg"] if i8 else float("nan"),
                 i4[0]["angle_deg"], i4[0]["baseline_deg"], i4[0]["excess_deg"])

    # ================================================= restated H4
    log.info("")
    log.info(RULE)
    log.info("H4 (restated) - does drift predict TRANSFER LOSS, per layer?")
    log.info(RULE)
    log.info("E1 found no probe degradation, so the original H4 had a constant")
    log.info("dependent variable. Transfer loss from E2 is the quantity that varies.")
    log.info("")

    correlations: dict[str, dict] = {}
    for precision in precisions:
        if precision == "fp16":
            continue
        common = [L for L in layers
                  if any(r["precision"] == precision and r["layer"] == L for r in drift_rows)]
        drift_lookup = {r["layer"]: r for r in drift_rows if r["precision"] == precision}
        loss_lookup = {
            r["layer"]: r["loss"] for r in loss_rows
            if r["train_precision"] == "fp16" and r["test_precision"] == precision
        }
        common = [L for L in common if L in loss_lookup]
        losses = [loss_lookup[L] for L in common]

        for metric in ("relative_drift", "mean_cosine", "linear_cka"):
            values = [drift_lookup[L][metric] for L in common]
            try:
                stats_out = correlate(values, losses)
            except Exception as exc:
                log.warning("  %s vs %s: %s", metric, precision, exc)
                continue
            correlations[f"fp16_to_{precision}__{metric}"] = stats_out
            if stats_out.get("pearson_r") is None:
                log.info("  %-16s vs loss(fp16->%s): %s", metric, precision, stats_out["note"])
            else:
                log.info(
                    "  %-16s vs loss(fp16->%s): pearson r=%+.3f (p=%.3f)  "
                    "spearman rho=%+.3f (p=%.3f)  n=%d",
                    metric, precision, stats_out["pearson_r"], stats_out["pearson_p"],
                    stats_out["spearman_rho"], stats_out["spearman_p"], stats_out["n"],
                )

    # =============================================================== outputs
    fig_e2 = plot_e2_transfer_heatmap(
        matrix, loss_matrix, precisions,
        paths.figures_dir() / f"e2_transfer_{model_short}.png",
        layer=peak, model_name=model_short,
    )
    fig_e3 = plot_e3_drift(
        drift_rows, paths.figures_dir() / f"e3_drift_{model_short}.png", model_name=model_short
    )
    fig_e4 = plot_e4_rotation(
        rotation_rows, paths.figures_dir() / f"e4_rotation_{model_short}.png",
        model_name=model_short,
    )

    metrics_path = write_json(
        paths.metrics_dir() / f"e2_e3_e4_{model_short}.json",
        provenance({
            "phase": "4-transfer-drift-rotation",
            "model_short_name": model_short,
            "precisions": precisions,
            "split_scheme": args.split,
            "folds": [f[0] for f in folds],
            "seeds": seed_record,
            "peak_layer": peak,
            "e2_transfer": transfer_rows,
            "e2_loss": loss_rows,
            "e2_matrix_at_peak": matrix.tolist(),
            "e2_loss_matrix_at_peak": loss_matrix.tolist(),
            "e3_drift": drift_rows,
            "e4_rotation": rotation_rows,
            "h4_restated_correlations": correlations,
            "figures": [str(fig_e2), str(fig_e3), str(fig_e4)],
        }),
    )

    log.info("")
    log.info("metrics -> %s", metrics_path)
    log.info(RULE)
    log.info("PHASE 4 COMPLETE")
    log.info(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
