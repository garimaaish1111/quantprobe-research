"""Phase 5b - the reduction demo.

Detection is the project's stated goal. This turns it into an intervention:
use the probe to refuse low-confidence assertions, and report what that buys.

    "At 70% coverage the INT4 model's false-assertion rate falls from 49.6%
     to X%, a Y% reduction."

That is the same fact as an AUROC, in the units someone deploying it would
use. Runs on CPU from the cached arrays - no model, no GPU.

Usage
-----
    python scripts/05_abstention.py --config configs/experiment_e1.yaml
    python scripts/05_abstention.py --config configs/experiment_e1.yaml --layer 11
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths  # noqa: E402
from src.analysis.abstention import evaluate_with_abstention  # noqa: E402
from src.config import load_config, require  # noqa: E402
from src.data.loaders import load_source  # noqa: E402
from src.data.splits import make_folds  # noqa: E402
from src.extract.hidden import ExtractionError, extraction_dir, load_extraction  # noqa: E402
from src.logging_utils import provenance, run_id, setup_logging, write_json  # noqa: E402
from src.probes.linear import align_records  # noqa: E402
from src.seeding import set_seed  # noqa: E402
from src.viz.plots import plot_coverage_risk  # noqa: E402

RULE = "=" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantProbe - abstention / reduction demo")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment_e1.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default="true_false")
    parser.add_argument("--layer", type=int, default=None,
                        help="default: the best layer from E1")
    parser.add_argument("--split", default="leave_one_topic_out",
                        choices=["leave_one_topic_out", "group_kfold"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging(log_dir=paths.logs_dir(), run_name=run_id("abstain"), level=logging.INFO)

    cfg = load_config(args.config)
    require(cfg, "seed", "probe", "experiment", "paths.cache_root")
    seed_record = set_seed(cfg.seed)

    cache_root = Path(cfg.paths.cache_root)
    model_short = args.model or cfg.experiment.model_short_name
    precisions = list(cfg.experiment.precisions)

    # Layer: the one E1 found strongest, unless overridden. Picking it from E1
    # rather than by sweeping here avoids choosing a layer on the same data
    # the curve is then reported on.
    layer = args.layer
    if layer is None:
        e1_path = paths.metrics_dir() / f"e1_per_layer_auroc_{model_short}.json"
        if not e1_path.is_file():
            log.error("no E1 results at %s - pass --layer or run scripts/02_probe.py", e1_path)
            return 1
        import json

        e1 = json.loads(e1_path.read_text(encoding="utf-8"))
        layer = int(e1["best_layer"]["fp16"]["layer"])

    log.info(RULE)
    log.info("QuantProbe - probe-guided abstention (reduction demo)")
    log.info(RULE)
    log.info("model %s | layer %d | precisions %s", model_short, layer, precisions)

    records = load_source(args.dataset, cache_root)
    arms = {}
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

    results = {}
    for precision in precisions:
        features = np.asarray(arms[precision]["hidden"][:, layer, :])
        results[precision] = evaluate_with_abstention(
            features, labels, folds,
            seed=int(cfg.probe.random_state),
            C=float(cfg.probe.C), max_iter=int(cfg.probe.max_iter),
        )

    log.info("")
    log.info(RULE)
    log.info("What probe-guided abstention buys, at layer %d", layer)
    log.info(RULE)
    log.info("  %-6s %10s %14s %14s %12s", "prec", "coverage", "false assert", "vs baseline", "reduction")
    for precision in precisions:
        r = results[precision]
        base = r["baseline_false_rate"]
        for key in ("coverage_90", "coverage_70", "coverage_50"):
            op = r["operating_points"][key]
            log.info("  %-6s %10.1f%% %13.1f%% %13.1f%% %11.1f%%",
                     precision if key == "coverage_90" else "",
                     op["coverage"] * 100,
                     op["false_assertion_rate"] * 100,
                     base * 100,
                     op["risk_reduction"] * 100)
        log.info("  %-6s AUC(coverage-risk) = %.4f   (lower is better; baseline %.4f)",
                 "", r["auc_coverage_risk"], base)

    log.info("")
    log.info("Read it as: 'answering only the %d%% it is most confident about, the",
             70)
    log.info("model's false-assertion rate falls from X%% to Y%%.' That is the")
    log.info("detection result restated as an intervention.")

    figure = plot_coverage_risk(
        results, paths.figures_dir() / f"abstention_{model_short}.png",
        layer=layer, model_name=model_short,
    )
    metrics_path = write_json(
        paths.metrics_dir() / f"abstention_{model_short}.json",
        provenance({
            "phase": "5b-abstention",
            "model_short_name": model_short,
            "layer": layer,
            "split_scheme": args.split,
            "seeds": seed_record,
            "results": results,
            "figure": str(figure),
        }),
    )
    log.info("")
    log.info("metrics -> %s", metrics_path)
    log.info("figure  -> %s", figure)
    log.info(RULE)
    log.info("ABSTENTION DEMO COMPLETE")
    log.info(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
