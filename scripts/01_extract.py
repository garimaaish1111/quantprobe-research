"""Phase 2 - Stage A extraction driver.

Runs one or more precision arms for one model and caches the hidden states.

Usage
-----
    # smoke first, always
    python scripts/01_extract.py --config configs/model_smoke.yaml --limit 10

    # the real thing, on a CUDA machine
    python scripts/01_extract.py --config configs/model_llama1b.yaml

    # one arm at a time, e.g. after a Colab disconnect
    python scripts/01_extract.py --config configs/model_llama1b.yaml \
        --precisions int4 --resume

    # verify the done-signal once all three arms exist
    python scripts/01_extract.py --config configs/model_llama1b.yaml --verify-only

Each arm loads the model, extracts, writes the array plus sidecar, then frees
the model before the next arm loads. Two quantized 1B models resident at once
would be pointless pressure on a free-tier T4.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths  # noqa: E402
from src.config import load_config, require  # noqa: E402
from src.data.loaders import DataError, load_source  # noqa: E402
from src.data.schema import summarize  # noqa: E402
from src.extract.hidden import (  # noqa: E402
    ExtractionError,
    cross_precision_token_check,
    extract_hidden_states,
    extraction_dir,
    is_complete,
)
from src.logging_utils import provenance, run_id, setup_logging, write_json  # noqa: E402
from src.models.loader import ModelLoadError, load_model_and_tokenizer, resolve_device  # noqa: E402
from src.seeding import set_seed  # noqa: E402

RULE = "=" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantProbe Phase 2 extraction")
    parser.add_argument("--config", required=True, help="model YAML config")
    parser.add_argument(
        "--precisions",
        nargs="*",
        default=None,
        help="subset of cfg.precisions to run. Default: all of them.",
    )
    parser.add_argument("--dataset", default="true_false", help="which dataset to extract")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N records only. Use for smoke runs."
    )
    parser.add_argument("--batch-size", type=int, default=None, help="override cfg.extract.batch_size")
    parser.add_argument("--resume", action="store_true", help="continue a partial run")
    parser.add_argument("--force", action="store_true", help="re-extract even if complete")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip extraction; just run the cross-precision token check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging(log_dir=paths.logs_dir(), run_name=run_id("extract"), level=logging.INFO)

    cfg = load_config(args.config)
    require(cfg, "seed", "model.hf_id", "model.short_name", "quantization", "extract", "precisions")

    seed_record = set_seed(cfg.seed)
    cache_root = Path(cfg.paths.cache_root)
    model_short = cfg.model.short_name
    precisions = args.precisions if args.precisions else list(cfg.precisions)
    batch_size = args.batch_size or int(cfg.extract.batch_size)

    log.info(RULE)
    log.info("QuantProbe Phase 2 - Stage A extraction")
    log.info(RULE)
    log.info("model      : %s (%s)", cfg.model.hf_id, model_short)
    log.info("precisions : %s", precisions)
    log.info("cache root : %s", cache_root)
    log.info("device     : %s", resolve_device(args.device))
    log.info("seed       : %s", seed_record["master_seed"])

    # -------------------------------------------------------------- dataset
    try:
        records = load_source(args.dataset, cache_root)
    except DataError as exc:
        log.error("dataset load failed: %s", exc)
        return 1

    if args.limit is not None:
        records = records[: args.limit]
        log.warning("LIMIT active: using only the first %d records (smoke run)", len(records))

    stats = summarize(records)
    log.info(
        "dataset    : %s | n=%d | positive rate %.3f | %d topics",
        args.dataset, stats["n"], stats["positive_rate"], stats["n_topics"],
    )

    if args.verify_only:
        return _verify(cache_root, model_short, precisions, log)

    # ---------------------------------------------------------- extraction
    for precision in precisions:
        directory = extraction_dir(cache_root, model_short, precision)
        log.info("")
        log.info(RULE)
        log.info("PRECISION: %s  ->  %s", precision, directory)
        log.info(RULE)

        if is_complete(directory, len(records)) and not args.force:
            log.info("already complete (%d rows). Use --force to redo.", len(records))
            continue

        model = tokenizer = None
        try:
            model, tokenizer, model_meta = load_model_and_tokenizer(cfg, precision, args.device)
            log.info(
                "loaded %s | %s hidden states | d_model=%s | %s",
                model_meta["architecture"],
                model_meta["n_hidden_states"],
                model_meta["d_model"],
                model_meta["effective_load_dtype"],
            )
            # Printed as well as asserted. The assertion in the loader is what
            # protects the result; this line is what lets a human confirm at a
            # glance that the INT arms really are quantized.
            log.info("param dtype counts: %s", model_meta["param_dtype_counts"])

            extract_hidden_states(
                model=model,
                tokenizer=tokenizer,
                records=records,
                directory=directory,
                model_meta=model_meta,
                cfg=cfg,
                precision=precision,
                seed_record=seed_record,
                batch_size=batch_size,
                resume=args.resume,
            )
        except (ModelLoadError, ExtractionError) as exc:
            log.error("precision %s FAILED:\n%s", precision, exc)
            return 1
        finally:
            # Free before the next arm loads. On a free T4 the difference
            # between doing this and not is an OOM on the third arm.
            del model, tokenizer
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    return _verify(cache_root, model_short, precisions, log)


def _verify(cache_root: Path, model_short: str, precisions, log) -> int:
    """The Phase 2 done-signal."""
    log.info("")
    log.info(RULE)
    log.info("CROSS-PRECISION TOKEN ALIGNMENT CHECK")
    log.info(RULE)

    try:
        result = cross_precision_token_check(cache_root, model_short, precisions)
    except ExtractionError as exc:
        log.error("TOKEN ALIGNMENT FAILED:\n%s", exc)
        return 1

    write_json(
        paths.metrics_dir() / f"phase2_extraction_{model_short}.json",
        provenance({"phase": "2-extraction", "token_alignment": result}),
    )

    log.info("")
    log.info("all arms saw byte-identical tokens: %s", result["shared_fingerprint"][:16])
    log.info(RULE)
    log.info("PHASE 2 EXTRACTION COMPLETE")
    log.info(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
