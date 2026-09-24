"""Phase 0 smoke test.

Smoke test first, always: 10 examples, 1 layer, 1 precision, before any
full run.

This proves the whole Stage A code path end to end at N=10:

    config load -> seeding -> model load at one precision -> identical
    tokenization -> forward with output_hidden_states -> last-real-token
    slice -> shape assertions -> provenance sidecar

It deliberately does NOT touch the real datasets (Phase 1) or write to the
activation cache (Phase 2). Ten statements are inlined below in the eventual
unified schema so the shape of that schema is exercised too.

Usage
-----
    python scripts/00_smoke_test.py --config configs/model_smoke.yaml
    python scripts/00_smoke_test.py --config configs/model_llama1b.yaml \
        --precision int4          # Colab only: needs CUDA
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repo root on sys.path so `from src...` works when run as a script from
# anywhere, without requiring an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths  # noqa: E402
from src.config import load_config, require  # noqa: E402
from src.logging_utils import (  # noqa: E402
    provenance,
    run_id,
    setup_logging,
    write_json,
)
from src.models.loader import (  # noqa: E402
    ModelLoadError,
    load_model_and_tokenizer,
    resolve_device,
)
from src.seeding import set_seed  # noqa: E402

# Ten statements in the unified schema (src/data/schema.py). Five true, five false,
# two topics. Not research data - a fixture whose only job is to have a known
# label balance and short, unambiguous text.
SMOKE_STATEMENTS: list[dict] = [
    {"id": "smoke-0", "statement": "The capital of France is Paris.", "label": 1, "topic": "cities"},
    {"id": "smoke-1", "statement": "The capital of Japan is Madrid.", "label": 0, "topic": "cities"},
    {"id": "smoke-2", "statement": "Cairo is located in Egypt.", "label": 1, "topic": "cities"},
    {"id": "smoke-3", "statement": "Sydney is the capital city of Australia.", "label": 0, "topic": "cities"},
    {"id": "smoke-4", "statement": "Lisbon is a city in Portugal.", "label": 1, "topic": "cities"},
    {"id": "smoke-5", "statement": "Gold is a chemical element.", "label": 1, "topic": "elements"},
    {"id": "smoke-6", "statement": "Helium is heavier than lead.", "label": 0, "topic": "elements"},
    {"id": "smoke-7", "statement": "Oxygen has the chemical symbol O.", "label": 1, "topic": "elements"},
    {"id": "smoke-8", "statement": "The symbol for iron is Xe.", "label": 0, "topic": "elements"},
    {"id": "smoke-9", "statement": "Carbon has an atomic number of ninety.", "label": 0, "topic": "elements"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantProbe Phase 0 smoke test")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "model_smoke.yaml"),
        help="path to a model YAML config",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="fp16 | int8 | int4. Defaults to the first entry in cfg.precisions.",
    )
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument("--n", type=int, default=10, help="how many statements to run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger = setup_logging(
        log_dir=paths.logs_dir(),
        run_name=run_id("smoke"),
        level=logging.INFO,
    )

    logger.info("=" * 72)
    logger.info("QuantProbe Phase 0 smoke test")
    logger.info("=" * 72)

    # ---------------------------------------------------------------- config
    cfg = load_config(args.config)
    require(cfg, "seed", "model.hf_id", "quantization", "extract.max_length", "precisions")
    logger.info("config: %s", cfg["_config_path"])
    logger.info("model:  %s", cfg.model.hf_id)

    precision = args.precision or cfg.precisions[0]
    device = resolve_device(args.device)
    logger.info("precision: %s | device: %s", precision, device)

    # ----------------------------------------------------------------- seeds
    seed_record = set_seed(cfg.seed)
    logger.info("seeds: %s", seed_record)

    # ------------------------------------------------------------ disk check
    cache_root = Path(cfg.paths.cache_root)
    free_gb = paths.free_space_gb(cache_root)
    logger.info("cache_root: %s (%.1f GB free on that volume)", cache_root, free_gb)

    # ------------------------------------------------------------ model load
    try:
        model, tokenizer, model_meta = load_model_and_tokenizer(cfg, precision, device)
    except ModelLoadError as exc:
        logger.error("MODEL LOAD FAILED\n%s", exc)
        return 1

    logger.info(
        "loaded %s | %s layers (%s hidden states) | d_model=%s | dtype=%s",
        model_meta["architecture"],
        model_meta["n_layers"],
        model_meta["n_hidden_states"],
        model_meta["d_model"],
        model_meta["effective_load_dtype"],
    )
    logger.info("param dtype counts: %s", model_meta["param_dtype_counts"])

    # ------------------------------------------------------------- tokenize
    import torch

    batch = SMOKE_STATEMENTS[: args.n]
    texts = [row["statement"] for row in batch]

    # Right padding, and the true last token found from the attention mask.
    #
    # This is the single easiest place in the project to silently produce
    # garbage. With right padding, position -1 of a short sequence is a PAD
    # token, so "the last token" would mean different things for different
    # examples. Left padding fixes that but changes position ids, which
    # perturbs the very activations we are trying to compare. So: right pad,
    # then index by (attention_mask.sum(dim=1) - 1).
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.extract.max_length,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    lengths = attention_mask.sum(dim=1)

    logger.info("tokenized batch: input_ids %s", tuple(input_ids.shape))
    logger.info("true token counts per example: %s", lengths.tolist())

    # Truncation would mean we probed a cut-off sentence. Fail loudly.
    assert int(input_ids.shape[1]) < cfg.extract.max_length, (
        f"a statement hit max_length={cfg.extract.max_length}; it was truncated "
        f"and the 'last token' is not the end of the claim"
    )

    # ------------------------------------------------------------- forward
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    n_hidden = len(hidden_states)
    logger.info("forward returned %d hidden states (embeddings + %d blocks)", n_hidden, n_hidden - 1)
    logger.info("hidden_states[0] dtype: %s", hidden_states[0].dtype)

    assert n_hidden == model_meta["n_hidden_states"], (
        f"expected {model_meta['n_hidden_states']} hidden states, got {n_hidden}"
    )

    # ------------------------------------------------- last-real-token slice
    # Stack to [n_hidden, N, seq, d] then gather position (len-1) per example.
    stacked = torch.stack(hidden_states, dim=0)
    n_examples = input_ids.shape[0]
    last_idx = (lengths - 1).to(stacked.device)

    gathered = stacked[:, torch.arange(n_examples, device=stacked.device), last_idx, :]
    # -> [n_hidden, N, d]. Transpose to [N, n_layers+1, d_model].
    features = gathered.permute(1, 0, 2).contiguous()

    logger.info("-" * 72)
    logger.info("HIDDEN STATE SHAPE: %s", tuple(features.shape))
    logger.info("  expected [N, n_layers+1, d_model] = [%d, %d, %d]",
                n_examples, model_meta["n_hidden_states"], model_meta["d_model"])
    logger.info("-" * 72)

    expected_shape = (n_examples, model_meta["n_hidden_states"], model_meta["d_model"])
    assert tuple(features.shape) == expected_shape, (
        f"shape mismatch: got {tuple(features.shape)}, expected {expected_shape}"
    )

    # Cast to float32 only at save time.
    features_np = features.to(torch.float32).cpu().numpy()
    assert features_np.dtype.name == "float32", features_np.dtype

    # Non-finite values would poison every downstream probe silently.
    import numpy as np

    assert np.isfinite(features_np).all(), "non-finite values in extracted hidden states"

    labels = np.array([row["label"] for row in batch], dtype=np.int64)
    logger.info(
        "label balance: %d true / %d false (of %d)",
        int(labels.sum()), int((1 - labels).sum()), len(labels),
    )

    # ------------------------------------------------------ size projection
    bytes_per_example = features_np[0].nbytes
    for n_full in (6000,):
        projected_gb = bytes_per_example * n_full / 1024**3
        logger.info(
            "projected cache size at N=%d: %.2f GB per precision, %.2f GB for all 3",
            n_full, projected_gb, projected_gb * 3,
        )

    # ------------------------------------------------------------- sidecar
    sidecar = provenance(
        {
            "phase": "0-smoke-test",
            "config": cfg.to_plain(),
            "seeds": seed_record,
            "model": model_meta,
            "n_examples": n_examples,
            "feature_shape": list(features_np.shape),
            "saved_dtype": features_np.dtype.name,
            "hidden_state_runtime_dtype": str(hidden_states[0].dtype),
            "token_counts": lengths.tolist(),
        }
    )
    sidecar_path = write_json(
        paths.logs_dir() / f"smoke_{cfg.model.short_name}_{precision}.json", sidecar
    )
    logger.info("provenance written to %s", sidecar_path)

    logger.info("=" * 72)
    logger.info("SMOKE TEST PASSED")
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
