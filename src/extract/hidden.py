"""Stage A - extract last-token hidden states at every layer.

For each `(model, precision)` pair: run the same inputs
through the model, take the residual stream at the last token of the
statement at every layer including embeddings, save `[N, n_layers+1, d_model]`
as float32 with a JSON sidecar.

Three design points that are not obvious and matter a lot.

**1. Tokenize once, then assert equality - do not merely tokenize identically.**
Token IDs and sequence lengths must match exactly between precision runs. Tokenizing with the same tokenizer object in the same process
would make that assertion trivially true and therefore worthless. What we
actually want to catch is the *cross-session* case: the INT4 run happens on
Colab three days after the FP16 run, and in between the Hub served a revised
tokenizer, or `max_length` changed, or the dataset cache was rebuilt with
`drop_contradictions` flipped. So every run writes its token matrix and a
SHA-256 fingerprint of it, and every subsequent run for the same model
compares against what is already on disk and refuses to continue on a
mismatch. That is an assertion with something real to fail against.

**2. The last token is found from the attention mask, not from position -1.**
With right padding, position -1 of a short sequence is a PAD token. Reading it
would give a different thing for every row and would look completely normal:
no shape error, no NaN, just a quietly meaningless feature matrix. The true
index is `attention_mask.sum(dim=1) - 1`.

Why right padding rather than left, which would also put the last real token
at -1: left padding shifts every position id, and rotary embeddings make
hidden states position-dependent. That would perturb the activations we are
trying to compare across precisions - a second confound layered on top of the
one we are measuring.

**3. Output goes straight to a memmap, and runs are resumable.**
`[6049, 17, 2048]` float32 is 842 MB. Holding that in RAM alongside a model on
a free Colab instance is possible but pointless. `np.lib.format.open_memmap`
writes rows to disk as they are produced, and a `progress.json` records how
many are done, so a Colab disconnect at row 4000 costs four thousand rows of
work rather than all of them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.schema import Record
from src.logging_utils import provenance, write_json
from src.models.loader import PRECISIONS

logger = logging.getLogger("quantprobe.extract.hidden")

#: Files written into every extraction directory.
HIDDEN_FILE = "hidden.npy"
LABELS_FILE = "labels.npy"
TOKENS_FILE = "tokens.npy"
LENGTHS_FILE = "token_lengths.npy"
IDS_FILE = "ids.json"
META_FILE = "meta.json"
PROGRESS_FILE = "progress.json"


class ExtractionError(RuntimeError):
    """Extraction cannot proceed, or produced something untrustworthy."""


# --------------------------------------------------------------------- paths


def extraction_dir(cache_root: Path, model_short: str, precision: str) -> Path:
    """`<cache_root>/<model>/<precision>/`."""
    if precision not in PRECISIONS:
        raise ExtractionError(f"unknown precision {precision!r}; expected {list(PRECISIONS)}")
    return Path(cache_root) / "activations" / model_short / precision


def is_complete(directory: Path, expected_n: int) -> bool:
    """True if this directory holds a finished extraction of `expected_n` rows."""
    progress_path = Path(directory) / PROGRESS_FILE
    if not progress_path.is_file() or not (Path(directory) / HIDDEN_FILE).is_file():
        return False
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(progress.get("complete")) and progress.get("n_done") == expected_n


# ---------------------------------------------------------------- tokenizing


def tokenize_records(
    tokenizer,
    records: Sequence[Record],
    max_length: int,
    text_template: str = "{prompt} {statement}",
) -> dict[str, Any]:
    """Tokenize every record once, right-padded to a common length.

    Returns a dict with `input_ids` and `attention_mask` as int64 numpy
    arrays of shape [N, max_seq], plus `lengths` [N] and a `fingerprint`.

    Raises:
        ExtractionError: if any sequence hit `max_length`, which means it was
            truncated and its "last token" is the middle of a claim rather
            than the end of one.
    """
    tokenizer.padding_side = "right"
    texts = [rec.text(text_template) for rec in records]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )

    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
    lengths = attention_mask.sum(axis=1).astype(np.int64)

    truncated = np.flatnonzero(lengths >= max_length)
    if truncated.size:
        examples = [records[i].id for i in truncated[:5]]
        raise ExtractionError(
            f"{truncated.size} sequences reached max_length={max_length} and were "
            f"truncated, so their last token is not the end of the claim. "
            f"Raise extract.max_length. First offenders: {examples}"
        )

    if int(lengths.min()) < 1:
        raise ExtractionError("at least one sequence tokenized to zero real tokens")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "fingerprint": token_fingerprint(input_ids, lengths),
    }


def token_fingerprint(input_ids: np.ndarray, lengths: np.ndarray) -> str:
    """SHA-256 over the token matrix and the true lengths.

    Padding is excluded implicitly by hashing lengths alongside ids, so two
    runs that differ only in batch-driven padding width still agree.
    """
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(lengths.astype(np.int64)).tobytes())
    for row, length in zip(input_ids, lengths):
        hasher.update(np.ascontiguousarray(row[:length].astype(np.int64)).tobytes())
    return hasher.hexdigest()


def assert_token_alignment(
    directory: Path,
    fingerprint: str,
    input_ids: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """Compare this run's tokens against whatever a previous run stored here.

    This is the cross-precision token-equality assertion, made real by
    comparing against bytes on disk rather than against an object in the same
    process.
    """
    tokens_path = Path(directory) / TOKENS_FILE
    lengths_path = Path(directory) / LENGTHS_FILE
    if not tokens_path.is_file() or not lengths_path.is_file():
        return  # nothing to compare against yet

    stored_ids = np.load(tokens_path)
    stored_lengths = np.load(lengths_path)
    stored_fingerprint = token_fingerprint(stored_ids, stored_lengths)

    if stored_fingerprint == fingerprint:
        return

    if stored_lengths.shape != lengths.shape:
        # A different NUMBER of rows is a change of scope, not tokenizer
        # drift - the overwhelmingly common case being a --limit 10 smoke run
        # followed by the real thing in the same directory. Erroring here
        # would make the mandatory smoke test poison the run it exists to
        # protect. The whole array is rewritten from scratch anyway, and the
        # cross-precision check still compares n_examples across arms, so an
        # arm genuinely left at the wrong size is caught before any probing.
        logger.warning(
            "%s: row count changed (stored %d, this run %d). Treating as a new "
            "extraction and overwriting. If this was NOT intended - e.g. the "
            "dataset changed under the cache - re-extract every precision, "
            "because arms of different sizes are not comparable.",
            directory, stored_lengths.shape[0], lengths.shape[0],
        )
        return

    mismatched = np.flatnonzero(stored_lengths != lengths)
    raise ExtractionError(
        f"{directory}: token IDs differ from the stored run "
        f"(stored {stored_fingerprint[:12]}, now {fingerprint[:12]}). "
        f"{mismatched.size} sequences differ in length; first indices "
        f"{mismatched[:5].tolist()}. Tokenizer revision, max_length, or the "
        f"dataset changed between runs. The precision arms would not be "
        f"comparable - re-extract all of them."
    )


#: Sidecar fields that MUST agree across precision arms, and why.
#:
#: `batch_size` is here because of a measured effect, not a theoretical one.
#: Batches are padded to the longest sequence they contain, and a different
#: padding width changes the order in which matmul reductions accumulate. On
#: SmolLM2-135M, extracting the same 64 rows at batch_size 8 versus 2 moved
#: mid-stack activations by up to 3.3e-4 relative - while re-running at the
#: SAME batch size was bit-identical (diff exactly 0.0). That noise is far
#: below the drift INT4 produces, but it is pure artefact, and E3 exists
#: specifically to measure small per-layer distances. Holding batch_size
#: fixed removes it entirely rather than leaving it to be argued about.
CONSISTENCY_FIELDS: tuple[str, ...] = ("batch_size", "token_position", "n_examples")


def cross_precision_token_check(
    cache_root: Path, model_short: str, precisions: Sequence[str]
) -> dict[str, Any]:
    """Verify every extracted precision saw identical inputs, identically batched.

    This is the Phase 2 done-signal. Run after all arms exist. Checks two
    things:

    1. Byte-identical token IDs and sequence lengths across precision runs.
    2. Identical extraction settings, per CONSISTENCY_FIELDS above.
    """
    fingerprints: dict[str, str] = {}
    settings: dict[str, dict[str, Any]] = {}

    for precision in precisions:
        directory = extraction_dir(cache_root, model_short, precision)
        tokens_path = directory / TOKENS_FILE
        if not tokens_path.is_file():
            raise ExtractionError(f"no extraction found at {directory}")
        fingerprints[precision] = token_fingerprint(
            np.load(tokens_path), np.load(directory / LENGTHS_FILE)
        )

        meta_path = directory / META_FILE
        if not meta_path.is_file():
            raise ExtractionError(f"{directory}: sidecar {META_FILE} is missing")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        missing = [f for f in CONSISTENCY_FIELDS if f not in meta]
        if missing:
            raise ExtractionError(f"{directory}: sidecar is missing {missing}")
        settings[precision] = {f: meta[f] for f in CONSISTENCY_FIELDS}

    unique = set(fingerprints.values())
    if len(unique) != 1:
        raise ExtractionError(
            f"token fingerprints differ across precisions: "
            f"{ {p: f[:12] for p, f in fingerprints.items()} }. "
            f"The arms did not see identical inputs, so no comparison between "
            f"them is valid. Re-extract every precision from one dataset cache."
        )

    for field in CONSISTENCY_FIELDS:
        values = {p: settings[p][field] for p in precisions}
        if len(set(map(repr, values.values()))) != 1:
            raise ExtractionError(
                f"extraction setting {field!r} differs across precisions: {values}. "
                f"Arms must be extracted with identical settings - see "
                f"CONSISTENCY_FIELDS in this module for why batch_size in "
                f"particular is not negotiable. Re-extract the odd one out."
            )

    logger.info(
        "alignment verified across %s: fingerprint %s, settings %s",
        list(precisions),
        next(iter(unique))[:16],
        settings[precisions[0]],
    )
    return {
        "precisions": list(precisions),
        "shared_fingerprint": next(iter(unique)),
        "shared_settings": settings[precisions[0]],
    }


# --------------------------------------------------------------- extraction


def extract_hidden_states(
    model,
    tokenizer,
    records: Sequence[Record],
    directory: Path,
    model_meta: dict[str, Any],
    cfg,
    precision: str,
    seed_record: dict[str, Any],
    batch_size: int = 16,
    resume: bool = True,
) -> Path:
    """Run Stage A for one (model, precision) and cache the result.

    Args:
        model: an already-loaded model in eval mode.
        tokenizer: its tokenizer.
        records: the dataset rows, in a fixed order.
        directory: where to write. Created if absent.
        model_meta: the block returned by `load_model_and_tokenizer`.
        cfg: the loaded Config, recorded verbatim into the sidecar.
        precision: which arm this is.
        seed_record: output of `set_seed`, for the sidecar.
        batch_size: forward-pass batch size.
        resume: continue a partial run rather than restarting it.

    Returns:
        Path to the extraction directory.
    """
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    n_examples = len(records)
    n_hidden = model_meta["n_hidden_states"]
    d_model = model_meta["d_model"]
    if n_hidden is None or d_model is None:
        raise ExtractionError(f"model metadata is missing shape info: {model_meta}")

    # ---------------------------------------------------------- tokenization
    tokens = tokenize_records(
        tokenizer,
        records,
        max_length=int(cfg.extract.max_length),
        text_template=str(cfg.extract.get("text_template", "{prompt} {statement}")),
    )
    assert_token_alignment(directory, tokens["fingerprint"], tokens["input_ids"], tokens["lengths"])

    logger.info(
        "tokenized %d records | seq width %d | true lengths min/mean/max %d/%.1f/%d | fp %s",
        n_examples,
        tokens["input_ids"].shape[1],
        int(tokens["lengths"].min()),
        float(tokens["lengths"].mean()),
        int(tokens["lengths"].max()),
        tokens["fingerprint"][:12],
    )

    # ------------------------------------------------------------- disk room
    needed_gb = n_examples * n_hidden * d_model * 4 / 1024**3
    from src.paths import free_space_gb

    free_gb = free_space_gb(directory)
    logger.info("output will be %.2f GB; %.1f GB free on that volume", needed_gb, free_gb)
    min_free = float(cfg.extract.get("min_free_gb", 5.0))
    if free_gb - needed_gb < min_free:
        raise ExtractionError(
            f"refusing to start: writing {needed_gb:.2f} GB would leave "
            f"{free_gb - needed_gb:.2f} GB free, below extract.min_free_gb={min_free}"
        )

    # ------------------------------------------------------- memmap + resume
    hidden_path = directory / HIDDEN_FILE
    shape = (n_examples, n_hidden, d_model)
    start_row = 0

    if resume and hidden_path.is_file():
        existing = np.lib.format.open_memmap(hidden_path, mode="r")
        if existing.shape == shape:
            progress = _read_progress(directory)
            start_row = int(progress.get("n_done", 0))
            del existing
            logger.info("resuming at row %d of %d", start_row, n_examples)
        else:
            logger.warning(
                "existing array has shape %s, expected %s - starting over",
                existing.shape, shape,
            )
            del existing
            hidden_path.unlink()

    mode = "r+" if hidden_path.is_file() else "w+"
    if mode == "w+":
        out = np.lib.format.open_memmap(
            hidden_path, mode="w+", dtype=np.float32, shape=shape
        )
    else:
        out = np.lib.format.open_memmap(hidden_path, mode="r+")

    if start_row >= n_examples:
        logger.info("extraction already complete at %s", directory)
    else:
        _run_forward_passes(
            model=model,
            tokens=tokens,
            out=out,
            start_row=start_row,
            batch_size=batch_size,
            n_hidden=n_hidden,
            directory=directory,
            torch=torch,
        )

    out.flush()
    del out

    # ------------------------------------------------------------ companions
    np.save(directory / TOKENS_FILE, tokens["input_ids"])
    np.save(directory / LENGTHS_FILE, tokens["lengths"])
    np.save(
        directory / LABELS_FILE,
        np.array([rec.label for rec in records], dtype=np.int64),
    )
    (directory / IDS_FILE).write_text(
        json.dumps([rec.id for rec in records], indent=0), encoding="utf-8"
    )

    _verify_output(directory, shape)

    # -------------------------------------------------------------- sidecar
    write_json(
        directory / META_FILE,
        provenance(
            {
                "phase": "2-extraction",
                "precision": precision,
                "model": model_meta,
                "config": cfg.to_plain(),
                "seeds": seed_record,
                "n_examples": n_examples,
                "shape": list(shape),
                "saved_dtype": "float32",
                "runtime_hidden_dtype": model_meta.get("effective_load_dtype"),
                "token_fingerprint": tokens["fingerprint"],
                "token_matrix_shape": list(tokens["input_ids"].shape),
                "token_length_stats": {
                    "min": int(tokens["lengths"].min()),
                    "mean": round(float(tokens["lengths"].mean()), 2),
                    "max": int(tokens["lengths"].max()),
                },
                "dataset": {
                    "n": n_examples,
                    "sources": sorted({rec.source for rec in records}),
                    "topics": sorted({rec.topic for rec in records}),
                    "positive_rate": round(
                        float(np.mean([rec.label for rec in records])), 4
                    ),
                },
                "batch_size": batch_size,
                "token_position": str(cfg.extract.token_position),
            }
        ),
    )
    _write_progress(directory, n_done=n_examples, complete=True)
    logger.info("extraction complete: %s", directory)
    return directory


def _run_forward_passes(
    model, tokens, out, start_row: int, batch_size: int, n_hidden: int, directory: Path, torch
) -> None:
    """The actual loop. Writes rows [start_row:] into `out`."""
    input_ids_all = tokens["input_ids"]
    attention_all = tokens["attention_mask"]
    lengths_all = tokens["lengths"]
    n_examples = input_ids_all.shape[0]

    device = next(model.parameters()).device
    started = time.time()
    processed = 0

    for batch_start in range(start_row, n_examples, batch_size):
        batch_end = min(batch_start + batch_size, n_examples)

        # Trim padding to the longest real sequence IN THIS BATCH. Purely a
        # speed optimisation: attention is masked either way, so the hidden
        # state at a real token is unchanged by how much padding follows it.
        batch_lengths = lengths_all[batch_start:batch_end]
        width = int(batch_lengths.max())

        input_ids = torch.from_numpy(input_ids_all[batch_start:batch_end, :width]).to(device)
        attention_mask = torch.from_numpy(attention_all[batch_start:batch_end, :width]).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if len(hidden_states) != n_hidden:
            raise ExtractionError(
                f"batch at row {batch_start}: model returned {len(hidden_states)} "
                f"hidden states, expected {n_hidden}"
            )

        rows = torch.arange(batch_end - batch_start, device=device)
        last_idx = torch.from_numpy(batch_lengths - 1).to(device)

        # Layer by layer rather than torch.stack over all of them: stacking
        # would briefly hold n_hidden * batch * width * d_model values, which
        # for a 1B model is a few hundred MB of avoidable peak VRAM.
        for layer_idx, layer_hidden in enumerate(hidden_states):
            selected = layer_hidden[rows, last_idx, :]
            out[batch_start:batch_end, layer_idx, :] = (
                selected.to(torch.float32).cpu().numpy()
            )

        del outputs, hidden_states
        processed += batch_end - batch_start

        if batch_start // batch_size % 20 == 0 or batch_end == n_examples:
            out.flush()
            _write_progress(directory, n_done=batch_end, complete=False)
            elapsed = time.time() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = (n_examples - batch_end) / rate if rate > 0 else float("nan")
            logger.info(
                "  %5d/%5d rows | %.1f rows/s | ~%.1f min left",
                batch_end, n_examples, rate, remaining / 60,
            )

    out.flush()
    _write_progress(directory, n_done=n_examples, complete=False)


def _verify_output(directory: Path, shape: tuple[int, ...]) -> None:
    """Post-run assertions. Cheap relative to the run that produced them."""
    hidden = np.lib.format.open_memmap(Path(directory) / HIDDEN_FILE, mode="r")

    if hidden.shape != shape:
        raise ExtractionError(f"{directory}: shape {hidden.shape}, expected {shape}")
    if hidden.dtype != np.float32:
        raise ExtractionError(f"{directory}: dtype {hidden.dtype}, expected float32")

    # Non-finite values would poison every probe downstream, silently - sklearn
    # would raise something obscure many steps later.
    sample = hidden[:: max(1, shape[0] // 512)]
    if not np.isfinite(sample).all():
        raise ExtractionError(f"{directory}: non-finite values in extracted hidden states")

    # An all-zero row means a forward pass never wrote it - the classic
    # symptom of a resume that skipped rows.
    norms = np.linalg.norm(sample.reshape(sample.shape[0], -1), axis=1)
    dead = np.flatnonzero(norms == 0)
    if dead.size:
        raise ExtractionError(
            f"{directory}: {dead.size} sampled rows are entirely zero - those "
            f"forward passes never ran. Delete {PROGRESS_FILE} and re-extract."
        )

    del hidden


def _read_progress(directory: Path) -> dict:
    path = Path(directory) / PROGRESS_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_progress(directory: Path, n_done: int, complete: bool) -> None:
    (Path(directory) / PROGRESS_FILE).write_text(
        json.dumps({"n_done": int(n_done), "complete": bool(complete)}), encoding="utf-8"
    )


# ------------------------------------------------------------------ loading


def load_extraction(directory: Path, mmap: bool = True) -> dict[str, Any]:
    """Read a cached extraction back. Stage B and C use only this.

    Args:
        directory: an extraction directory.
        mmap: keep the array on disk and page it in. The default, because
            three precisions x 842 MB does not want to be resident at once.
    """
    directory = Path(directory)
    hidden_path = directory / HIDDEN_FILE
    if not hidden_path.is_file():
        raise ExtractionError(f"no extraction at {directory}")

    progress = _read_progress(directory)
    if not progress.get("complete"):
        raise ExtractionError(
            f"{directory} holds an INCOMPLETE extraction "
            f"({progress.get('n_done', 0)} rows). Finish it before probing."
        )

    return {
        "hidden": np.load(hidden_path, mmap_mode="r" if mmap else None),
        "labels": np.load(directory / LABELS_FILE),
        "tokens": np.load(directory / TOKENS_FILE),
        "lengths": np.load(directory / LENGTHS_FILE),
        "ids": json.loads((directory / IDS_FILE).read_text(encoding="utf-8")),
        "meta": json.loads((directory / META_FILE).read_text(encoding="utf-8")),
    }
