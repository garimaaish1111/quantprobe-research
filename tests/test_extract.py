"""Phase 2 tests - tokenization, fingerprinting, alignment, resume, verification.

All offline. A tiny stub model stands in for a real one: extraction logic is
indexing and bookkeeping, and neither needs a billion parameters to be wrong.

The two tests that matter most are
`test_cross_precision_check_rejects_differing_batch_size` and
`test_assert_token_alignment_rejects_changed_tokens`. Both encode measured
behaviour rather than theory - see CONSISTENCY_FIELDS in src/extract/hidden.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.data.schema import Record  # noqa: E402
from src.extract import hidden as H  # noqa: E402


# ------------------------------------------------------------------- stubs


class StubTokenizer:
    """Whitespace tokenizer. Deterministic, and short enough to reason about."""

    def __init__(self, drift: bool = False):
        self.padding_side = "right"
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"
        self.drift = drift  # simulates a tokenizer revision

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="np"):
        offset = 1000 if self.drift else 0
        rows = [[len(w) + offset for w in t.split()][:max_length] for t in texts]
        width = max(len(r) for r in rows)
        ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
            mask[i, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def records(n: int = 6) -> list[Record]:
    return [
        Record(
            id=f"s:t:{i}",
            prompt="",
            statement=" ".join(["word"] * (i + 2)),
            label=i % 2,
            topic="t",
            source="s",
        )
        for i in range(n)
    ]


def cfg_for(tmp_path: Path):
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    cfg["paths"]["cache_root"] = str(tmp_path)
    cfg["extract"]["max_length"] = 64
    cfg["extract"]["min_free_gb"] = 0.0
    return cfg


def write_arm(
    root: Path,
    model: str,
    precision: str,
    ids: np.ndarray,
    lengths: np.ndarray,
    batch_size: int = 16,
    n_examples: int | None = None,
) -> Path:
    """Fabricate a finished extraction directory."""
    d = H.extraction_dir(root, model, precision)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / H.TOKENS_FILE, ids)
    np.save(d / H.LENGTHS_FILE, lengths)
    (d / H.META_FILE).write_text(
        json.dumps(
            {
                "batch_size": batch_size,
                "token_position": "last",
                "n_examples": n_examples if n_examples is not None else len(lengths),
            }
        ),
        encoding="utf-8",
    )
    return d


# -------------------------------------------------------------------- paths


def test_extraction_dir_layout(tmp_path):
    d = H.extraction_dir(tmp_path, "llama32-1b", "int4")
    assert d == tmp_path / "activations" / "llama32-1b" / "int4"


def test_extraction_dir_rejects_bad_precision(tmp_path):
    with pytest.raises(H.ExtractionError, match="unknown precision"):
        H.extraction_dir(tmp_path, "m", "int2")


def test_is_complete_false_when_absent(tmp_path):
    assert not H.is_complete(tmp_path / "nope", 10)


def test_is_complete_tracks_the_flag_and_the_count(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    np.lib.format.open_memmap(d / H.HIDDEN_FILE, mode="w+", dtype=np.float32, shape=(4, 2, 3))
    H._write_progress(d, n_done=4, complete=False)
    assert not H.is_complete(d, 4), "unfinished run must not count as complete"
    H._write_progress(d, n_done=4, complete=True)
    assert H.is_complete(d, 4)
    assert not H.is_complete(d, 5), "row count must match what was asked for"


# --------------------------------------------------------------- tokenizing


def test_tokenize_shapes_and_lengths():
    out = H.tokenize_records(StubTokenizer(), records(4), max_length=64)
    assert out["input_ids"].shape[0] == 4
    assert out["lengths"].tolist() == [2, 3, 4, 5]
    assert out["input_ids"].dtype == np.int64


def test_tokenize_rejects_truncation():
    """A truncated row's 'last token' is mid-claim - that must not pass silently."""
    with pytest.raises(H.ExtractionError, match="max_length"):
        H.tokenize_records(StubTokenizer(), records(6), max_length=3)


def test_tokenize_forces_right_padding():
    tok = StubTokenizer()
    tok.padding_side = "left"
    H.tokenize_records(tok, records(3), max_length=64)
    assert tok.padding_side == "right", "left padding shifts position ids under RoPE"


def test_padding_is_after_the_real_tokens():
    out = H.tokenize_records(StubTokenizer(), records(3), max_length=64)
    ids, lengths = out["input_ids"], out["lengths"]
    for i, length in enumerate(lengths):
        assert (ids[i, length:] == 0).all(), "pad must sit to the right of real tokens"
        assert (ids[i, :length] != 0).all()


# ------------------------------------------------------------- fingerprints


def test_fingerprint_is_stable():
    out = H.tokenize_records(StubTokenizer(), records(5), max_length=64)
    again = H.tokenize_records(StubTokenizer(), records(5), max_length=64)
    assert out["fingerprint"] == again["fingerprint"]


def test_fingerprint_ignores_trailing_padding_width():
    """Two runs differing only in pad width saw the same tokens."""
    ids = np.array([[7, 8, 0, 0], [9, 0, 0, 0]], dtype=np.int64)
    wide = np.array([[7, 8, 0, 0, 0, 0], [9, 0, 0, 0, 0, 0]], dtype=np.int64)
    lengths = np.array([2, 1], dtype=np.int64)
    assert H.token_fingerprint(ids, lengths) == H.token_fingerprint(wide, lengths)


def test_fingerprint_changes_when_a_real_token_changes():
    ids = np.array([[7, 8, 0]], dtype=np.int64)
    other = np.array([[7, 9, 0]], dtype=np.int64)
    lengths = np.array([2], dtype=np.int64)
    assert H.token_fingerprint(ids, lengths) != H.token_fingerprint(other, lengths)


def test_fingerprint_changes_when_lengths_change():
    ids = np.array([[7, 8, 9]], dtype=np.int64)
    assert H.token_fingerprint(ids, np.array([2])) != H.token_fingerprint(ids, np.array([3]))


# ---------------------------------------------------------------- alignment


def test_assert_token_alignment_passes_on_first_run(tmp_path):
    out = H.tokenize_records(StubTokenizer(), records(4), max_length=64)
    H.assert_token_alignment(tmp_path, out["fingerprint"], out["input_ids"], out["lengths"])


def test_assert_token_alignment_accepts_identical_tokens(tmp_path):
    out = H.tokenize_records(StubTokenizer(), records(4), max_length=64)
    np.save(tmp_path / H.TOKENS_FILE, out["input_ids"])
    np.save(tmp_path / H.LENGTHS_FILE, out["lengths"])
    H.assert_token_alignment(tmp_path, out["fingerprint"], out["input_ids"], out["lengths"])


def test_assert_token_alignment_rejects_changed_tokens(tmp_path):
    """Simulates a tokenizer revision between the FP16 and INT4 runs."""
    first = H.tokenize_records(StubTokenizer(), records(4), max_length=64)
    np.save(tmp_path / H.TOKENS_FILE, first["input_ids"])
    np.save(tmp_path / H.LENGTHS_FILE, first["lengths"])

    drifted = H.tokenize_records(StubTokenizer(drift=True), records(4), max_length=64)
    with pytest.raises(H.ExtractionError, match="token IDs differ"):
        H.assert_token_alignment(
            tmp_path, drifted["fingerprint"], drifted["input_ids"], drifted["lengths"]
        )


def test_assert_token_alignment_warns_but_proceeds_on_a_row_count_change(caplog):
    """A different N is a change of scope, not tokenizer drift.

    Superseded an earlier test that required this to raise. It raised on the
    smoke-run-then-full-run sequence, i.e. following the smoke-first rule
    broke the run that rule exists to protect. It must warn loudly and continue;
    a genuinely mismatched arm is still caught by the cross-precision check,
    which compares n_examples across all arms before any probing happens.
    """
    import logging
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    first = H.tokenize_records(StubTokenizer(), records(4), max_length=64)
    np.save(tmp / H.TOKENS_FILE, first["input_ids"])
    np.save(tmp / H.LENGTHS_FILE, first["lengths"])

    fewer = H.tokenize_records(StubTokenizer(), records(3), max_length=64)
    with caplog.at_level(logging.WARNING, logger="quantprobe.extract.hidden"):
        H.assert_token_alignment(
            tmp, fewer["fingerprint"], fewer["input_ids"], fewer["lengths"]
        )
    assert "row count changed" in caplog.text


# ------------------------------------------------ cross-precision done-signal


def test_cross_precision_check_passes_when_arms_agree(tmp_path):
    ids = np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64)
    lengths = np.array([2, 1], dtype=np.int64)
    for p in ("fp16", "int8", "int4"):
        write_arm(tmp_path, "m", p, ids, lengths, batch_size=16)
    result = H.cross_precision_token_check(tmp_path, "m", ["fp16", "int8", "int4"])
    assert len(result["shared_fingerprint"]) == 64
    assert result["shared_settings"]["batch_size"] == 16


def test_cross_precision_check_rejects_differing_tokens(tmp_path):
    lengths = np.array([2, 1], dtype=np.int64)
    write_arm(tmp_path, "m", "fp16", np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64), lengths)
    write_arm(tmp_path, "m", "int4", np.array([[1, 9, 0], [3, 0, 0]], dtype=np.int64), lengths)
    with pytest.raises(H.ExtractionError, match="fingerprints differ"):
        H.cross_precision_token_check(tmp_path, "m", ["fp16", "int4"])


def test_cross_precision_check_rejects_differing_batch_size(tmp_path):
    """Measured, not theoretical: batch size changes padding width, which
    changes matmul reduction order, which moves activations by ~3e-4."""
    ids = np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int64)
    lengths = np.array([2, 1], dtype=np.int64)
    write_arm(tmp_path, "m", "fp16", ids, lengths, batch_size=16)
    write_arm(tmp_path, "m", "int4", ids, lengths, batch_size=8)
    with pytest.raises(H.ExtractionError, match="batch_size"):
        H.cross_precision_token_check(tmp_path, "m", ["fp16", "int4"])


def test_cross_precision_check_reports_a_missing_arm(tmp_path):
    ids = np.array([[1, 2, 0]], dtype=np.int64)
    write_arm(tmp_path, "m", "fp16", ids, np.array([2], dtype=np.int64))
    with pytest.raises(H.ExtractionError, match="no extraction found"):
        H.cross_precision_token_check(tmp_path, "m", ["fp16", "int4"])


def test_cross_precision_check_requires_a_sidecar(tmp_path):
    ids = np.array([[1, 2, 0]], dtype=np.int64)
    d = write_arm(tmp_path, "m", "fp16", ids, np.array([2], dtype=np.int64))
    (d / H.META_FILE).unlink()
    with pytest.raises(H.ExtractionError, match="sidecar"):
        H.cross_precision_token_check(tmp_path, "m", ["fp16"])


# ------------------------------------------------------- output verification


def test_verify_output_accepts_a_good_array(tmp_path):
    arr = np.lib.format.open_memmap(
        tmp_path / H.HIDDEN_FILE, mode="w+", dtype=np.float32, shape=(8, 3, 4)
    )
    arr[:] = np.random.default_rng(0).normal(size=(8, 3, 4)).astype(np.float32)
    arr.flush()
    del arr
    H._verify_output(tmp_path, (8, 3, 4))


def test_verify_output_rejects_all_zero_rows(tmp_path):
    """The signature of a resume that skipped rows."""
    arr = np.lib.format.open_memmap(
        tmp_path / H.HIDDEN_FILE, mode="w+", dtype=np.float32, shape=(8, 3, 4)
    )
    arr[:] = 1.0
    arr[3] = 0.0
    arr.flush()
    del arr
    with pytest.raises(H.ExtractionError, match="entirely zero"):
        H._verify_output(tmp_path, (8, 3, 4))


def test_verify_output_rejects_non_finite(tmp_path):
    arr = np.lib.format.open_memmap(
        tmp_path / H.HIDDEN_FILE, mode="w+", dtype=np.float32, shape=(8, 3, 4)
    )
    arr[:] = 1.0
    arr[2, 1, 1] = np.nan
    arr.flush()
    del arr
    with pytest.raises(H.ExtractionError, match="non-finite"):
        H._verify_output(tmp_path, (8, 3, 4))


def test_verify_output_rejects_wrong_shape(tmp_path):
    arr = np.lib.format.open_memmap(
        tmp_path / H.HIDDEN_FILE, mode="w+", dtype=np.float32, shape=(8, 3, 4)
    )
    arr[:] = 1.0
    arr.flush()
    del arr
    with pytest.raises(H.ExtractionError, match="shape"):
        H._verify_output(tmp_path, (8, 3, 5))


# -------------------------------------------------------------- load_extraction


def test_load_extraction_refuses_an_incomplete_run(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    np.lib.format.open_memmap(d / H.HIDDEN_FILE, mode="w+", dtype=np.float32, shape=(2, 2, 2))
    H._write_progress(d, n_done=1, complete=False)
    with pytest.raises(H.ExtractionError, match="INCOMPLETE"):
        H.load_extraction(d)


def test_load_extraction_reports_a_missing_run(tmp_path):
    with pytest.raises(H.ExtractionError, match="no extraction"):
        H.load_extraction(tmp_path / "nothing")


# ------------------------------------------------- quantization actually happened


def test_quantization_assertion_rejects_an_unquantized_int_arm():
    """The most dangerous silent failure: bitsandbytes no-ops and returns a
    perfectly working FP16 model wearing an INT label."""
    from src.models.loader import ModelLoadError, _assert_quantization_happened

    fp16_only = {"param_dtype_counts": {"torch.float16": 200}}
    for precision in ("int8", "int4"):
        with pytest.raises(ModelLoadError, match="NO uint8 parameters"):
            _assert_quantization_happened(fp16_only, precision)


def test_quantization_assertion_accepts_a_real_quantized_model():
    from src.models.loader import _assert_quantization_happened

    quantized = {"param_dtype_counts": {"torch.uint8": 154, "torch.float16": 46}}
    _assert_quantization_happened(quantized, "int4")


def test_quantization_assertion_ignores_the_fp16_arm():
    """FP16 has no uint8 params and must not be flagged for it."""
    from src.models.loader import _assert_quantization_happened

    _assert_quantization_happened({"param_dtype_counts": {"torch.float16": 200}}, "fp16")


# ------------------------------------------------------- token hygiene


def test_hf_token_strips_whitespace(monkeypatch):
    """A token pasted with a trailing newline is truthy, sails past the
    'is a token present' guard, and dies 400 lines later as an opaque 401."""
    from src.models.loader import hf_token

    monkeypatch.setenv("HF_TOKEN", "  hf_abc123\n")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_token() == "hf_abc123"


def test_hf_token_treats_blank_as_absent(monkeypatch):
    from src.models.loader import hf_token

    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    for blank in ("", "   ", "\n", "\t "):
        monkeypatch.setenv("HF_TOKEN", blank)
        assert hf_token() is None, f"{blank!r} should count as no token"


def test_hf_token_falls_back_to_the_hub_standard_var(monkeypatch):
    from src.models.loader import hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_fallback")
    assert hf_token() == "hf_fallback"


def test_preflight_turns_a_401_into_an_actionable_message(monkeypatch):
    """Account access and token access are different things, and the raw
    HF error points at the licence page you already completed."""
    import src.models.loader as L

    class FakeGatedRepoError(Exception):
        pass
    FakeGatedRepoError.__name__ = "GatedRepoError"

    class FakeApi:
        def model_info(self, *a, **k):
            raise FakeGatedRepoError("401 Client Error. Cannot access gated repo")

    monkeypatch.setattr(L, "HfApi", FakeApi, raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    with pytest.raises(L.ModelLoadError) as exc:
        L.preflight_gated_access("meta-llama/Llama-3.2-1B-Instruct", "hf_xxx")
    msg = str(exc.value)
    assert "FINE-GRAINED" in msg
    assert "settings/gated-repos" in msg
    assert "not a missing-token problem" in msg


def test_preflight_ignores_unrelated_failures(monkeypatch):
    """Offline, DNS, rate limit - not ours to diagnose, must not block."""
    import src.models.loader as L
    import huggingface_hub

    class FakeApi:
        def model_info(self, *a, **k):
            raise ConnectionError("no route to host")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    L.preflight_gated_access("meta-llama/Llama-3.2-1B-Instruct", "hf_xxx")


def test_int8_storage_dtype_counts_as_quantized():
    """Regression: LLM.int8() stores torch.int8, NOT torch.uint8.

    These are the exact dtype counts observed from a real Llama-3.2-1B INT8
    load on a Colab T4. An earlier version of this check looked only for
    "uint8" and rejected a perfectly good INT8 arm.
    """
    from src.models.loader import _assert_quantization_happened

    observed_int8 = {"param_dtype_counts": {"torch.bfloat16": 34, "torch.int8": 112}}
    _assert_quantization_happened(observed_int8, "int8")


def test_nf4_storage_dtype_counts_as_quantized():
    """NF4 packs two 4-bit values per byte, so it lands in torch.uint8."""
    from src.models.loader import _assert_quantization_happened

    _assert_quantization_happened(
        {"param_dtype_counts": {"torch.float16": 34, "torch.uint8": 112}}, "int4"
    )


def test_quantization_check_matches_dtype_names_not_substrings():
    """'uint8' contains 'int8'; the check must compare names, not substrings.

    A naive `"int8" in dtype` would also accept float dtypes with unlucky
    names, and a naive `"uint8" in dtype` rejects real int8. Both directions
    are guarded here.
    """
    from src.models.loader import QUANTIZED_STORAGE_DTYPES, ModelLoadError, _assert_quantization_happened

    assert QUANTIZED_STORAGE_DTYPES == {"int8", "uint8"}
    for unquantized in ({"torch.bfloat16": 146}, {"torch.float16": 146}, {"torch.float32": 146}):
        with pytest.raises(ModelLoadError, match="NO uint8"):
            _assert_quantization_happened({"param_dtype_counts": unquantized}, "int4")


def test_alignment_allows_a_smoke_run_to_be_followed_by_the_full_run(tmp_path):
    """The mandatory N=10 smoke test must not poison the real run.

    Both write to the same directory. An earlier version raised "token count
    changed" here, which meant following the smoke-first rule broke the run it protects.
    """
    np.save(tmp_path / H.TOKENS_FILE, np.arange(50, dtype=np.int64).reshape(10, 5))
    np.save(tmp_path / H.LENGTHS_FILE, np.full(10, 5, dtype=np.int64))

    ids = np.arange(6049 * 5, dtype=np.int64).reshape(6049, 5)
    lens = np.full(6049, 5, dtype=np.int64)
    H.assert_token_alignment(tmp_path, H.token_fingerprint(ids, lens), ids, lens)


def test_alignment_still_rejects_drift_at_the_same_row_count(tmp_path):
    """Same N, different tokens, is real drift and must still fail."""
    ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    lens = np.array([3, 3], dtype=np.int64)
    np.save(tmp_path / H.TOKENS_FILE, ids)
    np.save(tmp_path / H.LENGTHS_FILE, lens)

    drifted = np.array([[1, 2, 3], [4, 5, 99]], dtype=np.int64)
    with pytest.raises(H.ExtractionError, match="token IDs differ"):
        H.assert_token_alignment(
            tmp_path, H.token_fingerprint(drifted, lens), drifted, lens
        )
