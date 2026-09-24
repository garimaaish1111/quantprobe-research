"""Phase 0 tests: the scaffold utilities, not the science.

These run in under a second and need no model, no GPU, and no network. Their
job is to catch the failure modes that would otherwise show up forty minutes
into an extraction run: a config that silently loses a key, a seed that does
not actually seed, an env-var path that resolves to the wrong drive.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import paths  # noqa: E402
from src.config import Config, ConfigError, load_config, require  # noqa: E402
from src.seeding import set_seed  # noqa: E402


# --------------------------------------------------------------------- config


def write_yaml(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_reads_plain_yaml(tmp_path):
    path = write_yaml(tmp_path, "a.yaml", "seed: 7\nmodel:\n  hf_id: foo/bar\n")
    cfg = load_config(path)
    assert cfg.seed == 7
    assert cfg.model.hf_id == "foo/bar"


def test_attribute_access_raises_with_helpful_message(tmp_path):
    path = write_yaml(tmp_path, "a.yaml", "seed: 7\n")
    cfg = load_config(path)
    with pytest.raises(AttributeError, match="available keys"):
        _ = cfg.nonexistent


def test_extends_merges_deeply_and_child_wins(tmp_path):
    write_yaml(
        tmp_path,
        "base.yaml",
        "seed: 1729\nextract:\n  batch_size: 16\n  max_length: 128\n",
    )
    child = write_yaml(
        tmp_path,
        "child.yaml",
        "extends: base.yaml\nextract:\n  batch_size: 4\n",
    )
    cfg = load_config(child)
    # child overrode one key...
    assert cfg.extract.batch_size == 4
    # ...and did NOT wipe its sibling. This is the whole point of deep merge:
    # a shallow merge would drop max_length and the run would silently use a
    # different truncation length than the other precision arms.
    assert cfg.extract.max_length == 128
    assert cfg.seed == 1729


def test_extends_cycle_is_detected(tmp_path):
    write_yaml(tmp_path, "a.yaml", "extends: b.yaml\nx: 1\n")
    write_yaml(tmp_path, "b.yaml", "extends: a.yaml\ny: 2\n")
    with pytest.raises(ConfigError, match="cyclic"):
        load_config(tmp_path / "a.yaml")


def test_env_interpolation_with_default(tmp_path, monkeypatch):
    monkeypatch.delenv("QP_TEST_VAR", raising=False)
    path = write_yaml(tmp_path, "a.yaml", 'p: "${QP_TEST_VAR:-fallback/path}"\n')
    assert load_config(path).p == "fallback/path"


def test_env_interpolation_prefers_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("QP_TEST_VAR", "D:/real/path")
    path = write_yaml(tmp_path, "a.yaml", 'p: "${QP_TEST_VAR:-fallback}"\n')
    assert load_config(path).p == "D:/real/path"


def test_env_interpolation_without_default_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("QP_TEST_VAR", raising=False)
    path = write_yaml(tmp_path, "a.yaml", 'p: "${QP_TEST_VAR}"\n')
    with pytest.raises(ConfigError, match="QP_TEST_VAR"):
        load_config(path)


def test_require_names_every_missing_key(tmp_path):
    path = write_yaml(tmp_path, "a.yaml", "seed: 7\nmodel:\n  hf_id: x\n")
    cfg = load_config(path)
    require(cfg, "seed", "model.hf_id")  # present: no raise
    with pytest.raises(ConfigError) as exc:
        require(cfg, "seed", "model.missing", "extract.batch_size")
    message = str(exc.value)
    assert "model.missing" in message
    assert "extract.batch_size" in message


def test_to_plain_is_a_plain_dict():
    cfg = Config._wrap({"a": {"b": [1, {"c": 2}]}})
    plain = cfg.to_plain()
    assert type(plain) is dict
    assert type(plain["a"]) is dict
    assert plain["a"]["b"][1]["c"] == 2


# ------------------------------------------------------- shipped repo configs


@pytest.mark.parametrize(
    "name", ["base.yaml", "model_smoke.yaml", "model_llama1b.yaml", "experiment_e1.yaml"]
)
def test_shipped_configs_load(name, monkeypatch):
    monkeypatch.delenv("QUANTPROBE_CACHE", raising=False)
    cfg = load_config(REPO_ROOT / "configs" / name)
    assert cfg.seed == 1729, "every config must inherit the one master seed"


def test_model_configs_declare_expected_architecture():
    """Shape assertions are only useful if the config actually states them."""
    for name in ("model_smoke.yaml", "model_llama1b.yaml"):
        cfg = load_config(REPO_ROOT / "configs" / name)
        for key in ("expected_n_layers", "expected_hidden_states", "expected_d_model"):
            assert key in cfg.model, f"{name} is missing model.{key}"
        assert cfg.model.expected_hidden_states == cfg.model.expected_n_layers + 1


def test_base_config_int4_uses_nf4_and_fp16_compute():
    """Guards the claim that activations stay FP16 under INT4."""
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    int4 = cfg.quantization.int4
    assert int4.load_in_4bit is True
    assert int4.bnb_4bit_quant_type == "nf4"
    assert int4.bnb_4bit_compute_dtype == "float16"


# -------------------------------------------------------------------- seeding


def test_set_seed_makes_python_numpy_and_torch_reproducible():
    set_seed(1729)
    a = (random.random(), np.random.rand(3).tolist())
    set_seed(1729)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_set_seed_records_every_rng_it_touched():
    record = set_seed(42)
    for key in ("master_seed", "PYTHONHASHSEED", "python_random", "numpy", "torch_cpu"):
        assert key in record, f"seed record is missing {key}"
    assert record["master_seed"] == 42


def test_different_seeds_give_different_draws():
    set_seed(1)
    a = np.random.rand(5).tolist()
    set_seed(2)
    b = np.random.rand(5).tolist()
    assert a != b


# ---------------------------------------------------------------------- paths


def test_cache_dir_follows_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTPROBE_CACHE", str(tmp_path / "elsewhere"))
    assert paths.cache_dir() == (tmp_path / "elsewhere").resolve()


def test_cache_dir_defaults_into_repo(monkeypatch):
    monkeypatch.delenv("QUANTPROBE_CACHE", raising=False)
    assert paths.cache_dir() == (paths.repo_root() / "cache").resolve()


def test_free_space_walks_up_to_an_existing_parent(tmp_path):
    """Must work for a cache dir that has not been created yet."""
    deep = tmp_path / "not" / "yet" / "created"
    assert paths.free_space_gb(deep) > 0


def test_ensure_dirs_is_idempotent(tmp_path):
    target = tmp_path / "a" / "b"
    paths.ensure_dirs(target)
    paths.ensure_dirs(target)
    assert target.is_dir()


# ------------------------------------------------------------- model loader


def test_quantized_precisions_refuse_to_run_without_cuda():
    """The single most important guard in the repo for this machine.

    If this ever silently succeeds on a CPU box, we would be comparing an
    INT4 arm that was never actually quantized.
    """
    from src.models import loader

    if loader.cuda_available():
        pytest.skip("machine has CUDA; this guard is for CPU-only boxes")

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    for precision in ("int8", "int4"):
        with pytest.raises(loader.ModelLoadError, match="CUDA"):
            loader.build_quantization_config(precision, cfg["quantization"])


def test_fp16_needs_no_quantization_config():
    from src.models import loader

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert loader.build_quantization_config("fp16", cfg["quantization"]) is None


def test_unknown_precision_is_rejected():
    from src.models import loader

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    with pytest.raises(loader.ModelLoadError, match="unknown precision"):
        loader.build_quantization_config("int2", cfg["quantization"])


def test_dtype_kwarg_matches_installed_transformers():
    """Do not guess API surfaces."""
    import transformers

    from src.models import loader

    name = loader._dtype_kwarg_name()
    assert name in ("dtype", "torch_dtype")
    major, minor = (int(p) for p in transformers.__version__.split(".")[:2])
    assert name == ("dtype" if (major, minor) >= (4, 56) else "torch_dtype")
