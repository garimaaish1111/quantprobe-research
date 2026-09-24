"""Phase 3 tests - alignment, scaling hygiene, probe fitting, aggregation.

The centrepiece is `test_probe_recovers_a_planted_signal`, which builds hidden
states with a known truth direction AND a much larger topic-specific nuisance
direction, then checks the probe recovers the former without exploiting the
latter. That is the only test here that could catch a leaking split, because a
leaking split produces high AUROC - which looks like success.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import Record  # noqa: E402
from src.data.splits import make_folds  # noqa: E402
from src.probes import linear as P  # noqa: E402


def make_records(n_per_topic: int = 60, topics=("a", "b", "c", "d", "e", "f")):
    out = []
    for t in topics:
        for i in range(n_per_topic):
            out.append(
                Record(
                    id=f"src:{t}:{i:04d}",
                    prompt="",
                    statement=f"{t} statement {i}",
                    label=i % 2,
                    topic=t,
                    source="src",
                )
            )
    return out


# ------------------------------------------------------------------ alignment


def test_align_records_maps_cached_order_onto_the_dataset():
    records = make_records(3, ("a", "b"))
    cached = [r.id for r in reversed(records)]
    idx = P.align_records(cached, records)
    assert [records[i].id for i in idx] == cached


def test_align_records_rejects_a_length_mismatch():
    records = make_records(3, ("a",))
    with pytest.raises(P.ProbeError, match="not the same run"):
        P.align_records([r.id for r in records[:2]], records)


def test_align_records_rejects_unknown_ids():
    records = make_records(3, ("a",))
    cached = [r.id for r in records]
    cached[0] = "src:a:9999"
    with pytest.raises(P.ProbeError, match="absent from the dataset"):
        P.align_records(cached, records)


def test_align_records_rejects_duplicate_dataset_ids():
    records = make_records(2, ("a",))
    records = records + [records[0]]
    with pytest.raises(P.ProbeError, match="duplicate record id"):
        P.align_records([r.id for r in records], records)


# ------------------------------------------------------------- scaling hygiene


def test_standardize_uses_train_statistics_only():
    """Fitting the scaler on all data leaks test mean/variance into training.

    It produces a better number, silently, with no error. This asserts the
    test fold is transformed by the TRAIN statistics, not its own.
    """
    x_train = np.array([[0.0], [2.0]])          # train mean 1, std 1
    x_test = np.array([[10.0], [20.0]])         # wildly different scale
    _, scaled_test = P._standardize(x_train, x_test)
    assert np.allclose(scaled_test.ravel(), [9.0, 19.0])
    assert abs(scaled_test.mean()) > 1.0, "test set must NOT be re-centred on itself"


def test_standardize_survives_a_constant_feature():
    """A zero-variance column would divide by zero and NaN the whole fit."""
    x_train = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    x_test = np.array([[2.0, 5.0]])
    scaled_train, scaled_test = P._standardize(x_train, x_test)
    assert np.isfinite(scaled_train).all()
    assert np.isfinite(scaled_test).all()
    assert scaled_test[0, 1] == 0.0


# ------------------------------------------------------------------- fitting


def separable(n=400, d=12, sep=1.4, seed=0):
    """Balanced, SHUFFLED separable data.

    Shuffled deliberately: np.repeat([0,1], n//2) puts every negative first,
    so a positional train/test slice hands the test side a single class and
    AUROC is undefined. That is a bug in the fixture, but it is exactly the
    shape of bug that produces confusing failures in real pipelines too.
    """
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    x = rng.normal(size=(n, d))
    x[y == 1, 0] += sep
    order = rng.permutation(n)
    return x[order], y[order]


def test_logistic_recovers_a_separable_signal():
    x, y = separable()
    out = P.fit_logistic(x[:300], y[:300], x[300:], y[300:], seed=1729)
    assert out["auroc"] > 0.8
    assert out["weights"].shape == (12,)
    assert out["converged"]


def test_mass_mean_recovers_a_separable_signal():
    x, y = separable()
    out = P.fit_mass_mean(x[:300], y[:300], x[300:], y[300:])
    assert out["auroc"] > 0.8
    assert out["weights"].shape == (12,)


def test_both_probes_sit_at_chance_on_pure_noise():
    """A probe that scores well on noise has a leak somewhere."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(600, 20))
    y = rng.integers(0, 2, size=600)
    for fn in (
        lambda: P.fit_logistic(x[:400], y[:400], x[400:], y[400:], seed=1729),
        lambda: P.fit_mass_mean(x[:400], y[:400], x[400:], y[400:]),
    ):
        assert 0.35 < fn()["auroc"] < 0.65


def test_mass_mean_rejects_a_single_class_training_fold():
    x = np.random.default_rng(0).normal(size=(20, 4))
    y = np.ones(20, dtype=int)
    with pytest.raises(P.ProbeError, match="both classes"):
        P.fit_mass_mean(x, y, x, y)


def test_unknown_probe_type_is_rejected():
    records = make_records(10, ("a", "b"))
    folds = make_folds(records, "leave_one_topic_out")
    features = np.random.default_rng(0).normal(size=(len(records), 5))
    labels = np.array([r.label for r in records])
    with pytest.raises(P.ProbeError, match="unknown probe type"):
        P.probe_layer(features, labels, folds, "fp16", 0, seed=1, probe_types=["magic"])


def test_probe_layer_rejects_a_shape_mismatch():
    records = make_records(10, ("a", "b"))
    folds = make_folds(records, "leave_one_topic_out")
    labels = np.array([r.label for r in records])
    with pytest.raises(P.ProbeError, match="rows"):
        P.probe_layer(
            np.zeros((len(records) + 1, 5)), labels, folds, "fp16", 0, seed=1
        )


# ------------------------------------------- the test that matters most


def test_probe_recovers_a_planted_signal_without_using_topic_structure():
    """End-to-end: known truth direction + a LARGER topic nuisance direction.

    The nuisance is deliberately 3x the signal. If the topic split leaked, the
    probe would latch onto the topic offset and score far above what the
    planted signal alone can support - and at the control layer, which has no
    signal at all, it would score well above chance.
    """
    rng = np.random.default_rng(11)
    records = make_records(80)
    topics = sorted({r.topic for r in records})
    topic_id = np.array([topics.index(r.topic) for r in records])
    labels = np.array([r.label for r in records])

    d = 24
    truth = rng.normal(size=d)
    truth /= np.linalg.norm(truth)
    topic_dirs = rng.normal(size=(len(topics), d)) * 3.0   # nuisance >> signal

    def build(strength: float) -> np.ndarray:
        x = rng.normal(size=(len(records), d))
        x += topic_dirs[topic_id]
        x += np.outer((2 * labels - 1) * strength, truth)
        return x

    folds = make_folds(records, "leave_one_topic_out")

    signal_rows = P.aggregate(
        P.probe_layer(build(1.0), labels, folds, "fp16", 1, seed=1729)
    )
    control_rows = P.aggregate(
        P.probe_layer(build(0.0), labels, folds, "fp16", 0, seed=1729)
    )

    signal = [r for r in signal_rows if r["probe_type"] == "logistic"][0]
    control = [r for r in control_rows if r["probe_type"] == "logistic"][0]

    assert signal["auroc_mean"] > 0.70, "planted signal was not recovered"
    assert 0.40 < control["auroc_mean"] < 0.60, (
        f"control layer has NO planted signal but scored "
        f"{control['auroc_mean']:.3f} - the topic split is leaking"
    )
    assert signal["n_folds"] == 6


# --------------------------------------------------------------- aggregation


def test_aggregate_reports_mean_and_std_over_folds():
    records = make_records(40, ("a", "b", "c"))
    folds = make_folds(records, "leave_one_topic_out")
    x, _ = separable(n=len(records), d=8)
    labels = np.array([r.label for r in records])
    rows = P.aggregate(P.probe_layer(x, labels, folds, "int4", 7, seed=1729))

    assert {r["probe_type"] for r in rows} == {"logistic", "mass_mean"}
    for r in rows:
        assert r["n_folds"] == 3
        assert r["layer"] == 7 and r["precision"] == "int4"
        assert r["auroc_min"] <= r["auroc_mean"] <= r["auroc_max"]
        assert set(r["per_fold"]) == {"a", "b", "c"}


def test_best_layer_picks_the_highest_mean():
    rows = [
        {"precision": "fp16", "probe_type": "logistic", "layer": L, "auroc_mean": a}
        for L, a in [(0, 0.51), (1, 0.72), (2, 0.68)]
    ]
    assert P.best_layer(rows, "fp16", "logistic")["layer"] == 1


def test_best_layer_errors_on_an_absent_combination():
    with pytest.raises(P.ProbeError, match="no results"):
        P.best_layer([], "fp16", "logistic")


def test_leakage_warning_fires_on_suspiciously_high_auroc():
    """Say so when a number is too good."""
    rows = [{"precision": "fp16", "probe_type": "logistic", "layer": 5, "auroc_mean": 0.994}]
    flags = P.leakage_warnings(rows, threshold=0.98)
    assert len(flags) == 1 and "leakage" in flags[0]
    assert P.leakage_warnings(
        [{"precision": "fp16", "probe_type": "logistic", "layer": 5, "auroc_mean": 0.87}]
    ) == []


def test_mean_direction_is_unit_length():
    records = make_records(40, ("a", "b", "c"))
    folds = make_folds(records, "leave_one_topic_out")
    x, _ = separable(n=len(records), d=8)
    labels = np.array([r.label for r in records])
    results = P.probe_layer(x, labels, folds, "fp16", 3, seed=1729)
    direction = P.mean_direction(results, "fp16", "logistic", 3)
    assert direction.shape == (8,)
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_mean_direction_errors_when_absent():
    with pytest.raises(P.ProbeError, match="no weight vectors"):
        P.mean_direction([], "fp16", "logistic", 0)
