"""Tests for probe-guided abstention.

The load-bearing property is that a *useless* probe must show no benefit. A
coverage-risk curve is easy to write in a way that looks like it helps even on
random scores - if it did, the whole reduction claim would be an artefact of
the metric rather than of the probe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import abstention as A  # noqa: E402


def perfect(n=200):
    labels = np.repeat([0, 1], n // 2)
    return labels.astype(float) * 10.0, labels


def useless(n=400, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    return rng.normal(size=n), labels


# ---------------------------------------------------------------- the curve


def test_full_coverage_equals_the_base_error_rate():
    """At 100% coverage nothing is filtered, so risk must equal the baseline.

    If this fails, the metric is crediting the probe for something before it
    has made a single decision.
    """
    scores, labels = useless()
    rows = A.coverage_risk_curve(scores, labels)
    full = max(rows, key=lambda r: r["coverage"])
    assert full["coverage"] == pytest.approx(1.0)
    assert full["false_assertion_rate"] == pytest.approx(full["baseline_false_rate"])
    assert full["risk_reduction"] == pytest.approx(0.0, abs=1e-9)


def test_a_perfect_probe_drives_risk_to_zero():
    scores, labels = perfect()
    rows = A.coverage_risk_curve(scores, labels)
    half = A.operating_point(rows, 0.5)
    assert half["false_assertion_rate"] == pytest.approx(0.0)
    assert half["risk_reduction"] == pytest.approx(1.0)


def test_a_useless_probe_buys_nothing():
    """The test that stops the whole reduction claim being a metric artefact."""
    scores, labels = useless(n=4000)
    rows = A.coverage_risk_curve(scores, labels)
    base = rows[0]["baseline_false_rate"]
    at_half = A.operating_point(rows, 0.5)["false_assertion_rate"]
    assert abs(at_half - base) < 0.05, (
        f"random scores moved the error rate from {base:.3f} to {at_half:.3f} - "
        f"the metric is crediting a probe that carries no information"
    )


def test_risk_falls_monotonically_ish_as_coverage_drops():
    scores, labels = perfect(400)
    rows = sorted(A.coverage_risk_curve(scores, labels), key=lambda r: -r["coverage"])
    risks = [r["false_assertion_rate"] for r in rows]
    assert risks[0] >= risks[-1]


def test_coverage_and_abstention_sum_to_one():
    scores, labels = useless()
    for row in A.coverage_risk_curve(scores, labels):
        assert row["coverage"] + row["abstention_rate"] == pytest.approx(1.0)


def test_curve_rejects_mismatched_shapes():
    with pytest.raises(A.AbstentionError, match="shape mismatch"):
        A.coverage_risk_curve(np.zeros(5), np.zeros(4))


def test_curve_rejects_empty_input():
    with pytest.raises(A.AbstentionError, match="no rows"):
        A.coverage_risk_curve(np.array([]), np.array([]))


# ------------------------------------------------------------------- AUC


def test_auc_is_lower_for_a_better_probe():
    good = A.area_under_coverage_risk(A.coverage_risk_curve(*perfect(400)))
    bad = A.area_under_coverage_risk(A.coverage_risk_curve(*useless(4000)))
    assert good < bad


def test_auc_of_a_useless_probe_sits_near_the_base_rate():
    scores, labels = useless(4000)
    rows = A.coverage_risk_curve(scores, labels)
    assert A.area_under_coverage_risk(rows) == pytest.approx(
        rows[0]["baseline_false_rate"], abs=0.05
    )


def test_auc_needs_two_points():
    with pytest.raises(A.AbstentionError, match="at least two"):
        A.area_under_coverage_risk([{"coverage": 1.0, "false_assertion_rate": 0.5}])


# ------------------------------------------------------- operating points


def test_operating_point_finds_the_nearest_coverage():
    scores, labels = useless()
    rows = A.coverage_risk_curve(scores, labels)
    op = A.operating_point(rows, 0.7)
    assert abs(op["coverage"] - 0.7) < 0.1


def test_operating_point_errors_on_an_empty_curve():
    with pytest.raises(A.AbstentionError, match="no curve"):
        A.operating_point([], 0.5)


# ------------------------------------------------------------ end to end


def test_evaluate_with_abstention_beats_baseline_on_real_signal():
    from src.data.schema import Record
    from src.data.splits import make_folds

    rng = np.random.default_rng(5)
    records = [
        Record(id=f"s:{t}:{i}", prompt="", statement="x", label=i % 2, topic=t, source="s")
        for t in ("a", "b", "c") for i in range(120)
    ]
    labels = np.array([r.label for r in records])
    d = 16
    direction = rng.normal(size=d)
    features = rng.normal(size=(len(records), d))
    features += np.outer((2 * labels - 1) * 1.2, direction / np.linalg.norm(direction))

    folds = make_folds(records, "leave_one_topic_out")
    out = A.evaluate_with_abstention(features, labels, folds, seed=1729)

    assert out["n"] == len(records)
    assert out["auc_coverage_risk"] < out["baseline_false_rate"]
    assert out["operating_points"]["coverage_50"]["false_assertion_rate"] < \
        out["baseline_false_rate"]
