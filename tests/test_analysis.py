"""Phase 4 tests - transfer, drift metrics, probe rotation, correlation.

Two tests carry most of the weight:

`test_transferred_probe_uses_its_own_scaler` - if the scaler were refit on the
test arm, every off-diagonal cell would improve for a reason unrelated to the
representation, and E2 would measure nothing.

`test_within_arm_angle_is_large_in_high_dimensions` - establishes that E4's
noise floor is real and not a formality. A 2048-dim probe on a few thousand
rows is under-determined, so two probes on identical data land tens of degrees
apart. Without that baseline, a cross-arm angle is uninterpretable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import drift as D  # noqa: E402
from src.analysis import transfer as T  # noqa: E402
from src.data.schema import Record  # noqa: E402
from src.data.splits import make_folds  # noqa: E402


def make_records(n_per_topic=40, topics=("a", "b", "c")):
    return [
        Record(id=f"s:{t}:{i:03d}", prompt="", statement=f"{t}{i}",
               label=i % 2, topic=t, source="s")
        for t in topics for i in range(n_per_topic)
    ]


# ------------------------------------------------------------------ transfer


def test_fitted_probe_keeps_its_standardisation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6)) * 10 + 5
    y = (x[:, 0] > 5).astype(int)
    probe = T.fit_transferable_probe(x, y, seed=1729)
    assert probe.mean.shape == (6,) and probe.std.shape == (6,)
    assert np.all(probe.std > 0)


def test_transferred_probe_uses_its_own_scaler():
    """The scaler is part of the detector, so it must NOT be refit on the
    target arm. Refitting would be a free recalibration - exactly the thing
    H3 says you have to do deliberately."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 5))
    y = (x[:, 0] > 0).astype(int)
    probe = T.fit_transferable_probe(x, y, seed=1729)

    shifted = x + 100.0  # a different arm with a wildly different offset
    scores_correct = probe.score(shifted)

    # What a bug would compute: re-standardise on the target's own statistics.
    m, s = shifted.mean(axis=0), shifted.std(axis=0)
    scores_leaky = ((shifted - m) / s) @ probe.weights + probe.intercept

    assert not np.allclose(scores_correct, scores_leaky), (
        "re-standardising on the target arm must change the scores - if it "
        "does not, the test cannot detect the bug it exists for"
    )
    # The correct behaviour carries the offset through as a large shift.
    assert abs(scores_correct.mean()) > abs(scores_leaky.mean())


def test_transfer_matrix_produces_all_nine_cells():
    records = make_records()
    folds = make_folds(records, "leave_one_topic_out")
    labels = np.array([r.label for r in records])
    rng = np.random.default_rng(2)
    base = rng.normal(size=(len(records), 8))
    base[labels == 1, 0] += 1.5
    arms = {"fp16": base, "int8": base + rng.normal(scale=0.05, size=base.shape),
            "int4": base + rng.normal(scale=0.20, size=base.shape)}

    out = T.transfer_matrix_for_layer(arms, labels, folds, layer=3, seed=1729)
    rows = out["rows"]
    assert len({(r["train_precision"], r["test_precision"]) for r in rows}) == 9
    assert len(rows) == 9 * len(folds)
    assert all(r["layer"] == 3 for r in rows)
    # directions captured for every (arm, fold)
    assert len(out["directions"]) == 3 * len(folds)
    for vec in out["directions"].values():
        assert np.isclose(np.linalg.norm(vec), 1.0)


def test_aggregate_and_loss_are_consistent():
    rows = [
        {"layer": 1, "train_precision": a, "test_precision": b, "fold": f,
         "auroc": 0.8 if a == b else 0.7, "matched": a == b}
        for a in ("fp16", "int4") for b in ("fp16", "int4") for f in ("x", "y")
    ]
    agg = T.aggregate_transfer(rows)
    losses = {(r["train_precision"], r["test_precision"]): r["loss"]
              for r in T.transfer_loss(agg)}
    assert losses[("fp16", "fp16")] == pytest.approx(0.0)
    assert losses[("fp16", "int4")] == pytest.approx(0.1)


def test_matrix_at_layer_orients_rows_as_train():
    rows = [
        {"layer": 0, "train_precision": a, "test_precision": b,
         "auroc_mean": 0.9 if a == "fp16" else 0.5, "matched": a == b}
        for a in ("fp16", "int4") for b in ("fp16", "int4")
    ]
    m = T.matrix_at_layer(rows, 0, ["fp16", "int4"])
    assert m[0, 0] == 0.9 and m[0, 1] == 0.9   # row 0 = trained on fp16
    assert m[1, 0] == 0.5 and m[1, 1] == 0.5


def test_matrix_at_layer_reports_missing_cells():
    with pytest.raises(T.TransferError, match="missing cells"):
        T.matrix_at_layer([], 0, ["fp16", "int4"])


def test_best_transfer_layer_picks_the_strongest_diagonal():
    rows = [
        {"layer": L, "train_precision": "fp16", "test_precision": "fp16",
         "auroc_mean": a, "matched": True}
        for L, a in [(0, 0.51), (5, 0.80), (9, 0.66)]
    ]
    assert T.best_transfer_layer(rows, ["fp16"]) == 5


def test_within_arm_angle_is_large_in_high_dimensions():
    """E4's noise floor is real, not a formality.

    Two probes fitted on DIFFERENT samples of the SAME distribution, in 512
    dimensions, land far apart - because the problem is under-determined and
    many directions separate the classes about equally. Any cross-arm angle
    at or below this is not evidence of rotation.
    """
    rng = np.random.default_rng(7)
    d = 512
    directions = {}
    for fold in ("a", "b", "c"):
        x = rng.normal(size=(400, d))
        y = (x[:, 0] > 0).astype(int)
        probe = T.fit_transferable_probe(x, y, seed=1729)
        raw = probe.weights / np.where(np.abs(probe.std) < 1e-8, 1.0, probe.std)
        directions[("fp16", fold)] = raw / np.linalg.norm(raw)

    angles = T.within_arm_angles(directions, "fp16", ["a", "b", "c"])
    assert len(angles) == 3
    assert np.mean(angles) > 15.0, (
        f"expected a substantial noise floor in {d} dims, got {np.mean(angles):.1f} deg - "
        f"if this is near zero, E4 needs no baseline and this test is wrong"
    )


def test_cross_arm_angle_is_zero_for_identical_representations():
    """Identical features must give identical probes - the layer-0 case."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=(300, 32))
    y = (x[:, 0] > 0).astype(int)
    probe = T.fit_transferable_probe(x, y, seed=1729)
    raw = probe.weights / probe.std
    unit = raw / np.linalg.norm(raw)
    directions = {("fp16", "f"): unit, ("int4", "f"): unit.copy()}
    assert T.cross_arm_angles(directions, "fp16", "int4", ["f"])[0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------- drift


def test_identical_representations_have_zero_drift():
    x = np.random.default_rng(0).normal(size=(100, 16))
    assert D.relative_drift(x, x) == pytest.approx(0.0)
    assert D.mean_cosine(x, x) == pytest.approx(1.0)
    assert D.linear_cka(x, x) == pytest.approx(1.0)


def test_relative_drift_is_scale_invariant_in_the_reference():
    """The whole reason raw L2 was rejected: at layer 16 the residual stream
    is 5.5x longer and an absolute distance inherits that."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 16))
    delta = rng.normal(size=(100, 16)) * 0.1
    small = D.relative_drift(x, x + delta)
    big = D.relative_drift(x * 50, (x + delta) * 50)
    assert small == pytest.approx(big, rel=1e-9)


def test_cka_is_invariant_to_isotropic_scaling_and_rotation():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    assert D.linear_cka(x, x * 7.0) == pytest.approx(1.0, abs=1e-9)
    assert D.linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-9)


def test_cka_falls_for_unrelated_representations():
    rng = np.random.default_rng(3)
    assert D.linear_cka(rng.normal(size=(300, 20)), rng.normal(size=(300, 20))) < 0.35


def test_cosine_detects_rotation_that_cka_does_not():
    """The two metrics answer different questions, and this is the case that
    separates them: a global rotation moves every vector but preserves all
    relative structure."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    rotated = x @ q
    assert D.linear_cka(x, rotated) == pytest.approx(1.0, abs=1e-9)
    assert abs(D.mean_cosine(x, rotated)) < 0.5


def test_drift_metrics_reject_mismatched_shapes():
    a = np.zeros((10, 4))
    with pytest.raises(D.DriftError, match="shape mismatch"):
        D.relative_drift(a, np.zeros((10, 5)))


def test_unstandardize_direction_returns_a_unit_vector():
    w = np.array([2.0, -4.0, 1.0])
    std = np.array([2.0, 4.0, 0.5])
    out = D.unstandardize_direction(w, std)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert np.allclose(out, np.array([1.0, -1.0, 2.0]) / np.linalg.norm([1.0, -1.0, 2.0]))


def test_unstandardize_survives_a_zero_std():
    out = D.unstandardize_direction(np.array([1.0, 1.0]), np.array([1.0, 0.0]))
    assert np.isfinite(out).all()


def test_direction_angle_ignores_sign():
    """A probe direction and its negation are the same boundary; the solver
    may return either, and a 180-degree flip is not rotation."""
    v = np.array([1.0, 0.0, 0.0])
    assert D.direction_angle(v, -v)["angle_deg"] == pytest.approx(0.0)
    assert D.direction_angle(v, np.array([0.0, 1.0, 0.0]))["angle_deg"] == pytest.approx(90.0)


def test_direction_angle_rejects_a_zero_vector():
    with pytest.raises(D.DriftError, match="zero-length"):
        D.direction_angle(np.zeros(3), np.array([1.0, 0.0, 0.0]))


# --------------------------------------------------------------- correlation


def test_correlate_finds_a_perfect_relationship():
    out = D.correlate([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert out["pearson_r"] == pytest.approx(1.0)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["n"] == 5


def test_correlate_reports_a_constant_series_instead_of_nan():
    """scipy returns NaN with a warning; saying so explicitly is more useful,
    and this is the exact case the original H4 ran into."""
    out = D.correlate([1, 2, 3, 4], [5, 5, 5, 5])
    assert out["pearson_r"] is None
    assert "constant" in out["note"]


def test_correlate_needs_enough_points():
    with pytest.raises(D.DriftError, match="at least 3"):
        D.correlate([1, 2], [3, 4])


def test_correlate_rejects_mismatched_lengths():
    with pytest.raises(D.DriftError, match="lengths differ"):
        D.correlate([1, 2, 3], [1, 2])
