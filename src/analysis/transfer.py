"""E2 - cross-precision probe transfer.

Train a probe on precision P, test it on precision Q, for all 9 (P, Q) pairs at
every layer. The diagonal reproduces E1; the off-diagonal is the experiment.

**The decision that determines whether E2 means anything: the scaler travels
with the probe.**

A probe is not just a weight vector. It is (mean, std, weights) - the
standardisation is part of the fitted object. So when an FP16-trained probe is
applied to INT4 activations, it standardises them using the *FP16* training
statistics, because those are the only statistics a deployed detector would
have. Re-standardising with INT4's own statistics would silently hand the
probe a free recalibration - exactly the thing H3 says you have to do
deliberately - and every off-diagonal cell would improve for a reason that has
nothing to do with the representation.

That single choice is the difference between measuring transfer and measuring
nothing. It is also why this module refuses to reuse `probes.linear.fit_*`,
which standardises internally: here the fit and the application must be
separable.

**Why the probe is fitted once and applied three times.** A probe trained on
FP16 fold `cities` is the same object regardless of which arm it is evaluated
on. Fitting it separately per test arm would be identical work with a
different random seed's worth of noise, and 3x the cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("quantprobe.analysis.transfer")


class TransferError(RuntimeError):
    """Cross-precision transfer could not be computed."""


@dataclass
class FittedProbe:
    """A probe plus the standardisation it was fitted with.

    The `mean` and `std` are not incidental - they are what makes this
    applicable to a different arm's raw activations, and keeping them together
    with the weights is what stops someone accidentally re-standardising.
    """

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    intercept: float

    def score(self, x_raw: np.ndarray) -> np.ndarray:
        """Apply to raw features, standardising with THIS probe's statistics."""
        return ((x_raw - self.mean) / self.std) @ self.weights + self.intercept


def fit_transferable_probe(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    C: float = 1.0,
    max_iter: int = 2000,
) -> FittedProbe:
    """Fit a probe and return it WITH its standardisation, unapplied."""
    from sklearn.linear_model import LogisticRegression

    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    # L2 is the default; passing it explicitly is deprecated in sklearn 1.8.
    model = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs", random_state=seed
    )
    model.fit((x_train_raw - mean) / std, y_train)

    return FittedProbe(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        weights=model.coef_.ravel().astype(np.float32),
        intercept=float(model.intercept_[0]),
    )


def transfer_matrix_for_layer(
    arms: dict[str, np.ndarray],
    labels: np.ndarray,
    folds: Sequence[tuple[str, np.ndarray, np.ndarray]],
    layer: int,
    seed: int,
    C: float = 1.0,
    max_iter: int = 2000,
) -> list[dict[str, Any]]:
    """All 9 (train, test) cells for one layer.

    Args:
        arms: precision -> [N, d_model] features for THIS layer.
        labels: [N] in {0, 1}.
        folds: (name, train_idx, test_idx).

    Returns:
        {"rows": one row per (train, test, fold),
         "directions": {(precision, fold): unit direction in RAW feature space}}
    """
    from sklearn.metrics import roc_auc_score

    precisions = list(arms)
    rows: list[dict[str, Any]] = []
    # Per-fold probe directions, kept so E4 can measure its own noise floor.
    # See `within_arm_angles` below for why that is not optional.
    directions: dict[tuple[str, str], np.ndarray] = {}

    for fold_name, train_idx, test_idx in folds:
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Fit once per training arm, then apply to every test arm.
        fitted = {
            p: fit_transferable_probe(
                arms[p][train_idx], y_train, seed=seed, C=C, max_iter=max_iter
            )
            for p in precisions
        }

        for train_p in precisions:
            probe = fitted[train_p]
            # Raw-space direction: w / sigma. Each arm standardises with its
            # own per-feature std, so comparing standardised weights across
            # arms would fold that scaling difference into the "rotation".
            safe = np.where(np.abs(probe.std) < 1e-8, 1.0, probe.std)
            raw = probe.weights / safe
            norm = np.linalg.norm(raw)
            directions[(train_p, fold_name)] = raw / norm if norm > 0 else raw

            for test_p in precisions:
                scores = probe.score(arms[test_p][test_idx])
                rows.append(
                    {
                        "layer": layer,
                        "train_precision": train_p,
                        "test_precision": test_p,
                        "fold": fold_name,
                        "auroc": float(roc_auc_score(y_test, scores)),
                        "matched": train_p == test_p,
                    }
                )
    return {"rows": rows, "directions": directions}


def aggregate_transfer(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse folds to mean +/- std per (layer, train, test)."""
    buckets: dict[tuple[int, str, str], list[float]] = {}
    for r in rows:
        buckets.setdefault(
            (r["layer"], r["train_precision"], r["test_precision"]), []
        ).append(r["auroc"])

    out = []
    for (layer, train_p, test_p), values in sorted(buckets.items()):
        arr = np.array(values)
        out.append(
            {
                "layer": layer,
                "train_precision": train_p,
                "test_precision": test_p,
                "matched": train_p == test_p,
                "auroc_mean": float(arr.mean()),
                "auroc_std": float(arr.std(ddof=0)),
                "n_folds": len(values),
            }
        )
    return out


def transfer_loss(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (layer, train, test): how much AUROC is lost versus the matched probe.

    Defined against the probe trained AND tested on the *test* arm - i.e. the
    native probe for that representation, which is what you would have if you
    recalibrated. So `loss` answers the deployment question directly: how much
    accuracy does refusing to recalibrate cost you?

    Positive loss = the transferred probe is worse than a native one.
    """
    by_key = {
        (r["layer"], r["train_precision"], r["test_precision"]): r for r in rows
    }
    out = []
    for r in rows:
        native = by_key.get((r["layer"], r["test_precision"], r["test_precision"]))
        if native is None:
            continue
        out.append(
            {
                "layer": r["layer"],
                "train_precision": r["train_precision"],
                "test_precision": r["test_precision"],
                "matched": r["matched"],
                "auroc_mean": r["auroc_mean"],
                "native_auroc": native["auroc_mean"],
                "loss": float(native["auroc_mean"] - r["auroc_mean"]),
            }
        )
    return out


def matrix_at_layer(
    rows: Sequence[dict[str, Any]], layer: int, precisions: Sequence[str]
) -> np.ndarray:
    """A 3x3 array of mean AUROC, rows = train, cols = test."""
    lookup = {
        (r["train_precision"], r["test_precision"]): r["auroc_mean"]
        for r in rows
        if r["layer"] == layer
    }
    missing = [
        (a, b) for a in precisions for b in precisions if (a, b) not in lookup
    ]
    if missing:
        raise TransferError(f"layer {layer} is missing cells: {missing}")
    return np.array(
        [[lookup[(a, b)] for b in precisions] for a in precisions], dtype=np.float64
    )


def best_transfer_layer(rows: Sequence[dict[str, Any]], precisions: Sequence[str]) -> int:
    """The layer whose matched probes are strongest - where E2 is worth reading.

    Transfer at a layer with no signal is uninterpretable: two probes that both
    score 0.50 "transfer perfectly" and mean nothing.
    """
    by_layer: dict[int, list[float]] = {}
    for r in rows:
        if r["matched"]:
            by_layer.setdefault(r["layer"], []).append(r["auroc_mean"])
    if not by_layer:
        raise TransferError("no matched (diagonal) cells found")
    return max(by_layer, key=lambda L: float(np.mean(by_layer[L])))


def within_arm_angles(
    directions: dict[tuple[str, str], np.ndarray],
    precision: str,
    folds: Sequence[str],
) -> list[float]:
    """Pairwise angles between probes fitted on the SAME arm, different folds.

    This is E4's noise floor, and without it E4 reports a number that cannot
    be read. A logistic probe in 2048 dimensions fitted on ~5000 rows is badly
    under-determined: many directions separate the classes about equally well,
    and the solver picks one. Two probes on *identical* data with different
    folds can therefore sit tens of degrees apart while describing the same
    signal.

    So "the INT4 probe is 50 degrees from the FP16 probe" is meaningless until
    compared against "two FP16 probes are N degrees apart". If N is also ~50,
    E4 has measured sampling noise. Only the excess over N is rotation
    attributable to quantization.
    """
    import itertools

    angles = []
    for a, b in itertools.combinations(folds, 2):
        va, vb = directions.get((precision, a)), directions.get((precision, b))
        if va is None or vb is None:
            continue
        cos = float(np.clip(abs(float(np.dot(va, vb))), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos))))
    return angles


def cross_arm_angles(
    directions: dict[tuple[str, str], np.ndarray],
    reference: str,
    other: str,
    folds: Sequence[str],
) -> list[float]:
    """Angle between the reference and other arm's probe, fold by fold.

    Compared fold-for-fold rather than mean-to-mean, so the comparison sees
    exactly the same training rows on both sides and any difference is
    attributable to the representation rather than to the sample.
    """
    angles = []
    for fold in folds:
        va, vb = directions.get((reference, fold)), directions.get((other, fold))
        if va is None or vb is None:
            continue
        cos = float(np.clip(abs(float(np.dot(va, vb))), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos))))
    return angles
