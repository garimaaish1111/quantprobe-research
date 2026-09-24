"""E3 and E4 - geometric drift, and whether the probe direction rotates.

**E3.** Per layer, how far the INT8/INT4 hidden states sit from FP16 for the
same inputs, by three measures: relative L2 distance, cosine similarity, and
linear CKA.

Raw L2 is deliberately absent. Phase 2 established why: the residual stream
grows roughly 20x from embeddings to the last block, and `hidden_states[16]`
additionally passes through the final RMSNorm, so `||h||` jumps 17.7 -> 97.2.
An absolute distance inherits all of that and reports a 5.6x "drift spike" at
layer 16 that vanishes the moment you normalise. Every metric here is
scale-invariant or explicitly normalised.

**CKA** answers a different question from the other two. Cosine and relative
distance are per-example: did *this* vector move? CKA is per-representation:
does the *geometry of the whole point cloud* still match - are the same pairs
of examples still close to each other? A representation can have every vector
displaced while preserving all relative structure (a global rotation gives
cosine < 1 everywhere but CKA = 1). Since a linear probe only cares about
structure, not absolute position, CKA is arguably the metric most predictive
of whether a probe still works.

**E4.** The angle between the FP16 probe's weight vector and the INT4 probe's
at the same layer. Distinguishes "the signal moved" from "the probe just got
noisier": a probe that degraded without rotating points the same way.

A subtlety that would otherwise invalidate E4: probe weights are fitted in
*standardised* space, and each arm standardises with its own per-feature std.
Comparing raw coefficient vectors therefore mixes true rotation with the
difference between the two arms' feature scales. The weights are converted
back to raw feature space (`w / sigma`) before the angle is taken.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("quantprobe.analysis.drift")


class DriftError(RuntimeError):
    """A drift metric could not be computed."""


# ----------------------------------------------------------------------- E3


def relative_drift(reference: np.ndarray, other: np.ndarray) -> float:
    """Mean ||other - ref|| / mean ||ref||, over examples.

    Normalised by the reference's own magnitude so the number is comparable
    across layers whose activations differ in scale by 100x.
    """
    _check_pair(reference, other)
    scale = float(np.linalg.norm(reference, axis=1).mean())
    if scale < 1e-12:
        return 0.0
    return float(np.linalg.norm(other - reference, axis=1).mean() / scale)


def mean_cosine(reference: np.ndarray, other: np.ndarray) -> float:
    """Mean per-example cosine similarity. Scale-invariant by construction."""
    _check_pair(reference, other)
    num = (reference * other).sum(axis=1)
    den = np.linalg.norm(reference, axis=1) * np.linalg.norm(other, axis=1)
    return float((num / np.maximum(den, 1e-12)).mean())


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two [N, d] representations. 1.0 = identical geometry.

    Computed in the Gram-free form: with column-centred X and Y,

        CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    This is O(N d^2) rather than the O(N^2 d) kernel form, which matters at
    N=6049, d=2048 - the N x N Gram matrix would be 37M entries per call and
    the d x d one is 4M.

    Invariant to isotropic scaling and to orthogonal transforms of either
    representation, which is exactly the point: it asks whether the relational
    structure survived, not whether the vectors sit in the same place.
    """
    _check_pair(x, y)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    # float64 throughout: at d=2048 the Frobenius sums accumulate enough terms
    # that float32 rounding is visible in the third decimal.
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    cross = np.linalg.norm(y.T @ x, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    if xx < 1e-12 or yy < 1e-12:
        return 0.0
    return float(cross / (xx * yy))


def drift_by_layer(
    reference_hidden,
    other_hidden,
    layers: Sequence[int],
    sample: int | None = None,
    seed: int = 1729,
) -> list[dict[str, Any]]:
    """All three E3 metrics, per layer.

    Args:
        reference_hidden: [N, L, d] memmap for the reference arm (FP16).
        other_hidden: [N, L, d] for the arm being compared.
        sample: cap on rows, for speed. CKA at N=6049, d=2048 is a few
            seconds per layer; sampling is offered but not the default,
            because the full set is affordable and sampling adds a seed
            dependence to a number that should be exact.
    """
    rng = np.random.default_rng(seed)
    n_rows = reference_hidden.shape[0]
    idx = (
        np.sort(rng.choice(n_rows, size=min(sample, n_rows), replace=False))
        if sample
        else slice(None)
    )

    out = []
    for layer in layers:
        ref = np.asarray(reference_hidden[idx, layer, :], dtype=np.float64)
        oth = np.asarray(other_hidden[idx, layer, :], dtype=np.float64)
        out.append(
            {
                "layer": layer,
                "relative_drift": relative_drift(ref, oth),
                "mean_cosine": mean_cosine(ref, oth),
                "linear_cka": linear_cka(ref, oth),
                "reference_norm": float(np.linalg.norm(ref, axis=1).mean()),
            }
        )
    return out


# ----------------------------------------------------------------------- E4


def unstandardize_direction(weights: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    """Map a standardised-space probe direction back to raw feature space.

    A probe scores `((x - mu) / sigma) . w`, which equals `x . (w / sigma)` up
    to a constant. So `w / sigma` is the direction in the space the two arms
    actually share. Comparing raw `w` across arms would fold each arm's own
    feature scaling into the "rotation".
    """
    safe = np.where(np.abs(feature_std) < 1e-8, 1.0, feature_std)
    direction = weights / safe
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else direction


def direction_angle(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Angle between two probe directions, in degrees, plus the cosine.

    Sign is ignored: a probe direction and its negation define the same
    decision boundary with the labels swapped, and sklearn's solver is free to
    land on either. Taking |cos| avoids reporting a 180-degree "rotation" that
    is an artefact of solver sign convention.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        raise DriftError("cannot take the angle of a zero-length direction")
    cos = float(np.clip(abs(np.dot(a, b) / (na * nb)), -1.0, 1.0))
    return {"cosine": cos, "angle_deg": float(np.degrees(np.arccos(cos)))}


# ------------------------------------------------- restated H4 correlation


def correlate(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Pearson and Spearman between two per-layer series.

    Both, because they answer different questions and can disagree
    informatively: Pearson asks whether the relationship is *linear*, Spearman
    only whether it is monotonic. With ~16 points, either is easy to move with
    one outlier, so n is reported alongside and neither should be quoted
    without it.
    """
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise DriftError(f"series lengths differ: {x.size} vs {y.size}")
    if x.size < 3:
        raise DriftError(f"need at least 3 points to correlate, got {x.size}")

    # A constant series has no correlation to report, and scipy returns NaN
    # with a warning rather than saying so.
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {
            "n": int(x.size),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "note": "one series is constant - correlation undefined",
        }

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise DriftError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise DriftError(f"expected [N, d], got {a.shape}")
    if a.shape[0] == 0:
        raise DriftError("no rows")
