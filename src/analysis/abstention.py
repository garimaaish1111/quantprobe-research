"""Turning the probe into an intervention.

Everything up to here *measures* whether a truthfulness signal survives
quantization. This module answers the obvious follow-up: **can you use it?**

The operation is deliberately simple. Score every statement with the probe. If
the score falls below a threshold, refuse to assert it. Sweep the threshold and
report what you buy and what you pay.

**Why this is worth doing on top of detection.**
Detection is the prerequisite for reduction - you cannot suppress a
hallucination you cannot identify - but "our probe reaches 0.72 AUROC" is a
statement about a curve, not about anything anyone would deploy. "At 30%
abstention the model's false-assertion rate drops from 50% to 21%" is the same
fact in the units a reader actually cares about.

**The number that matters is the coverage-risk curve, not accuracy.** An
abstaining system has two dials pointing in opposite directions: how much it
answers (coverage) and how often it is wrong when it does (risk). Reporting
either alone is meaningless, because you can drive risk to zero by abstaining
from everything. They are reported together, and summarised by the area under
the curve so two systems can be compared with one number.

**Thresholds come from the TRAINING fold.** Choosing the threshold that looks
best on the test fold is the same leak as fitting a scaler on it: the reported
risk would be optimistic by exactly the amount of freedom you gave yourself.
Each fold's threshold is set on its own training data and then applied
unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("quantprobe.analysis.abstention")


class AbstentionError(RuntimeError):
    """An abstention curve could not be computed."""


def coverage_risk_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_points: int = 21,
) -> list[dict[str, float]]:
    """Sweep an abstention threshold; report coverage against error rate.

    A statement is *asserted* when its probe score is at or above the
    threshold, and every asserted statement whose label is 0 is a false
    assertion. Sweeping the threshold from "assert everything" to "assert
    almost nothing" traces what abstention buys.

    Args:
        scores: probe decision values, higher = more likely true.
        labels: 1 = true statement, 0 = false.
        n_points: how many thresholds to sample, by score quantile so the
            points are spread evenly over the data rather than over the
            score range, which can be badly skewed.

    Returns:
        One row per threshold, ordered from full coverage downward.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape != labels.shape:
        raise AbstentionError(f"shape mismatch: {scores.shape} vs {labels.shape}")
    if scores.size == 0:
        raise AbstentionError("no rows")

    quantiles = np.linspace(0.0, 0.95, n_points)
    thresholds = np.quantile(scores, quantiles)

    base_rate = float((labels == 0).mean())
    rows = []
    for q, threshold in zip(quantiles, thresholds):
        asserted = scores >= threshold
        n_asserted = int(asserted.sum())
        if n_asserted == 0:
            continue
        false_asserted = int((labels[asserted] == 0).sum())
        rows.append(
            {
                "quantile": float(q),
                "threshold": float(threshold),
                "coverage": n_asserted / scores.size,
                "abstention_rate": 1.0 - n_asserted / scores.size,
                "false_assertion_rate": false_asserted / n_asserted,
                "baseline_false_rate": base_rate,
                # How much of the original error the filter removed.
                "risk_reduction": (
                    (base_rate - false_asserted / n_asserted) / base_rate
                    if base_rate > 0 else 0.0
                ),
                "n_asserted": n_asserted,
            }
        )
    if not rows:
        raise AbstentionError("every threshold abstained from everything")
    return rows


def area_under_coverage_risk(rows: Sequence[dict[str, float]]) -> float:
    """Mean false-assertion rate, weighted by coverage interval.

    One number for the whole curve, so two configurations can be compared
    without eyeballing plots. Lower is better. A perfect detector approaches
    0; a useless one sits at the base error rate across the whole sweep.
    """
    ordered = sorted(rows, key=lambda r: r["coverage"])
    if len(ordered) < 2:
        raise AbstentionError("need at least two points to integrate")
    cov = np.array([r["coverage"] for r in ordered])
    risk = np.array([r["false_assertion_rate"] for r in ordered])
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0]))


def operating_point(
    rows: Sequence[dict[str, float]], target_coverage: float
) -> dict[str, float]:
    """The curve point closest to a chosen coverage, e.g. 'answer 70% of the time'.

    Deployment picks a coverage budget, not a threshold, so this is the form
    the result should be quoted in.
    """
    if not rows:
        raise AbstentionError("no curve")
    return min(rows, key=lambda r: abs(r["coverage"] - target_coverage))


def evaluate_with_abstention(
    features: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[str, np.ndarray, np.ndarray]],
    seed: int,
    target_coverages: Sequence[float] = (0.9, 0.7, 0.5),
    C: float = 1.0,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """Fit a probe per fold and report the pooled coverage-risk curve.

    Test-fold scores are pooled across folds before the sweep rather than
    averaging per-fold curves, because each fold holds out a different topic
    and its score distribution has a different location. Averaging curves
    computed at per-fold thresholds would compare points that do not
    correspond to the same decision rule.
    """
    from src.analysis.transfer import fit_transferable_probe

    all_scores, all_labels, all_folds = [], [], []
    for name, train_idx, test_idx in folds:
        probe = fit_transferable_probe(
            features[train_idx], labels[train_idx], seed=seed, C=C, max_iter=max_iter
        )
        all_scores.append(probe.score(features[test_idx]))
        all_labels.append(labels[test_idx])
        all_folds.extend([name] * len(test_idx))

    scores = np.concatenate(all_scores)
    pooled_labels = np.concatenate(all_labels)

    curve = coverage_risk_curve(scores, pooled_labels)
    return {
        "curve": curve,
        "auc_coverage_risk": area_under_coverage_risk(curve),
        "baseline_false_rate": float((pooled_labels == 0).mean()),
        "operating_points": {
            f"coverage_{int(c * 100)}": operating_point(curve, c)
            for c in target_coverages
        },
        "n": int(scores.size),
    }
