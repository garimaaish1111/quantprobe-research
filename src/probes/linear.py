"""Stage B - linear probes on cached hidden states.

Logistic regression, standardized inputs, L2, fixed seed,
one probe per layer, AUROC primary, CV over topic groups, mean +/- std. Plus a
mass-mean probe as a second, simpler estimator.

Three things here are easy to get wrong in ways that inflate AUROC silently.

**1. The scaler is fit on the training fold only.**
`StandardScaler().fit_transform(X)` over the whole dataset leaks the test
fold's mean and variance into training. With 2048 features and ~1000 test
rows that is a real effect, and it produces a number that is too good with no
error and no warning. Every fit here scales inside the fold.

**2. Rows are aligned by id, not by position.**
The cached array and the dataset are both in "load order", so indexing by
position works right up until it doesn't - a changed `drop_contradictions`
flag, a re-cached dataset, a different `--limit`. Then labels silently belong
to different statements and the probe scores noise at chance, or worse,
scores something plausible. `align_records()` asserts id-for-id equality.

**3. AUROC, not accuracy, is the primary metric.**
Accuracy depends on a threshold and on class balance; AUROC does not. The
true/false pool is near-balanced so they mostly agree, but TruthfulQA MC1 sits
at a positive rate of 0.199 and accuracy there would be dominated by the
majority class. Both are reported; AUROC is the one that gets plotted.

**Why the mass-mean probe too.** It has no hyperparameters, no solver, and no
capacity to overfit: the direction is just `mean(h | true) - mean(h | false)`.
If logistic regression scores much higher than mass-mean, the extra points
come from fitting structure that a single difference-of-means cannot express -
which might be real signal, or might be the probe memorising the topic. When
the two disagree strongly, that disagreement is a finding, not a nuisance.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("quantprobe.probes.linear")

PROBE_TYPES = ("logistic", "mass_mean")


class ProbeError(RuntimeError):
    """A probe cannot be fit, or its inputs are inconsistent."""


@dataclass
class FoldResult:
    """One probe, one layer, one fold."""

    precision: str
    layer: int
    probe_type: str
    fold: str
    auroc: float
    accuracy: float
    n_train: int
    n_test: int
    train_positive_rate: float
    test_positive_rate: float
    converged: bool = True
    weights: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if k != "weights"
        }


# ------------------------------------------------------------------ alignment


def align_records(cached_ids: Sequence[str], records: Sequence) -> np.ndarray:
    """Map cached row order onto the dataset, by id.

    Returns an index array `idx` such that `records[idx[i]]` is the record for
    cached row `i`.

    Raises:
        ProbeError: any id missing, duplicated, or extra. Positional
            assumptions are exactly what this exists to stop.
    """
    by_id: dict[str, int] = {}
    for i, rec in enumerate(records):
        if rec.id in by_id:
            raise ProbeError(f"duplicate record id in dataset: {rec.id!r}")
        by_id[rec.id] = i

    if len(cached_ids) != len(records):
        raise ProbeError(
            f"cached extraction has {len(cached_ids)} rows but the dataset has "
            f"{len(records)}. These are not the same run - re-extract, or load "
            f"the dataset with the same options the extraction used."
        )

    missing = [cid for cid in cached_ids if cid not in by_id]
    if missing:
        raise ProbeError(
            f"{len(missing)} cached ids are absent from the dataset "
            f"(first: {missing[:3]}). The dataset changed under the cache."
        )

    return np.array([by_id[cid] for cid in cached_ids], dtype=np.int64)


# --------------------------------------------------------------------- probes


def _standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean unit-variance, fit on TRAIN ONLY.

    Constant features (variance 0) would divide by zero; their scale is set
    to 1 so they pass through as zeros rather than becoming NaN and poisoning
    the whole fit.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    C: float = 1.0,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """L2 logistic regression. Returns metrics plus the weight vector.

    The weight vector is kept because E4 asks whether the probe *direction*
    rotates under quantization - the angle between the FP16 and INT4 weight
    vectors at the same layer. Discarding it here would mean re-fitting every
    probe later.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    x_train, x_test = _standardize(x_train, x_test)

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        # `penalty="l2"` is the default and is deprecated as an explicit
        # argument in sklearn 1.8 (removed in 1.10). Omitted rather than
        # switched to l1_ratio=0, which is the same thing spelled worse.
        solver="lbfgs",
        random_state=seed,  # sklearn does NOT read the global numpy seed
    )

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False

    scores = model.decision_function(x_test)
    return {
        "auroc": float(roc_auc_score(y_test, scores)),
        "accuracy": float(accuracy_score(y_test, (scores > 0).astype(int))),
        "converged": converged,
        "weights": model.coef_.ravel().astype(np.float32),
    }


def fit_mass_mean(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Difference-of-means direction. No solver, no hyperparameters.

    direction = mean(x | y=1) - mean(x | y=0); score = x . direction.

    This cannot overfit in the usual sense - it has exactly as many free
    parameters as it has dimensions and fits them in closed form from class
    means. That makes it a useful floor: if logistic regression beats it by a
    lot, the difference is structure a single direction cannot express.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    x_train, x_test = _standardize(x_train, x_test)

    pos = x_train[y_train == 1]
    neg = x_train[y_train == 0]
    if pos.size == 0 or neg.size == 0:
        raise ProbeError("mass-mean probe needs both classes in the training fold")

    direction = pos.mean(axis=0) - neg.mean(axis=0)
    scores = x_test @ direction

    # Threshold at the midpoint of the projected class means, computed on
    # TRAIN. Using the test scores' own median would be leakage.
    threshold = ((pos.mean(axis=0) + neg.mean(axis=0)) / 2.0) @ direction

    return {
        "auroc": float(roc_auc_score(y_test, scores)),
        "accuracy": float(accuracy_score(y_test, (scores > threshold).astype(int))),
        "converged": True,
        "weights": direction.astype(np.float32),
    }


# ----------------------------------------------------------------- driver


def probe_layer(
    features: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[str, np.ndarray, np.ndarray]],
    precision: str,
    layer: int,
    seed: int,
    probe_types: Sequence[str] = PROBE_TYPES,
    C: float = 1.0,
    max_iter: int = 2000,
) -> list[FoldResult]:
    """Fit every probe type on every fold for one layer.

    Args:
        features: [N, d_model] for this layer.
        labels: [N] in {0, 1}.
        folds: (name, train_idx, test_idx) from src.data.splits.
    """
    if features.ndim != 2:
        raise ProbeError(f"expected [N, d_model], got shape {features.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ProbeError(
            f"features has {features.shape[0]} rows, labels has {labels.shape[0]}"
        )

    results: list[FoldResult] = []
    for name, train_idx, test_idx in folds:
        x_train, y_train = features[train_idx], labels[train_idx]
        x_test, y_test = features[test_idx], labels[test_idx]

        for probe_type in probe_types:
            if probe_type == "logistic":
                out = fit_logistic(x_train, y_train, x_test, y_test, seed, C, max_iter)
            elif probe_type == "mass_mean":
                out = fit_mass_mean(x_train, y_train, x_test, y_test)
            else:
                raise ProbeError(f"unknown probe type {probe_type!r}")

            results.append(
                FoldResult(
                    precision=precision,
                    layer=layer,
                    probe_type=probe_type,
                    fold=name,
                    auroc=out["auroc"],
                    accuracy=out["accuracy"],
                    n_train=int(train_idx.size),
                    n_test=int(test_idx.size),
                    train_positive_rate=round(float(y_train.mean()), 4),
                    test_positive_rate=round(float(y_test.mean()), 4),
                    converged=out["converged"],
                    weights=out["weights"],
                )
            )
    return results


def aggregate(results: Sequence[FoldResult]) -> list[dict[str, Any]]:
    """Collapse fold results to mean +/- std per (precision, probe_type, layer).

    Report mean +/- std, not a single number. A single number from
    one fold hides that `cities` and `elements` can differ by more than the
    precision effect we are trying to measure.
    """
    buckets: dict[tuple[str, str, int], list[FoldResult]] = {}
    for r in results:
        buckets.setdefault((r.precision, r.probe_type, r.layer), []).append(r)

    rows: list[dict[str, Any]] = []
    for (precision, probe_type, layer), group in sorted(buckets.items()):
        aurocs = np.array([g.auroc for g in group], dtype=np.float64)
        accs = np.array([g.accuracy for g in group], dtype=np.float64)
        rows.append(
            {
                "precision": precision,
                "probe_type": probe_type,
                "layer": layer,
                "auroc_mean": float(aurocs.mean()),
                "auroc_std": float(aurocs.std(ddof=0)),
                "auroc_min": float(aurocs.min()),
                "auroc_max": float(aurocs.max()),
                "accuracy_mean": float(accs.mean()),
                "accuracy_std": float(accs.std(ddof=0)),
                "n_folds": len(group),
                "n_not_converged": sum(1 for g in group if not g.converged),
                "per_fold": {g.fold: round(g.auroc, 4) for g in group},
            }
        )
    return rows


def best_layer(rows: Sequence[dict[str, Any]], precision: str, probe_type: str) -> dict[str, Any]:
    """The layer with the highest mean AUROC for one (precision, probe_type)."""
    candidates = [
        r for r in rows if r["precision"] == precision and r["probe_type"] == probe_type
    ]
    if not candidates:
        raise ProbeError(f"no results for precision={precision!r} probe={probe_type!r}")
    return max(candidates, key=lambda r: r["auroc_mean"])


def mean_direction(results: Sequence[FoldResult], precision: str, probe_type: str, layer: int):
    """Average probe weight vector across folds, L2-normalised.

    E4 compares the FP16 and INT4 probe *directions*. Averaging across folds
    first gives a more stable estimate than picking one fold arbitrarily, and
    normalising means the later comparison is a pure angle - a probe that got
    noisier without rotating should not register as rotation.
    """
    vectors = [
        r.weights
        for r in results
        if r.precision == precision and r.probe_type == probe_type and r.layer == layer
        and r.weights is not None
    ]
    if not vectors:
        raise ProbeError(f"no weight vectors for {precision}/{probe_type}/layer {layer}")
    mean_vec = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else mean_vec


def leakage_warnings(rows: Sequence[dict[str, Any]], threshold: float = 0.98) -> list[str]:
    """Flag results that are too good to be true.

    An AUROC of 0.99 on a topic-held-out split is far more
    likely to be leakage than a real result, and the honest move is to say so
    before anyone puts it in a figure.
    """
    flags = []
    for r in rows:
        if r["auroc_mean"] >= threshold:
            flags.append(
                f"{r['precision']}/{r['probe_type']} layer {r['layer']}: "
                f"AUROC {r['auroc_mean']:.4f} >= {threshold}. On a topic-held-out "
                f"split this is more likely leakage than signal - check that the "
                f"folds really isolate topics and that labels are aligned by id."
            )
    return flags
