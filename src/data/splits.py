"""Topic-based splitting.

Split by topic, not randomly: train on 5 topics, test on the held-out one.
Random splits leak near-duplicate statements and inflate AUROC by a lot -
this is a known failure in probe papers.

Why that is true here, concretely. The `cities` topic contains 1458 rows built
from a small template pool - "Thimphu is a name of a city.", "Praia is a name
of a country." A random 80/20 split puts near-identical sentences on both
sides. The probe then has to distinguish "Thimphu" from "Praia" having already
seen sentences of that exact shape with their answers, which is memorisation
of the template, not detection of a truthfulness signal. Reported AUROC goes
up and means less.

A note on two readings of the protocol: "train on 5 topics, test on the
held-out one" and "5-fold CV over topic groups". With six topics those are
different procedures. Leave-one-topic-out gives six folds and
is exactly the first reading; GroupKFold(5) packs six topics into five folds,
so one fold holds two topics and the folds are unequal. We default to
leave-one-topic-out because it matches that reading literally, gives one more estimate
for the mean +/- std, and keeps every fold interpretable ("this is the
held-out `elements` number"). GroupKFold is implemented and selectable so the
alternative is one config change away, not a rewrite.
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

import numpy as np

from src.data.schema import Record

logger = logging.getLogger("quantprobe.data.splits")

SCHEMES = ("leave_one_topic_out", "group_kfold")


class SplitError(ValueError):
    """A split is invalid - overlapping topics, empty fold, or one class."""


def topics_of(records: Sequence[Record]) -> list[str]:
    """Sorted unique topics, for stable fold ordering across runs."""
    return sorted({rec.topic for rec in records})


def leave_one_topic_out(records: Sequence[Record]) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_topic, train_idx, test_idx) for each topic.

    Six topics gives six folds. Fold order follows sorted topic names so a
    rerun produces folds in the same order and fold i always means the same
    topic.
    """
    groups = np.array([rec.topic for rec in records])
    for topic in topics_of(records):
        test_mask = groups == topic
        test_idx = np.flatnonzero(test_mask)
        train_idx = np.flatnonzero(~test_mask)
        yield topic, train_idx, test_idx


def group_kfold(
    records: Sequence[Record], n_splits: int = 5
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """GroupKFold over topics. Kept for the 5-fold reading of the protocol.

    sklearn's GroupKFold is deterministic and takes no random_state - it
    assigns groups to folds greedily by size. With 6 topics and n_splits=5 one
    fold necessarily holds two topics.
    """
    from sklearn.model_selection import GroupKFold

    groups = np.array([rec.topic for rec in records])
    n_groups = len(set(groups))
    if n_splits > n_groups:
        raise SplitError(
            f"group_kfold needs n_splits <= number of topics; got n_splits={n_splits} "
            f"with {n_groups} topics ({sorted(set(groups))})"
        )

    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(records), 1))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy_x, groups=groups)):
        held_out = sorted(set(groups[test_idx]))
        yield "+".join(held_out) if len(held_out) > 1 else held_out[0], train_idx, test_idx
        del fold


def make_folds(
    records: Sequence[Record],
    scheme: str = "leave_one_topic_out",
    n_splits: int = 5,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build folds and verify every invariant before returning them.

    Raises:
        SplitError: unknown scheme, a topic appearing on both sides of a
            fold, an empty fold, or a fold whose test side has only one class
            (AUROC is undefined there, and sklearn would raise a confusing
            error much later).
    """
    if scheme not in SCHEMES:
        raise SplitError(f"unknown split scheme {scheme!r}; expected one of {list(SCHEMES)}")

    if scheme == "leave_one_topic_out":
        folds = list(leave_one_topic_out(records))
    else:
        folds = list(group_kfold(records, n_splits=n_splits))

    verify_folds(records, folds)
    logger.info(
        "built %d folds with scheme=%r over topics %s",
        len(folds),
        scheme,
        topics_of(records),
    )
    return folds


def verify_folds(
    records: Sequence[Record],
    folds: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """Assert the properties that make a topic split actually a topic split.

    This is the guard that would have caught the classic probe-paper bug. If
    it ever passes silently while topics overlap, every AUROC in the paper is
    inflated and nothing else in the pipeline would notice.
    """
    if not folds:
        raise SplitError("no folds were produced")

    topics = np.array([rec.topic for rec in records])
    labels = np.array([rec.label for rec in records])
    problems: list[str] = []

    for name, train_idx, test_idx in folds:
        if train_idx.size == 0 or test_idx.size == 0:
            problems.append(f"fold {name!r}: empty side (train={train_idx.size}, test={test_idx.size})")
            continue

        overlap_rows = np.intersect1d(train_idx, test_idx)
        if overlap_rows.size:
            problems.append(f"fold {name!r}: {overlap_rows.size} rows appear in BOTH sides")

        train_topics = set(topics[train_idx])
        test_topics = set(topics[test_idx])
        shared = train_topics & test_topics
        if shared:
            problems.append(f"fold {name!r}: topics on both sides: {sorted(shared)}")

        for side, idx in (("train", train_idx), ("test", test_idx)):
            present = set(labels[idx])
            if len(present) < 2:
                problems.append(
                    f"fold {name!r}: {side} side has only class {present} - AUROC undefined"
                )

    if problems:
        raise SplitError("fold verification failed:\n  " + "\n  ".join(problems))


def held_out_split(
    records: Sequence[Record], test_topic: str
) -> tuple[np.ndarray, np.ndarray]:
    """A single train/test split holding out one named topic."""
    available = topics_of(records)
    if test_topic not in available:
        raise SplitError(f"topic {test_topic!r} not present; available: {available}")
    groups = np.array([rec.topic for rec in records])
    test_mask = groups == test_topic
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def fold_summary(
    records: Sequence[Record],
    folds: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> list[dict]:
    """Per-fold row counts and label balance, for the report and the sidecar."""
    labels = np.array([rec.label for rec in records])
    rows = []
    for name, train_idx, test_idx in folds:
        rows.append(
            {
                "held_out": name,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "train_positive_rate": round(float(labels[train_idx].mean()), 4),
                "test_positive_rate": round(float(labels[test_idx].mean()), 4),
            }
        )
    return rows
