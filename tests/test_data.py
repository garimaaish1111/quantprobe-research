"""Phase 1 tests: schema, duplicate handling, splitting, and the real datasets.

Two tiers.

**Offline** (the default): schema validation, duplicate resolution, and fold
construction, all on synthetic records. Fast, hermetic, and they are where the
actual logic lives.

**Network** (`-m network`): assertions against the real Hub datasets - row
counts, label conventions, no nulls. These are the checks that would catch a
dataset being silently revised upstream. They are skipped by default so that
`pytest` stays usable offline and on a plane.

    python -m pytest tests/ -q                 # offline only
    python -m pytest tests/ -q -m network      # the real datasets
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import loaders  # noqa: E402
from src.data.schema import (  # noqa: E402
    LABEL_FALSE,
    LABEL_TRUE,
    Record,
    SchemaError,
    label_balance,
    summarize,
    validate_records,
)
from src.data.splits import (  # noqa: E402
    SplitError,
    fold_summary,
    held_out_split,
    make_folds,
    verify_folds,
)


def make_record(i: int, topic: str = "cities", label: int = 1, **kw) -> Record:
    defaults = dict(
        id=f"src:{topic}:{i}",
        prompt="",
        statement=f"statement number {i}",
        label=label,
        topic=topic,
        source="src",
    )
    defaults.update(kw)
    return Record(**defaults)


def balanced_pool(topics=("a", "b", "c", "d", "e", "f"), per_topic: int = 10) -> list[Record]:
    """A pool with both classes present in every topic."""
    out = []
    for t in topics:
        for i in range(per_topic):
            out.append(make_record(i, topic=t, label=i % 2, id=f"src:{t}:{i}"))
    return out


# ------------------------------------------------------------------- schema


def test_text_omits_empty_prompt():
    rec = make_record(0, prompt="", statement="Paris is in France.")
    assert rec.text() == "Paris is in France."


def test_text_joins_prompt_and_statement():
    rec = make_record(0, prompt="Where is Paris?", statement="In France.")
    assert rec.text() == "Where is Paris? In France."


def test_text_has_no_leading_space_when_prompt_blank():
    """A stray leading space shifts every token position for that row only."""
    assert not make_record(0, prompt="").text().startswith(" ")


def test_round_trip_through_dict_preserves_everything():
    rec = make_record(3, prompt="q", meta={"knowledge": "k"})
    assert Record.from_dict(rec.to_dict()) == rec


def test_from_dict_rejects_missing_fields():
    with pytest.raises(SchemaError, match="missing required fields"):
        Record.from_dict({"id": "x", "statement": "s", "label": 1})


def test_validate_rejects_empty_statement():
    with pytest.raises(SchemaError, match="empty statement"):
        validate_records([make_record(0, statement="   ")])


def test_validate_rejects_out_of_range_label():
    with pytest.raises(SchemaError, match="expected 0 or 1"):
        validate_records([make_record(0, label=2)])


def test_validate_rejects_empty_topic():
    """An empty topic silently collapses group splitting into a random split."""
    with pytest.raises(SchemaError, match="empty topic"):
        validate_records([make_record(0, topic="")])


def test_validate_rejects_duplicate_ids():
    with pytest.raises(SchemaError, match="duplicate id"):
        validate_records([make_record(0), make_record(0)])


def test_validate_rejects_empty_input():
    with pytest.raises(SchemaError, match="no records"):
        validate_records([])


def test_label_balance_counts_per_topic():
    pool = [make_record(0, "x", 1), make_record(1, "x", 0), make_record(2, "y", 1)]
    balance = label_balance(pool)
    assert balance["x"] == {"true": 1, "false": 1, "n": 2}
    assert balance["y"] == {"true": 1, "false": 0, "n": 1}


def test_summarize_reports_positive_rate():
    stats = summarize(balanced_pool(("a",), per_topic=10))
    assert stats["n"] == 10
    assert stats["positive_rate"] == 0.5
    assert stats["n_topics"] == 1


def test_label_constants_are_the_verified_convention():
    """1 = TRUE was read off real rows of all three sources. Pin it."""
    assert (LABEL_TRUE, LABEL_FALSE) == (1, 0)


# ------------------------------------------------------- duplicate handling


def test_contradictory_statements_are_dropped_entirely():
    """Both copies must go - keeping either would assert a label we know is disputed."""
    raw = [
        ("cities", "Gibraltar is a name of a city.", 1),
        ("cities", "Gibraltar is a name of a city.", 0),
        ("cities", "Paris is a name of a city.", 1),
    ]
    kept = loaders._resolve_duplicates(raw, drop_contradictions=True, dedupe=True)
    assert [s for _, s, _ in kept] == ["Paris is a name of a city."]


def test_contradictions_can_be_retained():
    raw = [
        ("cities", "Gibraltar is a name of a city.", 1),
        ("cities", "Gibraltar is a name of a city.", 0),
    ]
    kept = loaders._resolve_duplicates(raw, drop_contradictions=False, dedupe=True)
    assert len(kept) == 2


def test_agreeing_duplicates_are_collapsed():
    raw = [("cities", "Paris is a city.", 1)] * 3
    kept = loaders._resolve_duplicates(raw, drop_contradictions=True, dedupe=True)
    assert len(kept) == 1


def test_dedupe_off_keeps_agreeing_duplicates():
    raw = [("cities", "Paris is a city.", 1)] * 3
    kept = loaders._resolve_duplicates(raw, drop_contradictions=True, dedupe=False)
    assert len(kept) == 3


# ------------------------------------------------------------------- splits


def test_leave_one_topic_out_gives_one_fold_per_topic():
    pool = balanced_pool()
    folds = make_folds(pool, scheme="leave_one_topic_out")
    assert len(folds) == 6
    assert [name for name, _, _ in folds] == ["a", "b", "c", "d", "e", "f"]


def test_leave_one_topic_out_holds_out_exactly_that_topic():
    pool = balanced_pool()
    topics = np.array([r.topic for r in pool])
    for name, train_idx, test_idx in make_folds(pool, scheme="leave_one_topic_out"):
        assert set(topics[test_idx]) == {name}
        assert name not in set(topics[train_idx])


def test_every_row_is_tested_exactly_once_across_folds():
    pool = balanced_pool()
    seen = np.concatenate([test for _, _, test in make_folds(pool, "leave_one_topic_out")])
    assert sorted(seen.tolist()) == list(range(len(pool)))


def test_group_kfold_packs_six_topics_into_five_folds():
    pool = balanced_pool()
    folds = make_folds(pool, scheme="group_kfold", n_splits=5)
    assert len(folds) == 5
    assert any("+" in name for name, _, _ in folds), "one fold must hold two topics"


def test_group_kfold_rejects_more_splits_than_topics():
    pool = balanced_pool(("a", "b"), per_topic=10)
    with pytest.raises(SplitError, match="n_splits"):
        make_folds(pool, scheme="group_kfold", n_splits=5)


def test_unknown_scheme_is_rejected():
    with pytest.raises(SplitError, match="unknown split scheme"):
        make_folds(balanced_pool(), scheme="random_80_20")


def test_verify_folds_catches_a_topic_on_both_sides():
    """The guard against the classic inflated-AUROC bug.

    A random split is constructed deliberately; verification must reject it.
    """
    pool = balanced_pool()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pool))
    bad = [("random", perm[: len(pool) // 2], perm[len(pool) // 2 :])]
    with pytest.raises(SplitError, match="topics on both sides"):
        verify_folds(pool, bad)


def test_verify_folds_catches_overlapping_rows():
    pool = balanced_pool()
    idx = np.arange(len(pool))
    with pytest.raises(SplitError, match="BOTH sides"):
        verify_folds(pool, [("bad", idx, idx)])


def test_verify_folds_catches_single_class_test_side():
    """AUROC is undefined on a one-class fold; catch it here, not in sklearn."""
    pool = [make_record(i, "a", 1) for i in range(5)] + [
        make_record(i + 5, "b", i % 2) for i in range(5)
    ]
    with pytest.raises(SplitError, match="only class"):
        make_folds(pool, scheme="leave_one_topic_out")


def test_verify_folds_rejects_empty_fold_list():
    with pytest.raises(SplitError, match="no folds"):
        verify_folds(balanced_pool(), [])


def test_held_out_split_selects_one_topic():
    pool = balanced_pool()
    train_idx, test_idx = held_out_split(pool, "c")
    topics = np.array([r.topic for r in pool])
    assert set(topics[test_idx]) == {"c"}
    assert "c" not in set(topics[train_idx])


def test_held_out_split_rejects_unknown_topic():
    with pytest.raises(SplitError, match="not present"):
        held_out_split(balanced_pool(), "nonexistent")


def test_fold_summary_shape():
    pool = balanced_pool()
    rows = fold_summary(pool, make_folds(pool, "leave_one_topic_out"))
    assert len(rows) == 6
    assert set(rows[0]) == {"held_out", "n_train", "n_test", "train_positive_rate", "test_positive_rate"}


# -------------------------------------------------------------------- jsonl


def test_jsonl_round_trip(tmp_path):
    pool = balanced_pool(("a", "b"), per_topic=3)
    path = loaders.write_jsonl(tmp_path / "d.jsonl", pool)
    assert loaders.read_jsonl(path) == pool


def test_jsonl_preserves_meta(tmp_path):
    pool = [make_record(0, meta={"knowledge": "k", "row_index": 7})]
    path = loaders.write_jsonl(tmp_path / "d.jsonl", pool)
    assert loaders.read_jsonl(path)[0].meta == {"knowledge": "k", "row_index": 7}


def test_read_jsonl_reports_the_bad_line_number(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"id":"a","prompt":"","statement":"s","label":1,"topic":"t","source":"x"}\n{oops\n')
    with pytest.raises(loaders.DataError, match=":2:"):
        loaders.read_jsonl(path)


def test_load_source_rejects_unknown_dataset():
    with pytest.raises(loaders.DataError, match="unknown dataset"):
        loaders.load_source("not_a_dataset")


def test_clean_normalises_whitespace():
    assert loaders._clean("  a\n b \t c ") == "a b c"


def test_clean_leaves_real_unicode_alone():
    """U+2013 is legitimate data, not mojibake - see the note in _clean."""
    assert loaders._clean("Arthur's Magazine (1844-1846)") == "Arthur's Magazine (1844-1846)"


def test_slug_makes_safe_topic_keys():
    assert loaders._slug("Confusion: people") == "confusion_people"
    assert loaders._slug("  Myths & Fairytales ") == "myths_fairytales"


# ------------------------------------------------------------------ network

pytestmark_network = pytest.mark.network


@pytest.mark.network
def test_true_false_matches_verified_counts():
    """Row counts and balance verified on 2026-09-02. Catches upstream revisions."""
    records = loaders.load_true_false()
    stats = summarize(records)
    assert stats["n"] == 6049, f"expected 6049 rows after cleaning, got {stats['n']}"
    assert stats["n_topics"] == 6
    assert set(stats["topics"]) == set(loaders.TRUE_FALSE_TOPICS)
    assert 0.49 < stats["positive_rate"] < 0.51, "true/false pool should be near-balanced"


@pytest.mark.network
def test_true_false_label_convention_is_one_equals_true():
    """The single assumption everything else rests on."""
    records = loaders.load_true_false(topics=("cities",))
    by_text = {r.statement: r.label for r in records}
    assert by_text["Thimphu is a name of a city."] == LABEL_TRUE
    assert by_text["Praia is a name of a country."] == LABEL_FALSE


@pytest.mark.network
def test_true_false_has_no_cross_topic_duplicates():
    """If this fails, topic splits stop isolating and AUROC inflates."""
    records = loaders.load_true_false()
    topics_by_text: dict[str, set[str]] = {}
    for rec in records:
        topics_by_text.setdefault(rec.statement, set()).add(rec.topic)
    offenders = {t: ts for t, ts in topics_by_text.items() if len(ts) > 1}
    assert not offenders, f"statements spanning topics: {list(offenders)[:5]}"


@pytest.mark.network
def test_truthfulqa_mc1_has_exactly_one_true_per_question():
    records = loaders.load_truthfulqa_mc1()
    assert len(records) == 4114
    per_question: dict[int, int] = {}
    for rec in records:
        per_question[rec.meta["question_index"]] = per_question.get(
            rec.meta["question_index"], 0
        ) + rec.label
    assert set(per_question.values()) == {1}
    assert len(per_question) == 817


@pytest.mark.network
def test_halueval_is_balanced_by_construction():
    records = loaders.load_halueval_qa(limit=50)
    stats = summarize(records)
    assert stats["n"] == 100
    assert stats["n_true"] == stats["n_false"] == 50


@pytest.mark.network
def test_no_nulls_or_empties_in_any_dataset():
    """Phase 1 done-signal: no nulls."""
    for name, kwargs in (
        ("true_false", {}),
        ("truthfulqa_mc1", {}),
        ("halueval_qa", {"limit": 50}),
    ):
        records = loaders.LOADERS[name](**kwargs)
        for rec in records:
            assert rec.statement.strip(), f"{name}: empty statement at {rec.id}"
            assert rec.topic.strip(), f"{name}: empty topic at {rec.id}"
            assert rec.label in (0, 1), f"{name}: bad label at {rec.id}"
            assert rec.text().strip(), f"{name}: empty text() at {rec.id}"
