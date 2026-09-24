"""Dataset loaders -> the unified schema.

Dataset IDs and file paths are never guessed. Every id below was found
by searching the Hub, and every field name and label convention was read off
real rows before this file was written. What was verified, on 2026-09-02:

``pminervini/true-false``  (Azaria & Mitchell)
    Loads as a DatasetDict whose *splits are the topics*: animals, cities,
    companies, elements, facts, inventions, plus `cieacf` and `generated`.
    Columns: `statement`, `label`. 1 = TRUE.
    The six topic splits total 6085 rows.
    `cieacf` is EXACTLY the concatenation of those six (1008 + 1200 + 876 +
    1458 + 613 + 930 = 6085), so loading it as well would duplicate the whole
    dataset. We load the six individually because that is the only way to
    keep the topic label, and topic is our splitting variable.
    `generated` (245 rows) is a separate synthetic set, excluded by default.

``truthfulqa/truthful_qa``  config ``multiple_choice``
    817 questions in `validation`. `mc1_targets` is a dict of `choices` and
    `labels`, exactly one label 1 per question. 4114 choices total, positive
    rate 0.199. The `generation` config carries a `category` field (38
    values) keyed by the same `question` string - we join on it to get a
    topic, because `multiple_choice` has no topic of its own.

``pminervini/HaluEval``  config ``qa``
    10000 rows in a split literally named `data`. Columns: `knowledge`,
    `question`, `right_answer`, `hallucinated_answer`. There is NO label
    column - it is a pair format. Each row yields two records, so the result
    is balanced by construction.

Results are cached to JSONL under the cache root. Network access happens once.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from src.data.schema import Record, SchemaError, summarize, validate_records

logger = logging.getLogger("quantprobe.data.loaders")

# --- Verified Hub identifiers. Do not edit without re-running the checks in
#     scripts/data_report.py. ---
TRUE_FALSE_REPO = "pminervini/true-false"
TRUE_FALSE_TOPICS: tuple[str, ...] = (
    "animals",
    "cities",
    "companies",
    "elements",
    "facts",
    "inventions",
)
#: Splits present in the repo that we deliberately do NOT load.
#: `cieacf` duplicates the six topics; `generated` is a different distribution.
TRUE_FALSE_EXCLUDED: tuple[str, ...] = ("cieacf", "generated")

TRUTHFULQA_REPO = "truthfulqa/truthful_qa"
HALUEVAL_REPO = "pminervini/HaluEval"


class DataError(RuntimeError):
    """A dataset could not be loaded or failed its verification checks."""


# ---------------------------------------------------------------- true/false


def load_true_false(
    topics: tuple[str, ...] = TRUE_FALSE_TOPICS,
    drop_contradictions: bool = True,
    dedupe: bool = True,
) -> list[Record]:
    """Azaria & Mitchell true-false statements, one topic per split.

    Args:
        topics: which topic splits to load.
        drop_contradictions: remove statements whose identical text carries
            BOTH labels. There are 14 such statements, all in `cities`, all
            genuinely ambiguous entities - Gibraltar, Monaco, Djibouti,
            Vatican City and friends. Each is a city AND a
            country/territory, so the annotation is not wrong so much as
            underdetermined. Keeping them puts an irreducible noise floor on
            every probe; worse, that floor sits entirely in one topic, so it
            would distort exactly the held-out fold that topic is used for.
            Dropping 28 rows of 6085 (0.5%) is cheap. Toggleable, and the
            count is always logged.

            The count is 14 rather than 12 because statements are stripped
            before comparison: two `cities` rows differ only by trailing
            whitespace, and without the strip they read as distinct text.
        dedupe: collapse exact-duplicate statements that agree on their
            label (8 of them). Harmless either way; removed so that a row and
            its twin cannot land in different CV folds.

    Returns:
        Validated records with `topic` set to the source topic and `prompt`
        empty (these are standalone declaratives).
    """
    from datasets import load_dataset

    try:
        dsd = load_dataset(TRUE_FALSE_REPO)
    except Exception as exc:
        raise DataError(f"could not load {TRUE_FALSE_REPO}: {exc}") from exc

    available = set(dsd.keys())
    missing = [t for t in topics if t not in available]
    if missing:
        raise DataError(
            f"{TRUE_FALSE_REPO} has no splits {missing}; available: {sorted(available)}"
        )

    raw: list[tuple[str, str, int]] = []
    for topic in topics:
        split = dsd[topic]
        for column in ("statement", "label"):
            if column not in split.column_names:
                raise DataError(
                    f"{TRUE_FALSE_REPO}[{topic}] has no {column!r} column; "
                    f"columns are {split.column_names}"
                )
        for row in split:
            raw.append((topic, str(row["statement"]).strip(), int(row["label"])))

    logger.info("%s: %d raw rows across %d topics", TRUE_FALSE_REPO, len(raw), len(topics))

    kept = _resolve_duplicates(raw, drop_contradictions=drop_contradictions, dedupe=dedupe)

    records = [
        Record(
            id=f"true_false:{topic}:{i:05d}",
            prompt="",
            statement=statement,
            label=label,
            topic=topic,
            source="true_false",
        )
        for i, (topic, statement, label) in enumerate(kept)
    ]
    return validate_records(records, where=TRUE_FALSE_REPO)


def _resolve_duplicates(
    raw: list[tuple[str, str, int]],
    drop_contradictions: bool,
    dedupe: bool,
) -> list[tuple[str, str, int]]:
    """Handle exact-duplicate statements, loudly."""
    labels_by_text: dict[str, set[int]] = defaultdict(set)
    for _, statement, label in raw:
        labels_by_text[statement].add(label)

    contradictory = {s for s, labels in labels_by_text.items() if len(labels) > 1}
    if contradictory:
        logger.warning(
            "%d statements carry BOTH labels (ambiguous entities). drop=%s. examples: %s",
            len(contradictory),
            drop_contradictions,
            sorted(contradictory)[:3],
        )

    kept: list[tuple[str, str, int]] = []
    seen: set[tuple[str, int]] = set()
    n_dropped_contradictory = 0
    n_deduped = 0

    for topic, statement, label in raw:
        if drop_contradictions and statement in contradictory:
            n_dropped_contradictory += 1
            continue
        key = (statement, label)
        if dedupe and key in seen:
            n_deduped += 1
            continue
        seen.add(key)
        kept.append((topic, statement, label))

    logger.info(
        "duplicate handling: dropped %d contradictory rows, %d exact duplicates; %d -> %d rows",
        n_dropped_contradictory,
        n_deduped,
        len(raw),
        len(kept),
    )
    return kept


# ----------------------------------------------------------------- truthfulqa


def load_truthfulqa_mc1() -> list[Record]:
    """TruthfulQA MC1, one record per candidate answer.

    Each of the 817 questions becomes N records - one per choice - carrying
    that choice's MC1 label. Exactly one choice per question is true, so the
    pool is deliberately imbalanced (positive rate ~0.199). That is fine and
    intended: AUROC is invariant to class prevalence, and this project uses this
    set as an adversarial *generalization test*, never as probe training data.

    Topic comes from the `category` field of the `generation` config, joined
    on the question string. Without it every row would share one topic and
    group-aware splitting would silently degenerate to a random split.
    """
    from datasets import load_dataset

    try:
        mc = load_dataset(TRUTHFULQA_REPO, "multiple_choice")["validation"]
        gen = load_dataset(TRUTHFULQA_REPO, "generation")["validation"]
    except Exception as exc:
        raise DataError(f"could not load {TRUTHFULQA_REPO}: {exc}") from exc

    category_by_question: dict[str, str] = {
        str(row["question"]): str(row["category"]) for row in gen
    }

    unmatched = 0
    records: list[Record] = []
    for q_index, row in enumerate(mc):
        question = str(row["question"])
        targets = row["mc1_targets"]
        choices, labels = targets["choices"], targets["labels"]

        if len(choices) != len(labels):
            raise DataError(
                f"truthfulqa q{q_index}: {len(choices)} choices but {len(labels)} labels"
            )
        if sum(labels) != 1:
            raise DataError(
                f"truthfulqa q{q_index}: MC1 must have exactly one correct choice, "
                f"got {sum(labels)}"
            )

        category = category_by_question.get(question)
        if category is None:
            unmatched += 1
            category = "uncategorized"

        for c_index, (choice, label) in enumerate(zip(choices, labels)):
            records.append(
                Record(
                    id=f"truthfulqa_mc1:{q_index:04d}:{c_index:02d}",
                    prompt=question,
                    statement=str(choice).strip(),
                    label=int(label),
                    topic=_slug(category),
                    source="truthfulqa_mc1",
                    meta={"question_index": q_index},
                )
            )

    if unmatched:
        logger.warning(
            "%d TruthfulQA questions had no category match in the generation "
            "config; they were assigned topic 'uncategorized'",
            unmatched,
        )

    logger.info(
        "%s: %d questions -> %d choice records", TRUTHFULQA_REPO, len(mc), len(records)
    )
    return validate_records(records, where=TRUTHFULQA_REPO)


# ------------------------------------------------------------------ halueval


def load_halueval_qa(limit: int | None = None) -> list[Record]:
    """HaluEval QA, expanded from pairs into labelled records.

    The source has no label column: each row holds a `right_answer` and a
    `hallucinated_answer` for the same question. We emit both, labelled 1 and
    0, which makes the result exactly balanced by construction.

    `knowledge` is preserved in `meta` rather than discarded. E5 has to decide
    whether the behavioural check is open-book (model sees the passage) or
    closed-book (pure recall); dropping the passage here would make that
    decision by accident.

    Args:
        limit: keep only the first N source rows. 10000 rows -> 20000 records
            is far more than E5 needs and would dominate extraction cost.
    """
    from datasets import load_dataset

    try:
        dsd = load_dataset(HALUEVAL_REPO, "qa")
    except Exception as exc:
        raise DataError(f"could not load {HALUEVAL_REPO}: {exc}") from exc

    split_name = "data" if "data" in dsd else next(iter(dsd))
    split = dsd[split_name]

    expected = {"knowledge", "question", "right_answer", "hallucinated_answer"}
    missing = expected - set(split.column_names)
    if missing:
        raise DataError(
            f"{HALUEVAL_REPO}[qa] is missing columns {sorted(missing)}; "
            f"columns are {split.column_names}"
        )

    n_rows = len(split) if limit is None else min(limit, len(split))
    records: list[Record] = []
    for i in range(n_rows):
        row = split[i]
        question = _clean(str(row["question"]))
        knowledge = _clean(str(row["knowledge"]))
        for suffix, column, label in (
            ("t", "right_answer", 1),
            ("f", "hallucinated_answer", 0),
        ):
            records.append(
                Record(
                    id=f"halueval_qa:{i:05d}:{suffix}",
                    prompt=question,
                    statement=_clean(str(row[column])),
                    label=label,
                    # HaluEval QA is built on HotpotQA and carries no topic
                    # field. It is the E5 behavioural set, not probe training
                    # data, so it is never topic-split; a single constant
                    # topic is honest about that rather than inventing one.
                    topic="halueval_qa",
                    source="halueval_qa",
                    meta={"knowledge": knowledge, "row_index": i},
                )
            )

    logger.info(
        "%s[qa]: %d source rows -> %d records (balanced by construction)",
        HALUEVAL_REPO,
        n_rows,
        len(records),
    )
    return validate_records(records, where=HALUEVAL_REPO)


# -------------------------------------------------------------------- helpers


def _slug(text: str) -> str:
    """Lowercase, underscore-separated topic key."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_") or "unknown"


def _clean(text: str) -> str:
    """Normalise whitespace, and guard against genuine decode damage.

    A note, because this looked like a bug and is not. HaluEval rows such as
    "Arthur's Magazine (1844-1846)" render as "1844<?>1846" in a Windows
    console. That is the *console* failing to draw U+2013 EN DASH in cp1252,
    not corrupt data - checked by printing the codepoint, which is 0x2013.
    The text is fine and is left alone.

    The U+FFFD substitution below therefore never fires on today's data. It
    stays as a guard: a real replacement character would mean the bytes were
    already lost upstream, and it would tokenize to a byte-fallback sequence
    that shifts the last-token position for those rows only - a silent,
    row-specific misalignment of exactly the position we probe. If it ever
    fires, we want it normalised rather than quietly extracted.
    """
    if "�" in text:
        logger.warning("U+FFFD found in source text; replacing: %r", text[:80])
        text = text.replace("�", "-")
    return re.sub(r"\s+", " ", text).strip()


# -------------------------------------------------------------------- registry

LOADERS: dict[str, Callable[..., list[Record]]] = {
    "true_false": load_true_false,
    "truthfulqa_mc1": load_truthfulqa_mc1,
    "halueval_qa": load_halueval_qa,
}


def load_source(name: str, cache_dir: Path | None = None, **kwargs: Any) -> list[Record]:
    """Load one dataset by name, with a JSONL cache.

    Extraction is the expensive step, but re-downloading and re-parsing three
    datasets on every Colab reconnect is a needless several minutes. The cache
    key includes the loader kwargs so that flipping `drop_contradictions`
    does not silently serve the wrong rows.
    """
    if name not in LOADERS:
        raise DataError(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")

    cache_path: Path | None = None
    if cache_dir is not None:
        signature = "_".join(f"{k}-{v}" for k, v in sorted(kwargs.items())) or "default"
        cache_path = Path(cache_dir) / "datasets" / f"{name}__{signature}.jsonl"
        if cache_path.is_file():
            records = read_jsonl(cache_path)
            logger.info("loaded %d records for %r from cache %s", len(records), name, cache_path)
            return records

    records = LOADERS[name](**kwargs)

    if cache_path is not None:
        write_jsonl(cache_path, records)
        logger.info("cached %d records for %r to %s", len(records), name, cache_path)

    return records


def load_all(
    cache_dir: Path | None = None,
    halueval_limit: int | None = 1000,
) -> dict[str, list[Record]]:
    """Load all three datasets. Returns a mapping keyed by source name."""
    out: dict[str, list[Record]] = {
        "true_false": load_source("true_false", cache_dir),
        "truthfulqa_mc1": load_source("truthfulqa_mc1", cache_dir),
        "halueval_qa": load_source("halueval_qa", cache_dir, limit=halueval_limit),
    }
    for name, records in out.items():
        logger.info("%s: %s", name, summarize(records)["n"])
    return out


# ------------------------------------------------------------------- jsonl io


def write_jsonl(path: Path, records: list[Record]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(Record.from_dict(json.loads(line)))
            except (json.JSONDecodeError, SchemaError) as exc:
                raise DataError(f"{path}:{line_no}: {exc}") from exc
    return records
