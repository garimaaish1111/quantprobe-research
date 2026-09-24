"""The unified record schema.

Every dataset is mapped onto one schema:

    { "id": str, "prompt": str, "statement": str,
      "label": int, "topic": str, "source": str }

All three datasets are coerced into it so that Stage A never needs to know
which dataset a row came from.

Field meanings, made precise because two of them are easy to get subtly wrong:

``prompt``
    Context shown to the model *before* the claim. Empty string for
    standalone declaratives (Azaria & Mitchell). The question, for the
    QA-shaped datasets.

``statement``
    The claim whose truth ``label`` describes. This is the span whose last
    token we read the residual stream at.

``label``
    **1 = the statement is TRUE, 0 = FALSE.** Verified empirically against
    real rows of all three sources, not assumed.

``topic``
    The grouping variable for splitting. Splits are by topic, never random.

``source``
    Which dataset produced the row. Lets a mixed pool be filtered later.

One documented extension to the core schema: an optional ``meta`` dict for
source-specific fields that would otherwise be destroyed. Today it carries
HaluEval's ``knowledge`` passage, which E5 may or may not want to condition
on; throwing it away at load time would quietly foreclose that design choice.
The six required keys are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: The six required keys. `meta` is an optional seventh.
REQUIRED_FIELDS: tuple[str, ...] = ("id", "prompt", "statement", "label", "topic", "source")

#: Label convention, verified against real rows. Referenced by tests so that a
#: future dataset swap that flips the convention fails loudly.
LABEL_TRUE: int = 1
LABEL_FALSE: int = 0


class SchemaError(ValueError):
    """A record violates the unified schema. Fails loudly."""


@dataclass(frozen=True, slots=True)
class Record:
    """One labelled statement."""

    id: str
    prompt: str
    statement: str
    label: int
    topic: str
    source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def text(self, template: str = "{prompt} {statement}") -> str:
        """The string actually fed to the tokenizer.

        For a standalone statement (``prompt == ""``) this is just the
        statement, with no stray leading space. For a QA row it is the
        question followed by the candidate answer, so that the last token of
        the sequence is the last token of the *claim* - which is the position
        the whole probing literature reads.
        """
        if not self.prompt:
            return self.statement.strip()
        return template.format(prompt=self.prompt.strip(), statement=self.statement.strip()).strip()

    def to_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in REQUIRED_FIELDS}
        if self.meta:
            out["meta"] = self.meta
        return out

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Record":
        missing = [k for k in REQUIRED_FIELDS if k not in row]
        if missing:
            raise SchemaError(f"row is missing required fields {missing}: {row!r}")
        return cls(
            id=str(row["id"]),
            prompt=str(row["prompt"]),
            statement=str(row["statement"]),
            label=int(row["label"]),
            topic=str(row["topic"]),
            source=str(row["source"]),
            meta=dict(row.get("meta", {})),
        )


def validate_records(records: Iterable[Record], where: str = "<records>") -> list[Record]:
    """Assert the invariants every downstream stage relies on.

    Checks, in the order they would bite:

    1. Non-empty statement. An empty string still tokenizes, to a single BOS
       token, and would silently contribute a meaningless feature vector.
    2. Label in {0, 1}. A stray -1 or None would be cast to a float by
       sklearn and turn the probe into a regression.
    3. Non-empty topic. A missing topic collapses group-aware splitting back
       into a random split, which is exactly the leakage topic splits exist to prevent, and it
       would do so without raising anything.
    4. Unique ids. Duplicate ids make a cached activation array
       unattributable to a row.

    Returns the records unchanged so this can be used inline.
    """
    records = list(records)
    if not records:
        raise SchemaError(f"{where}: no records loaded")

    problems: list[str] = []
    seen_ids: set[str] = set()

    for i, rec in enumerate(records):
        if not isinstance(rec, Record):
            problems.append(f"[{i}] not a Record: {type(rec).__name__}")
            continue
        if not rec.statement or not rec.statement.strip():
            problems.append(f"[{i}] id={rec.id!r} has an empty statement")
        if rec.label not in (LABEL_FALSE, LABEL_TRUE):
            problems.append(f"[{i}] id={rec.id!r} has label={rec.label!r}, expected 0 or 1")
        if not rec.topic or not rec.topic.strip():
            problems.append(f"[{i}] id={rec.id!r} has an empty topic")
        if not rec.source:
            problems.append(f"[{i}] id={rec.id!r} has an empty source")
        if rec.id in seen_ids:
            problems.append(f"[{i}] duplicate id {rec.id!r}")
        seen_ids.add(rec.id)

        if len(problems) > 20:
            problems.append("... further problems suppressed")
            break

    if problems:
        raise SchemaError(f"{where}: {len(problems)} schema violations:\n  " + "\n  ".join(problems))

    return records


def label_balance(records: Iterable[Record]) -> dict[str, dict[str, int]]:
    """Per-topic label counts. Printed by scripts/data_report.py."""
    table: dict[str, dict[str, int]] = {}
    for rec in records:
        row = table.setdefault(rec.topic, {"true": 0, "false": 0, "n": 0})
        row["true" if rec.label == LABEL_TRUE else "false"] += 1
        row["n"] += 1
    return dict(sorted(table.items()))


def summarize(records: Iterable[Record]) -> dict[str, Any]:
    """Compact summary for logging and for the JSON sidecar."""
    records = list(records)
    balance = label_balance(records)
    n_true = sum(r.label == LABEL_TRUE for r in records)
    return {
        "n": len(records),
        "n_true": n_true,
        "n_false": len(records) - n_true,
        "positive_rate": round(n_true / len(records), 4) if records else 0.0,
        "n_topics": len(balance),
        "topics": sorted(balance),
        "sources": sorted({r.source for r in records}),
        "per_topic": balance,
    }
