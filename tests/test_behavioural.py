"""Phase 5 tests - E5 scoring, aggregation, and the length confound.

The one that matters most is
`test_likelihood_scoring_is_confounded_by_length_on_halueval_shapes`. It
reproduces, in miniature, a real failure caught during development: on
HaluEval the true answer is an extractive span and the false one is a fluent
sentence, so summed log-probability scored 1.00 accuracy and mean
log-probability scored 0.00 on the *same data*. Both were measuring length.
That test exists so the trap cannot be re-entered silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import behavioural as B  # noqa: E402


class StubTokenizer:
    """Whitespace tokenizer with an optional boundary-merge quirk."""

    def __init__(self, merge_boundary: bool = False):
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.merge_boundary = merge_boundary

    def __call__(self, text, add_special_tokens=True):
        words = text.split()
        ids = [len(w) + 10 for w in words]
        if self.merge_boundary and len(ids) >= 2:
            # Simulate the last prompt token fusing with the first
            # continuation token, which breaks the naive prefix assumption.
            ids[-1] = 999
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [len(text.strip()) + 100]


def scored_row(idx, label, lp_sum, lp_mean, n_tokens, key="question_index"):
    return {
        "id": f"r{idx}", "label": label, "logprob_sum": lp_sum,
        "logprob_mean": lp_mean, "n_tokens": n_tokens, "meta": {key: idx // 10},
    }


# ------------------------------------------------------------ token spans


def test_continuation_span_finds_the_prompt_prefix():
    tok = StubTokenizer()
    ids, start = B._continuation_span(tok, "one two three", "four five")
    assert start == 3
    assert len(ids) == 5


def test_continuation_span_survives_a_boundary_merge():
    """A trailing word and a leading space can fuse into one token; the naive
    len(tok(prompt)) is then wrong and silently scores the wrong tokens."""
    tok = StubTokenizer(merge_boundary=True)
    ids, start = B._continuation_span(tok, "one two three", "four five")
    assert 0 <= start < len(ids)
    assert ids[:start] == tok("one two three")["input_ids"][:start]


def test_continuation_span_rejects_an_empty_continuation():
    tok = StubTokenizer()
    with pytest.raises(B.BehaviouralError, match="tokenized to nothing"):
        B._continuation_span(tok, "one two", "")


# ------------------------------------------------------- multiple choice


def test_multiple_choice_picks_the_argmax():
    rows = [
        scored_row(0, 1, -3.0, -1.0, 3),
        scored_row(1, 0, -9.0, -3.0, 3),
        scored_row(2, 0, -8.0, -2.7, 3),
    ]
    for r in rows:
        r["meta"] = {"question_index": 0}
    out = B.multiple_choice_accuracy(rows)
    assert out["n_questions"] == 1
    assert out["accuracy_logprob_sum"] == 1.0
    assert out["accuracy_logprob_mean"] == 1.0


def test_multiple_choice_counts_a_tie_as_wrong():
    rows = [{"label": 1, "logprob_sum": -2.0, "logprob_mean": -1.0, "n_tokens": 2,
             "meta": {"question_index": 0}},
            {"label": 0, "logprob_sum": -2.0, "logprob_mean": -1.0, "n_tokens": 2,
             "meta": {"question_index": 0}}]
    out = B.multiple_choice_accuracy(rows)
    assert out["accuracy_logprob_sum"] == 0.0


def test_multiple_choice_skips_groups_with_no_true_answer():
    rows = [{"label": 0, "logprob_sum": -1.0, "logprob_mean": -1.0, "n_tokens": 1,
             "meta": {"question_index": 0}},
            {"label": 0, "logprob_sum": -2.0, "logprob_mean": -2.0, "n_tokens": 1,
             "meta": {"question_index": 0}}]
    with pytest.raises(B.BehaviouralError, match="no scorable"):
        B.multiple_choice_accuracy(rows)


# -------------------------------------------------- THE LENGTH CONFOUND


def test_likelihood_scoring_is_confounded_by_length_on_halueval_shapes():
    """Reproduces the real failure: 1.00 under one normalisation, 0.00 under
    the other, on identical data, purely from answer length.

    HaluEval's true answer is a short extractive span; its false answer is a
    fluent sentence. Every extra token multiplies in another probability < 1,
    so summed log-prob always prefers the short one; per-token mean always
    prefers the fluent one. Neither has anything to do with truth.
    """
    rows = []
    for i in range(20):
        # true: 2 tokens, mediocre per-token prob
        rows.append({"label": 1, "logprob_sum": -4.0, "logprob_mean": -2.0,
                     "n_tokens": 2, "meta": {"row_index": i}})
        # false: 10 tokens, BETTER per-token prob (fluent sentence)
        rows.append({"label": 0, "logprob_sum": -10.0, "logprob_mean": -1.0,
                     "n_tokens": 10, "meta": {"row_index": i}})

    out = B.pairwise_discrimination(rows)
    assert out["accuracy_logprob_sum"] == 1.0, "sum should always pick the shorter"
    assert out["accuracy_logprob_mean"] == 0.0, "mean should always pick the fluent"

    # And the diagnostic must name it.
    report = B.length_confound_report(rows, "row_index")
    assert report["accuracy_if_always_shortest"] == 1.0
    assert report["accuracy_if_always_longest"] == 0.0


def test_length_confound_report_is_uninformative_when_lengths_match():
    rows = []
    for i in range(10):
        rows.append({"label": 1, "logprob_sum": -3.0, "logprob_mean": -1.0,
                     "n_tokens": 5, "meta": {"row_index": i}})
        rows.append({"label": 0, "logprob_sum": -5.0, "logprob_mean": -1.7,
                     "n_tokens": 5, "meta": {"row_index": i}})
    report = B.length_confound_report(rows, "row_index")
    # With equal lengths, min/max pick arbitrarily - the point is only that
    # the diagnostic runs and does not claim a confound where none exists.
    assert report["n"] == 10


# ---------------------------------------------------- judged discrimination


def test_judged_discrimination_uses_the_verdict_margin():
    rows = []
    for i in range(10):
        rows.append({"label": 1, "verdict_margin": 0.5, "meta": {"row_index": i}})
        rows.append({"label": 0, "verdict_margin": -0.5, "meta": {"row_index": i}})
    out = B.judged_discrimination(rows)
    assert out["accuracy"] == 1.0
    assert out["n_pairs"] == 10 and out["chance"] == 0.5


def test_judged_discrimination_is_immune_to_answer_length():
    """The same data that breaks likelihood scoring must not break judging.

    The judgement compares two single tokens on the same prompt, so length
    cannot enter. Here the model judges correctly despite a 5x length gap.
    """
    rows = []
    for i in range(10):
        rows.append({"label": 1, "verdict_margin": 1.2, "n_tokens": 2,
                     "meta": {"row_index": i}})
        rows.append({"label": 0, "verdict_margin": -0.3, "n_tokens": 10,
                     "meta": {"row_index": i}})
    assert B.judged_discrimination(rows)["accuracy"] == 1.0


def test_judged_discrimination_skips_malformed_pairs():
    rows = [{"label": 1, "verdict_margin": 1.0, "meta": {"row_index": 0}},
            {"label": 1, "verdict_margin": 0.5, "meta": {"row_index": 0}},
            {"label": 1, "verdict_margin": 1.0, "meta": {"row_index": 1}},
            {"label": 0, "verdict_margin": 0.0, "meta": {"row_index": 1}}]
    out = B.judged_discrimination(rows)
    assert out["n_pairs"] == 1 and out["n_skipped"] == 1


def test_judge_prompt_includes_knowledge_only_when_given():
    assert "PASSAGE" not in B.judge_prompt("q", "a")
    assert "PASSAGE" in B.judge_prompt("q", "a", knowledge="PASSAGE")
    assert B.judge_prompt("q", "a").rstrip().endswith("Verdict:")


# ------------------------------------------------------------------- CIs


def test_wilson_interval_brackets_the_estimate():
    lo, hi = B.binomial_ci(60, 100)
    assert lo < 0.60 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_interval_stays_in_range_at_the_extremes():
    """The normal approximation goes below 0 here; Wilson does not."""
    lo, hi = B.binomial_ci(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.3
    lo, hi = B.binomial_ci(20, 20)
    assert hi == 1.0 and 0.7 < lo < 1.0


def test_wilson_interval_narrows_with_more_data():
    small = B.binomial_ci(30, 50)
    large = B.binomial_ci(600, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_handles_zero_n():
    assert B.binomial_ci(0, 0) == (0.0, 0.0)


# ------------------------------------------------------------- McNemar


def test_mcnemar_finds_a_real_paired_difference():
    """20 items where B wins, 5 where A does, on the same 100 items."""
    a = {str(i): 1 for i in range(100)}
    b = dict(a)
    for i in range(20):
        a[str(i)], b[str(i)] = 0, 1
    for i in range(20, 25):
        a[str(i)], b[str(i)] = 1, 0
    out = B.mcnemar(a, b)
    assert out["n_discordant"] == 25
    assert out["delta_accuracy"] == pytest.approx(0.15)
    assert out["p_value"] < 0.01


def test_mcnemar_reports_no_difference_when_arms_agree():
    a = {str(i): i % 2 for i in range(50)}
    out = B.mcnemar(a, dict(a))
    assert out["n_discordant"] == 0
    assert out["p_value"] == 1.0
    assert "agreed on every item" in out["note"]


def test_mcnemar_ignores_concordant_items():
    """Adding items both arms get right must not change the p-value.

    That is the whole point of pairing: ties carry no information about a
    difference, and an unpaired test would let them dilute the evidence.
    """
    a = {"x": 1, "y": 0}
    b = {"x": 0, "y": 1}
    before = B.mcnemar(a, b)["p_value"]
    for i in range(500):
        a[f"pad{i}"] = 1
        b[f"pad{i}"] = 1
    assert B.mcnemar(a, b)["p_value"] == pytest.approx(before)


def test_mcnemar_uses_only_shared_items():
    a = {"x": 1, "y": 0, "only_a": 1}
    b = {"x": 0, "y": 1, "only_b": 0}
    assert B.mcnemar(a, b)["n_shared"] == 2


def test_mcnemar_errors_when_nothing_is_shared():
    with pytest.raises(B.BehaviouralError, match="share no items"):
        B.mcnemar({"a": 1}, {"b": 1})


def test_judged_discrimination_emits_per_item_outcomes():
    """Without these, arms can only be compared unpaired."""
    rows = []
    for i in range(5):
        rows.append({"label": 1, "verdict_margin": 1.0, "meta": {"row_index": i}})
        rows.append({"label": 0, "verdict_margin": 0.0, "meta": {"row_index": i}})
    out = B.judged_discrimination(rows)
    assert set(out["per_item"]) == {"0", "1", "2", "3", "4"}
    assert all(v == 1 for v in out["per_item"].values())
