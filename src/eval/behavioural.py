"""E5 - the behavioural control.

*Does the INT4 model actually hallucinate more?* This is the
experiment that keeps the paper honest. If INT4 task accuracy is unchanged but
probe AUROC drops, the **detector** broke, not the model. Without it, any probe
result is ambiguous.

**Scoring by likelihood, not by generation.**

The obvious implementation is: generate an answer, string-match it against the
reference. Rejected, for three reasons that would each be fatal on their own:

1. *It measures formatting.* A model that answers "Nauru." versus "The smallest
   is Nauru" is equally right and unequally matched. Any string-match rule
   becomes a hidden hyperparameter, and a rule tuned on FP16 output may suit
   INT4 output worse - which would show up as a precision effect that is
   really a parsing effect.
2. *It is not deterministic in the way we need.* Greedy decoding is
   deterministic, but a single token flip early cascades through the whole
   continuation, so one rounding difference can produce a completely different
   string. That turns a small representational change into a large, noisy
   accuracy swing.
3. *It is slow.* Autoregressive decoding is one forward pass per token.

Likelihood scoring instead: for each candidate answer, compute the model's
total log-probability of that answer given the question, and pick the
highest-scoring candidate. One forward pass per candidate, exactly
deterministic, no parsing, and it is the standard protocol for TruthfulQA MC1.

**Two normalisations, both reported.** Raw summed log-probability favours short
answers, because every extra token multiplies in another probability < 1.
Dividing by token count removes that bias but over-corrects toward long generic
answers. Neither is "correct"; the honest move is to report both and check the
precision comparison does not depend on which one you pick.

**Closed-book by default.** HaluEval ships a `knowledge` passage with each
question. Including it tests reading comprehension; excluding it tests what the
model actually knows, which is what "hallucinate" means here. Configurable, and
the choice is recorded in the sidecar.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("quantprobe.eval.behavioural")


class BehaviouralError(RuntimeError):
    """A behavioural evaluation could not be run."""


def _continuation_span(tokenizer, prompt: str, continuation: str) -> tuple[list[int], int]:
    """Token ids for prompt+continuation, and where the continuation starts.

    The subtlety: tokenizers merge across the boundary. `tok(prompt)` is not
    always a prefix of `tok(prompt + " " + continuation)` - a trailing word and
    a leading space can fuse into one token. When that happens the naive
    "start = len(tok(prompt))" is off by one and scores the wrong tokens, with
    no error.

    So the prefix property is checked, and the start index is walked back to
    the longest genuine common prefix when it fails. The caller is told how
    often this happened.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + " " + continuation, add_special_tokens=True)["input_ids"]

    start = len(prompt_ids)
    if full_ids[:start] != prompt_ids:
        # Walk back to the longest actual common prefix.
        start = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            start += 1
    if start >= len(full_ids):
        raise BehaviouralError(
            f"continuation tokenized to nothing: {continuation[:60]!r}"
        )
    return full_ids, start


def score_batch(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    max_length: int = 512,
) -> list[dict[str, float]]:
    """Log-probability of each continuation given its prompt.

    Args:
        pairs: (prompt, continuation).

    Returns:
        Per pair: {"logprob_sum", "logprob_mean", "n_tokens"}.
    """
    import torch

    device = next(model.parameters()).device

    encoded, starts = [], []
    boundary_fixes = 0
    for prompt, continuation in pairs:
        ids, start = _continuation_span(tokenizer, prompt, continuation)
        expected = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        if start != expected:
            boundary_fixes += 1
        ids = ids[:max_length]
        if start >= len(ids):
            start = len(ids) - 1
        encoded.append(ids)
        starts.append(start)

    if boundary_fixes:
        logger.debug("%d/%d pairs needed a token-boundary correction",
                     boundary_fixes, len(pairs))

    width = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    input_ids = np.full((len(encoded), width), pad_id, dtype=np.int64)
    attention = np.zeros((len(encoded), width), dtype=np.int64)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = ids
        attention[i, : len(ids)] = 1

    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention).to(device)

    with torch.no_grad():
        logits = model(input_ids=ids_t, attention_mask=mask_t, use_cache=False).logits

    # Token t is predicted by the logits at position t-1.
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = ids_t[:, 1:]
    gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1).cpu().numpy()

    out = []
    for i, ids in enumerate(encoded):
        # Continuation token j (absolute) is scored at gathered index j-1.
        lo, hi = starts[i] - 1, len(ids) - 1
        span = gathered[i, lo:hi]
        n = max(int(span.size), 1)
        out.append(
            {
                "logprob_sum": float(span.sum()),
                "logprob_mean": float(span.sum() / n),
                "n_tokens": int(span.size),
            }
        )
    return out


def score_records(
    model,
    tokenizer,
    records: Sequence,
    batch_size: int = 8,
    max_length: int = 512,
    include_knowledge: bool = False,
    progress_every: int = 20,
) -> list[dict[str, Any]]:
    """Score every record's statement as a continuation of its prompt."""
    pairs = []
    for rec in records:
        prompt = rec.prompt
        if include_knowledge and rec.meta.get("knowledge"):
            prompt = f"{rec.meta['knowledge']}\n\n{prompt}"
        pairs.append((f"Q: {prompt}\nA:", rec.statement))

    results = []
    for n, start in enumerate(range(0, len(pairs), batch_size), 1):
        chunk = pairs[start : start + batch_size]
        scored = score_batch(model, tokenizer, chunk, max_length=max_length)
        for rec, s in zip(records[start : start + batch_size], scored):
            results.append({**s, "id": rec.id, "label": rec.label, "meta": rec.meta})
        if progress_every and n % progress_every == 0:
            logger.info("  scored %d/%d", min(start + batch_size, len(pairs)), len(pairs))
    return results


# ------------------------------------------------------------------- tasks


def multiple_choice_accuracy(
    scored: Sequence[dict[str, Any]], group_key: str = "question_index"
) -> dict[str, Any]:
    """Pick the highest-scoring candidate per question; is it the true one?

    Reported under both normalisations. A tie is counted as wrong rather than
    as half-right: with continuous log-probs an exact tie means the two
    candidates were identical strings, which is a data problem, not a
    borderline judgement.
    """
    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in scored:
        groups[row["meta"][group_key]].append(row)

    totals = {"logprob_sum": 0, "logprob_mean": 0}
    n_groups = 0
    skipped = 0

    for rows in groups.values():
        if not any(r["label"] == 1 for r in rows) or len(rows) < 2:
            skipped += 1
            continue
        n_groups += 1
        for metric in totals:
            best = max(rows, key=lambda r: r[metric])
            ties = [r for r in rows if r[metric] == best[metric]]
            totals[metric] += int(best["label"] == 1 and len(ties) == 1)

    if n_groups == 0:
        raise BehaviouralError("no scorable question groups")

    return {
        "n_questions": n_groups,
        "n_skipped": skipped,
        "accuracy_logprob_sum": totals["logprob_sum"] / n_groups,
        "accuracy_logprob_mean": totals["logprob_mean"] / n_groups,
        "chance": float(np.mean([1.0 / len(rows) for rows in groups.values() if len(rows) >= 2])),
    }


def pairwise_discrimination(
    scored: Sequence[dict[str, Any]], group_key: str = "row_index"
) -> dict[str, Any]:
    """Does the model score the true answer above the hallucinated one?

    A cleaner behavioural measure than MC1 because chance is exactly 0.50 by
    construction, so any deviation is interpretable without reference to how
    many distractors a question happened to have.
    """
    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in scored:
        groups[row["meta"][group_key]].append(row)

    totals = {"logprob_sum": 0, "logprob_mean": 0}
    n_pairs = 0
    skipped = 0

    for rows in groups.values():
        true_rows = [r for r in rows if r["label"] == 1]
        false_rows = [r for r in rows if r["label"] == 0]
        if len(true_rows) != 1 or len(false_rows) != 1:
            skipped += 1
            continue
        n_pairs += 1
        for metric in totals:
            totals[metric] += int(true_rows[0][metric] > false_rows[0][metric])

    if n_pairs == 0:
        raise BehaviouralError("no scorable answer pairs")

    return {
        "n_pairs": n_pairs,
        "n_skipped": skipped,
        "accuracy_logprob_sum": totals["logprob_sum"] / n_pairs,
        "accuracy_logprob_mean": totals["logprob_mean"] / n_pairs,
        "chance": 0.5,
    }


def binomial_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval. Correct near 0 and 1, unlike the normal approximation.

    Reported because the whole point of E5 is a comparison between precisions,
    and "0.61 vs 0.60" means nothing without knowing the interval is +/- 0.03.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


# --------------------------------------------------- length-immune judging

#: The verdict tokens. Compared against each other for the SAME prompt, so the
#: comparison is between two single tokens and length cannot enter it.
YES_NO = (" Yes", " No")


def judge_prompt(question: str, answer: str, knowledge: str | None = None) -> str:
    """Frame a (question, answer) pair as a yes/no factuality judgement."""
    head = f"{knowledge}\n\n" if knowledge else ""
    return (
        f"{head}Question: {question}\n"
        f"Proposed answer: {answer}\n"
        f"Is the proposed answer factually correct? Answer Yes or No.\n"
        f"Verdict:"
    )


def judge_batch(model, tokenizer, prompts, max_length: int = 512):
    """P(Yes) - P(No) in log space, for each prompt.

    Why this exists: scoring the ANSWER's likelihood measures its length as
    much as its truth. On HaluEval the true answer is an extractive span
    ("Delhi") while the hallucinated one is a fluent sentence, so summed
    log-prob picks the true answer every time and mean log-prob picks the
    false one every time - 1.00 and 0.00 accuracy, both pure artefact.

    Here the two candidate continuations are single tokens (" Yes" / " No")
    attached to the *same* prompt. They are the same length by construction,
    so the comparison cannot be contaminated by it. This is also HaluEval's
    own protocol: the task is judging an answer, not producing one.
    """
    import torch

    device = next(model.parameters()).device

    yes_id = tokenizer.encode(YES_NO[0], add_special_tokens=False)[0]
    no_id = tokenizer.encode(YES_NO[1], add_special_tokens=False)[0]

    encoded = [tokenizer(p, add_special_tokens=True)["input_ids"][:max_length] for p in prompts]
    width = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    input_ids = np.full((len(encoded), width), pad_id, dtype=np.int64)
    attention = np.zeros((len(encoded), width), dtype=np.int64)
    for i, ids in enumerate(encoded):
        # LEFT pad: the verdict token is predicted from the FINAL position, so
        # every row's prediction must sit at the same index.
        input_ids[i, width - len(ids):] = ids
        attention[i, width - len(ids):] = 1

    ids_t = torch.from_numpy(input_ids).to(device)
    mask_t = torch.from_numpy(attention).to(device)

    with torch.no_grad():
        logits = model(input_ids=ids_t, attention_mask=mask_t, use_cache=False).logits

    last = torch.log_softmax(logits[:, -1, :].float(), dim=-1).cpu().numpy()
    return [
        {
            "logp_yes": float(last[i, yes_id]),
            "logp_no": float(last[i, no_id]),
            "verdict_margin": float(last[i, yes_id] - last[i, no_id]),
        }
        for i in range(len(encoded))
    ]


def judge_records(
    model,
    tokenizer,
    records: Sequence,
    batch_size: int = 8,
    max_length: int = 512,
    include_knowledge: bool = False,
    progress_every: int = 20,
) -> list[dict[str, Any]]:
    """Yes/No factuality judgement for every record."""
    prompts = [
        judge_prompt(
            r.prompt,
            r.statement,
            r.meta.get("knowledge") if include_knowledge else None,
        )
        for r in records
    ]
    out = []
    for n, start in enumerate(range(0, len(prompts), batch_size), 1):
        scored = judge_batch(
            model, tokenizer, prompts[start : start + batch_size], max_length=max_length
        )
        for rec, s in zip(records[start : start + batch_size], scored):
            out.append({**s, "id": rec.id, "label": rec.label, "meta": rec.meta})
        if progress_every and n % progress_every == 0:
            logger.info("  judged %d/%d", min(start + batch_size, len(prompts)), len(prompts))
    return out


def judged_discrimination(scored: Sequence[dict[str, Any]], group_key: str = "row_index"):
    """Does the model judge the true answer more favourably than the false one?

    Uses `verdict_margin`, so chance is exactly 0.50 and length is not in play.
    """
    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in scored:
        groups[row["meta"][group_key]].append(row)

    correct = 0
    n_pairs = 0
    skipped = 0
    # Per-item outcomes, keyed by group. Every arm judges the SAME items, so
    # keeping these turns an unpaired two-proportion test into a paired
    # McNemar test, which is strictly more powerful on exactly this design.
    per_item: dict[Any, int] = {}
    for key, rows in groups.items():
        t = [r for r in rows if r["label"] == 1]
        f = [r for r in rows if r["label"] == 0]
        if len(t) != 1 or len(f) != 1:
            skipped += 1
            continue
        n_pairs += 1
        hit = int(t[0]["verdict_margin"] > f[0]["verdict_margin"])
        per_item[key] = hit
        correct += hit

    if n_pairs == 0:
        raise BehaviouralError("no scorable answer pairs")
    return {
        "n_pairs": n_pairs,
        "n_skipped": skipped,
        "accuracy": correct / n_pairs,
        "n_correct": correct,
        "chance": 0.5,
        "per_item": {str(k): v for k, v in sorted(per_item.items())},
    }


def length_confound_report(scored: Sequence[dict[str, Any]], group_key: str) -> dict[str, Any]:
    """How much does answer LENGTH alone predict the pick?

    A diagnostic, not a metric. If picking the shortest candidate scores as
    well as the model does, the model is not being measured - length is. This
    is exactly the failure that made the first HaluEval numbers 1.00 and 0.00.
    """
    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in scored:
        groups[row["meta"][group_key]].append(row)

    shortest_correct = 0
    longest_correct = 0
    usable = 0
    for rows in groups.values():
        if not any(r["label"] == 1 for r in rows) or len(rows) < 2:
            continue
        usable += 1
        shortest_correct += int(min(rows, key=lambda r: r["n_tokens"])["label"] == 1)
        longest_correct += int(max(rows, key=lambda r: r["n_tokens"])["label"] == 1)

    if usable == 0:
        return {"n": 0}
    return {
        "n": usable,
        "accuracy_if_always_shortest": shortest_correct / usable,
        "accuracy_if_always_longest": longest_correct / usable,
    }


def mcnemar(a_correct: dict[str, int], b_correct: dict[str, int]) -> dict[str, Any]:
    """Paired comparison of two arms judged on the SAME items.

    Both arms answer identical questions, so an unpaired two-proportion test
    throws away the pairing and loses power: it treats "both got item 42 right"
    as two independent coin flips rather than as a tie carrying no information.

    McNemar looks only at the *discordant* items - the ones where the arms
    disagree - which is where all the evidence about a difference actually
    lives. Exact binomial rather than the chi-square approximation, because
    the discordant count can be small and the approximation is unreliable
    there.

    Args:
        a_correct, b_correct: {item_id: 0/1} for each arm.

    Returns:
        Counts, the difference in accuracy, and a two-sided exact p-value.
    """
    from scipy import stats

    shared = sorted(set(a_correct) & set(b_correct))
    if not shared:
        raise BehaviouralError("the two arms share no items - nothing to pair")

    b_only = sum(1 for k in shared if a_correct[k] == 1 and b_correct[k] == 0)
    c_only = sum(1 for k in shared if a_correct[k] == 0 and b_correct[k] == 1)
    discordant = b_only + c_only

    if discordant == 0:
        return {
            "n_shared": len(shared), "a_only_correct": 0, "b_only_correct": 0,
            "n_discordant": 0, "delta_accuracy": 0.0, "p_value": 1.0,
            "note": "the arms agreed on every item - identical predictions",
        }

    p_value = float(stats.binomtest(b_only, discordant, 0.5).pvalue)
    return {
        "n_shared": len(shared),
        "a_only_correct": b_only,
        "b_only_correct": c_only,
        "n_discordant": discordant,
        "delta_accuracy": float(
            sum(b_correct[k] for k in shared) / len(shared)
            - sum(a_correct[k] for k in shared) / len(shared)
        ),
        "p_value": p_value,
    }
