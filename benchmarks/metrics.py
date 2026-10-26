"""Counts first; uncertainty resamples source groups rather than fault variants."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def divide(numerator, denominator):
    return round(float(numerator / denominator), 6) if denominator else None


def confusion(rows: list[dict]) -> dict:
    binary = [r for r in rows if r["expected"] in {"passed", "failed"}]
    tp = sum(r["expected"] == "failed" and r["actual"] == "failed" for r in binary)
    fn = sum(r["expected"] == "failed" and r["actual"] != "failed" for r in binary)
    fp = sum(r["expected"] == "passed" and r["actual"] == "failed" for r in binary)
    tn = sum(r["expected"] == "passed" and r["actual"] == "passed" for r in binary)
    return {
        "observations": len(rows),
        "groups": len({r["group"] for r in rows}),
        "status_matches": sum(r["expected"] == r["actual"] for r in rows),
        "status_accuracy": divide(sum(r["expected"] == r["actual"] for r in rows), len(rows)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "false_positive_rate": divide(fp, fp + tn),
        "false_negative_rate": divide(fn, tp + fn),
        "unknown_rate": divide(sum(r["actual"] == "unknown" for r in rows), len(rows)),
        "unknown_expected": sum(r["expected"] == "unknown" for r in rows),
    }


def grouped_interval(rows: list[dict], *, repetitions: int = 1000, seed: int = 42) -> dict:
    groups = defaultdict(lambda: np.zeros(5, dtype=float))
    for row in rows:
        values = groups[row["group"]]
        expected, actual = row["expected"], row["actual"]
        values[0] += expected == actual
        values[1] += 1
        values[2] += expected == "failed" and actual == "failed"
        values[3] += expected == "failed"
        values[4] += expected == "passed" and actual == "failed"
    matrix = np.array(list(groups.values()))
    if not len(matrix):
        return {}
    rng = np.random.default_rng(seed)
    accuracy, recall = [], []
    for _ in range(repetitions):
        totals = matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0)
        accuracy.append(totals[0] / totals[1])
        if totals[3]:
            recall.append(totals[2] / totals[3])
    return {
        "source_groups": len(matrix),
        "repetitions": repetitions,
        "status_accuracy_95ci": [round(float(x), 6) for x in np.quantile(accuracy, [0.025, 0.975])],
        "failure_recall_95ci": [round(float(x), 6) for x in np.quantile(recall, [0.025, 0.975])]
        if recall
        else None,
        "interpretation": "Resampling observed source groups; not evidence of real-world prevalence or semantic label correctness",
    }


def summarize(rows: list[dict]) -> dict:
    result = {
        "all_splits": confusion(rows),
        "by_split": {},
        "by_family_test": {},
        "by_domain_test": {},
    }
    for name in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == name]
        result["by_split"][name] = confusion(subset)
    test = [r for r in rows if r["split"] == "test"]
    for key, output in (("family", "by_family_test"), ("domain", "by_domain_test")):
        for value in sorted({r[key] for r in test}):
            result[output][value] = confusion([r for r in test if r[key] == value])
    result["test_group_bootstrap"] = grouped_interval(test)
    return result
