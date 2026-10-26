"""External human annotations; simple lexical baseline, with no fabricated metadata."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from data import source_context
from metrics import confusion, grouped_interval
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def tokens(text):
    return set(re.findall(r"[^\W_]+", text.casefold())) - ENGLISH_STOP_WORDS


def evaluate(sources, responses, output):
    contexts = {s["source_id"]: tokens(source_context(s)) for s in sources}
    rows = []
    for response in responses:
        answer_tokens = tokens(response["response"])
        novel = answer_tokens - contexts[response["source_id"]]
        score = len(novel) / max(1, len(answer_tokens))
        labels = [label for label in response["labels"] if not label.get("implicit_true", False)]
        rows.append(
            {
                "id": response["id"],
                "group": response["source_id"],
                "split": response["split"],
                "task": response["task_type"],
                "model": response["model"],
                "quality": response["quality"],
                "score": score,
                "expected": "failed" if labels else "passed",
                "any_annotation_label": bool(response["labels"]),
                "strict_span_types": [label["label_type"] for label in labels],
            }
        )
    development = [r for r in rows if r["split"] == "train" and r["quality"] == "good"]
    thresholds = np.linspace(0, 1, 101)

    def balanced_accuracy(threshold):
        bad = [r for r in development if r["expected"] == "failed"]
        good = [r for r in development if r["expected"] == "passed"]
        return 0.5 * (
            sum(r["score"] >= threshold for r in bad) / len(bad)
            + sum(r["score"] < threshold for r in good) / len(good)
        )

    threshold = max(thresholds, key=lambda t: (balanced_accuracy(t), float(t)))
    for row in rows:
        row["actual"] = "failed" if row["score"] >= threshold else "passed"
        output.write(__import__("json").dumps(row) + "\n")
    test = [r for r in rows if r["split"] == "test" and r["quality"] == "good"]
    return {
        "tool_semantic_score": None,
        "reason_unscored": "ChangeLab requires caller-written answer contracts and answer-citation metadata. RAGTruth free-form responses do not provide them. Injecting oracle citations/contracts would be invalid.",
        "baseline": "unmatched content-token fraction, threshold calibrated on official train only",
        "baseline_is_not": "Promptfoo, Ragas, an NLI model, or a claimed state-of-the-art comparator",
        "threshold": float(threshold),
        "threshold_selection": "maximize train balanced accuracy, tie prefer higher threshold; 101 fixed candidates",
        "train_good_responses": len(development),
        "test": confusion(test),
        "test_group_bootstrap": grouped_interval(test),
        "by_task_test": {
            task: confusion([r for r in test if r["task"] == task])
            for task in sorted({r["task"] for r in test})
        },
        "by_model_test": {
            model: confusion([r for r in test if r["model"] == model])
            for model in sorted({r["model"] for r in test})
        },
        "test_quality_exclusions": dict(
            Counter(r["quality"] for r in rows if r["split"] == "test" and r["quality"] != "good")
        ),
        "test_strict_labels": dict(
            Counter(label for r in test for label in r["strict_span_types"])
        ),
        "test_label_sensitivity": {
            "any_annotation_positive_responses": sum(r["any_annotation_label"] for r in test),
            "excluding_implicit_true_positive_responses": sum(
                r["expected"] == "failed" for r in test
            ),
        },
        "native_output_support": "Unscored unsupported format is not a false negative and not a pass",
    }
