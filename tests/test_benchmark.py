"""Check corpus handling and independent statistical units, without downloading data."""

import importlib
import json
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audit_modules():
    path = str(Path(__file__).resolve().parents[1] / "benchmarks")
    sys.path.insert(0, path)
    yield {
        name: importlib.import_module(name)
        for name in ("data", "releases", "metrics", "promptfoo_compare")
    }
    sys.path.remove(path)


def test_release_parser_ignores_code_headings_and_preserves_parent_context(audit_modules):
    text = "# Authentication\n" + "intro " * 20 + "\n## Examples\n" + "content " * 20
    text += "\n```python\n# This is not a documentation heading\n```\n"
    text += "# Retries\n" + "retry " * 20 + "\n## Examples\n" + "usage " * 20
    sections = audit_modules["releases"].headings(text)
    assert [title for title, _ in sections] == [
        "Authentication",
        "Authentication / Examples",
        "Retries",
        "Retries / Examples",
    ]


def test_rst_sections_are_read_without_treating_underlines_as_content(audit_modules):
    text = "Usage\n=====\n" + "usage paragraph " * 10 + "\nOptions\n-------\n" + "options " * 20
    sections = audit_modules["releases"].headings(text)
    assert [title for title, _ in sections] == ["Usage", "Usage / Options"]
    assert "=====" not in sections[0][1]


def test_real_doc_reader_excludes_translations_and_release_notes(audit_modules, tmp_path):
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in (
            "docs/en/docs/tutorial.md",
            "docs/fr/docs/tutorial.md",
            "docs/en/docs/release-notes.md",
        ):
            archive.writestr("source/" + name, "# Tutorial\n" + "text " * 30)
    documents = audit_modules["releases"].documentation(path, "fastapi/fastapi")
    assert list(documents) == ["docs/tutorial.md"]


def test_temporal_contexts_stay_together_and_empty_gold_is_quarantined(audit_modules, tmp_path):
    base = {
        "index": 0,
        "category": "amendment",
        "query_insert_time": "2024-01-01 00:00:00",
        "query_ko": "Which rule applies in 2024?",
        "answer": "Rule A",
        "docs_ko": [
            {
                "text": "Rule A applies.",
                "doc_insert_time": "2023-01-01 00:00:00",
                "global_doc_id": 1,
            }
        ],
        "gold_docs_final": [0],
    }
    later = {**base, "index": 1, "query_insert_time": "2024-02-01 00:00:00", "answer": "Rule B"}
    invalid = {**base, "index": 2, "answer": ""}
    with zipfile.ZipFile(tmp_path / "timelyqa.zip", "w") as archive:
        archive.writestr(
            "timelyrag_law_dataset_v2.jsonl",
            "\n".join(json.dumps(r) for r in (base, later, base, invalid)),
        )
    rows, inventory = audit_modules["data"].timelyqa(tmp_path)
    assert len(rows) == 2
    assert rows[0]["group"] == rows[1]["group"]
    assert rows[0]["split"] == rows[1]["split"]
    assert inventory["domains"]["law"]["empty_reference_or_question_rows"] == 1
    assert inventory["domains"]["law"]["exact_duplicate_context_rows"] == 1


def test_numeric_date_variants_cannot_leak_across_splits(audit_modules):
    group = audit_modules["data"].template_group
    assert group("Policy valid on 2023-01-01?") == group("Policy valid on 2024-02-02?")


def test_timestamps_are_parsed_as_dates_without_guessing_corrupt_values(audit_modules):
    parse = audit_modules["data"].timestamp
    assert parse("2022년 7월 1일 09:00:00") == parse("2022-07-01 09:00:00")
    assert parse("2022-7-01 09:00:00") == parse("2022-07-01 09:00:00")
    assert parse("2020-!1-01 09:00:00") is None
    assert parse("2016-02-30 08:00:00") is None


def test_uncertainty_resamples_groups_not_many_correlated_variants(audit_modules):
    rows = [
        {"group": group, "expected": "failed", "actual": actual}
        for group, actual in (("one", "failed"), ("two", "passed"))
        for _ in range(100)
    ]
    result = audit_modules["metrics"].grouped_interval(rows, repetitions=1000)
    assert result["source_groups"] == 2
    assert result["status_accuracy_95ci"] == [0, 1]
    assert result["failure_recall_95ci"] == [0, 1]


def test_promptfoo_assertion_failure_and_redacted_hashes_are_parsed_correctly(
    audit_modules, tmp_path, monkeypatch
):
    module = audit_modules["promptfoo_compare"]
    monkeypatch.setattr(module, "report", lambda *_: None)
    tests = [
        {
            "description": str(i),
            "metadata": {
                "family": "fixture",
                "group": str(i) * 64,
                "changelab_status": "passed" if i == 0 else "failed",
            },
        }
        for i in range(2)
    ]
    rows = [
        {
            "testIdx": i,
            "testCase": {"description": str(i), "metadata": {"group": "[REDACTED]"}},
            "success": i == 0,
            "error": None if i == 0 else "Expected output to contain",
        }
        for i in range(2)
    ]
    (tmp_path / "promptfoo-config.json").write_text(json.dumps({"tests": tests}))
    raw = tmp_path / "results.json"
    raw.write_text(
        json.dumps(
            {"results": {"results": rows, "stats": {"errors": 0, "tokenUsage": {"total": 0}}}}
        )
    )
    (tmp_path / "summary.json").write_text("{}")
    module.compare(raw, tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())["promptfoo_comparison"]
    assert result["source_groups"] == 2
    assert result["status_mismatches"] == 0


@pytest.mark.parametrize("failure", ["missing", "duplicate", "runtime-error"])
def test_promptfoo_incomplete_or_error_results_cannot_claim_parity(
    audit_modules, tmp_path, monkeypatch, failure
):
    module = audit_modules["promptfoo_compare"]
    monkeypatch.setattr(module, "report", lambda *_: None)
    tests = [
        {
            "description": str(i),
            "metadata": {"family": "fixture", "group": str(i), "changelab_status": "passed"},
        }
        for i in range(2)
    ]
    rows = [{"testIdx": i, "testCase": {"description": str(i)}, "success": True} for i in range(2)]
    if failure == "missing":
        rows.pop()
    if failure == "duplicate":
        rows[1]["testIdx"] = 0
    (tmp_path / "promptfoo-config.json").write_text(json.dumps({"tests": tests}))
    raw = tmp_path / "results.json"
    raw.write_text(
        json.dumps(
            {
                "results": {
                    "results": rows,
                    "stats": {
                        "errors": int(failure == "runtime-error"),
                        "tokenUsage": {"total": 0},
                    },
                }
            }
        )
    )
    with pytest.raises(ValueError):
        module.compare(raw, tmp_path)
