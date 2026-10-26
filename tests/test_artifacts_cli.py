import json

import pytest
from pydantic import ValidationError

from rag_changelab.artifacts import read_model, snapshot_directory, write_model
from rag_changelab.cli import main
from rag_changelab.demo import inject
from rag_changelab.evaluate import compare_runs
from rag_changelab.models import Cases, Snapshot
from rag_changelab.planner import build_plan
from rag_changelab.reporting import render_report


def test_snapshots_bind_relative_paths_and_content_independent_of_directory(tmp_path):
    for root_name in ("a", "b"):
        root = tmp_path / root_name
        (root / "sub").mkdir(parents=True)
        (root / "sub/policy.md").write_text("日志保存七天。", encoding="utf-8")
        (root / "ignored.png").write_bytes(b"not a text document")
    a = snapshot_directory(tmp_path / "a")
    b = snapshot_directory(tmp_path / "b")
    assert a.digest == b.digest
    assert a.documents[0].source_id == "sub/policy.md"
    write_model(tmp_path / "snapshot.json", a)
    assert read_model(tmp_path / "snapshot.json", Snapshot) == a
    payload = a.model_dump(mode="json")
    payload["documents"][0]["content"] = "Changed without updating the digest"
    with pytest.raises(ValidationError, match="digest"):
        Snapshot.model_validate(payload)


def test_snapshot_does_not_follow_external_document_symlink(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (tmp_path / "outside.md").write_text("Outside the corpus", encoding="utf-8")
    (root / "symlink.md").symlink_to(tmp_path / "outside.md")
    with pytest.raises(ValueError, match="symlink"):
        snapshot_directory(root)


def test_duplicate_source_and_case_ids_are_rejected(knowledge_update):
    before, _, cases, _, _ = knowledge_update
    with pytest.raises(ValidationError, match="duplicate"):
        Snapshot(name="duplicate", documents=before.documents + before.documents[:1])
    with pytest.raises(ValidationError, match="duplicate"):
        Cases(cases=cases + cases[:1])


def test_html_escapes_answers_questions_and_source_content(knowledge_update):
    before, after, cases, baseline, candidate = knowledge_update
    payload = '<script>alert("untrusted")</script>'
    candidate = inject(
        candidate, candidate.by_id["retention"].model_copy(update={"answer": payload})
    )
    report = compare_runs(
        before, after, cases, baseline, candidate, build_plan(before, after, cases, baseline)
    )
    html = render_report(report, before, after)
    assert payload not in html
    assert "&lt;script&gt;" in html
    assert "<script" not in html


def test_offline_demo_and_ci_exit_codes_are_replayable(tmp_path, capsys):
    demo = tmp_path / "demo"
    assert main(["demo", "--out", str(demo)]) == 0
    verification = json.loads((demo / "verification.json").read_text())
    assert verification["all_expected"]
    assert len(verification["scenarios"]) == 7
    for scenario in (
        "healthy",
        "stale-answer",
        "citation-loss",
        "deleted-source",
        "unsupported-quote",
    ):
        code = main(
            [
                "check",
                str(demo / scenario / "suite.json"),
                "--out",
                str(tmp_path / "replay" / scenario),
            ]
        )
        assert code == (0 if scenario == "healthy" else 1)
    assert (
        main(
            [
                "check",
                str(demo / "healthy/suite.json"),
                "--mode",
                "impact",
                "--out",
                str(tmp_path / "scoped"),
            ]
        )
        == 3
    )
    assert (
        main(
            [
                "check",
                str(demo / "healthy/suite.json"),
                "--mode",
                "impact",
                "--allow-scoped-pass",
                "--out",
                str(tmp_path / "opted-in"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "plan",
                str(demo / "healthy/suite.json"),
                "--mode",
                "impact",
                "--out",
                str(tmp_path / "plan.json"),
            ]
        )
        == 0
    )
    assert len(json.loads((tmp_path / "plan.json").read_text())["selected"]) == 4
    assert "decision" in capsys.readouterr().out


def test_invalid_input_is_distinct_from_a_regression(tmp_path):
    assert main(["check", str(tmp_path / "missing.json"), "--out", str(tmp_path / "out")]) == 2
    (tmp_path / "invalid.json").write_text('{"before_snapshot":"missing"}', encoding="utf-8")
    assert main(["check", str(tmp_path / "invalid.json"), "--out", str(tmp_path / "out")]) == 2
