import pytest

from rag_changelab.demo import inject
from rag_changelab.evaluate import compare_runs, inspect_observation
from rag_changelab.models import Citation, Document, Expectation, Observation, Snapshot
from rag_changelab.planner import build_plan


def test_legitimate_knowledge_update_is_not_a_regression(knowledge_update):
    before, after, cases, baseline, candidate = knowledge_update
    assert "30 days" in baseline.by_id["retention"].answer
    assert "7 days" in candidate.by_id["retention"].answer
    plan = build_plan(before, after, cases, baseline)
    report = compare_runs(before, after, cases, baseline, candidate, plan)
    assert report.decision == "passed"
    assert report.summary["checked"] == 8
    assert report.summary.get("regression", 0) == 0


def test_stale_answer_has_separate_fact_and_provenance_diagnostics(knowledge_update):
    before, after, cases, baseline, candidate = knowledge_update
    candidate = inject(candidate, baseline.by_id["retention"])
    report = compare_runs(
        before, after, cases, baseline, candidate, build_plan(before, after, cases, baseline)
    )
    retention = next(r for r in report.results if r.case_id == "retention")
    assert report.exit_code == 1
    assert retention.transition == "regression"
    assert {f.code for f in retention.after_findings} == {
        "required_answer_missing",
        "forbidden_answer_present",
        "stale_source_version",
        "unsupported_quote",
    }


def test_retrieved_inventory_does_not_count_as_a_citation(knowledge_update):
    _, after, cases, _, candidate = knowledge_update
    observation = candidate.by_id["retention"].model_copy(update={"citations": ()})
    assert observation.retrieved_sources == ("retention.md",)
    expected = next(c.after for c in cases if c.case_id == "retention")
    status, findings = inspect_observation(observation, after, expected)
    assert status == "failed"
    assert {f.code for f in findings} == {"required_citation_missing", "evidence_missing"}


def test_removed_source_is_a_failure_even_without_answer_contract(knowledge_update):
    _, after, _, baseline, _ = knowledge_update
    status, findings = inspect_observation(
        baseline.by_id["legacy"], after, Expectation(), removed=frozenset({"legacy.md"})
    )
    assert status == "failed"
    assert [f.code for f in findings] == ["deleted_source_cited"]


@pytest.mark.parametrize(
    "citation, expected_status, expected_code",
    [
        (Citation(source_id="retention.md"), "unknown", "unverified_citation"),
        (Citation(source_id="unmapped.pdf", quote="7 days"), "unknown", "source_outside_snapshot"),
        (Citation(source_id="retention.md", quote="999 years"), "failed", "unsupported_quote"),
    ],
)
def test_unverifiable_and_invalid_evidence_are_distinct(
    knowledge_update, citation, expected_status, expected_code
):
    _, after, cases, _, candidate = knowledge_update
    expected = next(c.after for c in cases if c.case_id == "retention")
    expected = expected.model_copy(update={"must_cite": ()})
    observation = candidate.by_id["retention"].model_copy(update={"citations": (citation,)})
    status, findings = inspect_observation(observation, after, expected)
    assert status == expected_status
    assert expected_code in {f.code for f in findings}


def test_backend_error_cannot_become_a_pass(knowledge_update):
    before, after, cases, baseline, candidate = knowledge_update
    failed = candidate.by_id["retention"].model_copy(update={"error": "HTTP 503"})
    report = compare_runs(
        before,
        after,
        cases,
        baseline,
        inject(candidate, failed),
        build_plan(before, after, cases, baseline),
    )
    assert report.decision == "inconclusive"
    assert report.exit_code == 3
    assert report.summary["unknown"] == 1


def test_preexisting_failure_is_not_counted_as_new_regression(knowledge_update):
    before, after, cases, baseline, candidate = knowledge_update
    baseline = inject(baseline, baseline.by_id["upload"].model_copy(update={"citations": ()}))
    candidate = inject(candidate, candidate.by_id["upload"].model_copy(update={"citations": ()}))
    report = compare_runs(
        before, after, cases, baseline, candidate, build_plan(before, after, cases, baseline)
    )
    assert report.decision == "failed"
    assert report.summary["existing_failure"] == 1
    assert report.summary.get("regression", 0) == 0


def test_repaired_baseline_failure_is_an_improvement(knowledge_update):
    before, after, cases, baseline, candidate = knowledge_update
    baseline = inject(baseline, baseline.by_id["upload"].model_copy(update={"citations": ()}))
    report = compare_runs(
        before, after, cases, baseline, candidate, build_plan(before, after, cases, baseline)
    )
    assert report.decision == "passed"
    assert report.summary["improved"] == 1


def test_multilingual_text_contracts_and_exact_quote_evidence():
    document = Document(source_id="政策.md", content="日志保存期限为七天。")
    snapshot = Snapshot(name="中文知识库", documents=(document,))
    answer = Observation(
        case_id="保存期限",
        question="日志保存多久？",
        answer="日志保存七天 [1]",
        citations=(Citation(source_id="政策.md", quote="保存期限为七天"),),
    )
    status, findings = inspect_observation(
        answer, snapshot, Expectation(contains=("七天",), excludes=("三十天",))
    )
    assert (status, findings) == ("passed", ())
