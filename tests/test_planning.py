import pytest

from rag_changelab.demo import inject
from rag_changelab.evaluate import compare_runs
from rag_changelab.models import Document, Run, Snapshot
from rag_changelab.planner import build_plan
from rag_changelab.workflow import run_candidate


def test_change_graph_selects_dependents_and_one_reproducible_control(knowledge_update):
    before, after, cases, baseline, _ = knowledge_update
    plan = build_plan(before, after, cases, baseline, mode="impact", audit_controls=1)
    assert set(plan.impacted) == {"retention", "authentication", "legacy"}
    assert len(plan.controls) == 1
    assert set(plan.controls).isdisjoint(plan.impacted)
    assert len(plan.selected) == 4
    assert len(plan.skipped) == 4
    assert plan == build_plan(before, after, cases, baseline, mode="impact", audit_controls=1)
    assert plan.reasons["legacy"] == ("removed:legacy.md",)


@pytest.mark.parametrize("change", ["added", "unmapped"])
def test_unseen_source_changes_force_full_retest(knowledge_update, change):
    before, after, cases, baseline, _ = knowledge_update
    untracked = Document(source_id="new-policy.md", content="New source can change any retrieval.")
    if change == "unmapped":
        before = Snapshot(name=before.name, documents=before.documents + (untracked,))
        baseline = baseline.model_copy(update={"snapshot_digest": before.digest})
        untracked = untracked.model_copy(update={"content": "Changed untracked source."})
    after = Snapshot(name=after.name, documents=after.documents + (untracked,))
    plan = build_plan(before, after, cases, baseline, mode="impact", audit_controls=0)
    assert len(plan.selected) == len(cases)
    assert not plan.skipped
    assert all(
        "new_or_unmapped_source_requires_full_retest" in why for why in plan.reasons.values()
    )


def test_missing_baseline_observation_is_always_selected(knowledge_update):
    before, after, cases, baseline, _ = knowledge_update
    baseline = baseline.model_copy(
        update={"observations": tuple(o for o in baseline.observations if o.case_id != "upload")}
    )
    plan = build_plan(before, after, cases, baseline, mode="impact", audit_controls=0)
    assert "upload" in plan.selected
    assert "baseline_dependency_unknown" in plan.reasons["upload"]


def test_clean_subset_never_implies_full_suite_pass(knowledge_update):
    before, after, cases, baseline, candidate = knowledge_update
    plan = build_plan(before, after, cases, baseline, mode="impact", audit_controls=1)
    report = compare_runs(before, after, cases, baseline, candidate, plan)
    assert report.decision == "inconclusive"
    assert report.exit_code == 3
    assert report.summary["skipped"] == 4
    opted_in = compare_runs(before, after, cases, baseline, candidate, plan, allow_scoped_pass=True)
    assert opted_in.decision == "scoped_pass"
    assert opted_in.exit_code == 0


@pytest.mark.parametrize("control_error", [False, True])
def test_bad_sentinel_expands_run_without_duplicate_backend_calls(knowledge_update, control_error):
    before, after, cases, baseline, healthy = knowledge_update
    initial = build_plan(before, after, cases, baseline, mode="impact", audit_controls=1)
    control = initial.controls[0]
    hidden_failure = initial.skipped[0]

    class CountingBackend:
        name = "test-counting"

        def __init__(self):
            self.calls = []

        def observe(self, case, snapshot):
            self.calls.append(case.case_id)
            observation = healthy.by_id[case.case_id]
            if case.case_id == control and control_error:
                return observation.model_copy(update={"error": "timeout"})
            if case.case_id in {control, hidden_failure}:
                return observation.model_copy(update={"citations": ()})
            return observation

    backend = CountingBackend()
    candidate, report = run_candidate(
        before, after, cases, baseline, backend, mode="impact", audit_controls=1
    )
    assert len(backend.calls) == len(set(backend.calls)) == len(cases)
    assert len(candidate.observations) == 8
    assert report.plan.mode == "full"
    assert report.decision == "failed"
    assert report.summary["candidate_failures"] == (1 if control_error else 2)
    assert all(
        any(reason.startswith("sentinel_escalation:") for reason in reasons)
        for reasons in report.plan.reasons.values()
    )


def test_clean_sentinel_does_not_hide_an_unchecked_failure(knowledge_update):
    before, after, cases, baseline, candidate = knowledge_update
    plan = build_plan(before, after, cases, baseline, mode="impact", audit_controls=1)
    candidate = inject(
        candidate, candidate.by_id[plan.skipped[0]].model_copy(update={"citations": ()})
    )
    scoped = compare_runs(before, after, cases, baseline, candidate, plan)
    full = compare_runs(
        before, after, cases, baseline, candidate, build_plan(before, after, cases, baseline)
    )
    assert scoped.decision == "inconclusive"
    assert full.decision == "failed"


@pytest.mark.parametrize("binding", ["question", "baseline", "snapshot"])
def test_stale_plans_are_rejected(knowledge_update, binding):
    before, after, cases, baseline, candidate = knowledge_update
    plan = build_plan(before, after, cases, baseline)
    if binding == "question":
        cases = (cases[0].model_copy(update={"question": "A different question?"}),) + cases[1:]
    elif binding == "baseline":
        baseline = baseline.model_copy(update={"producer": "another-producer"})
    else:
        after = Snapshot(name=after.name, documents=after.documents[:-1])
    with pytest.raises(ValueError):
        compare_runs(before, after, cases, baseline, candidate, plan)


@pytest.mark.parametrize(
    "mutation",
    [
        {"selected": (), "skipped": ()},
        {"selected": ("retention", "retention")},
        {"mode": "full", "skipped": ("upload",)},
        {"controls": ("unrecognized",)},
        {"changes": ()},
    ],
)
def test_malformed_plan_cannot_manufacture_a_pass(knowledge_update, mutation):
    before, after, cases, baseline, candidate = knowledge_update
    plan = build_plan(before, after, cases, baseline).model_copy(update=mutation)
    with pytest.raises(ValueError):
        compare_runs(before, after, cases, baseline, candidate, plan)


def test_no_corpus_change_still_requires_explicit_scope_policy(knowledge_update):
    before, _, cases, baseline, _ = knowledge_update
    # Only unchanged cases have the same before/after expectation in this fixture.
    cases = tuple(c for c in cases if c.before == c.after)
    ids = {c.case_id for c in cases}
    baseline = Run(
        snapshot_digest=before.digest,
        producer=baseline.producer,
        observations=tuple(o for o in baseline.observations if o.case_id in ids),
    )
    plan = build_plan(before, before, cases, baseline, mode="impact", audit_controls=0)
    assert not plan.selected
    assert compare_runs(before, before, cases, baseline, baseline, plan).decision == "inconclusive"
