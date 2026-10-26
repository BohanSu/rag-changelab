"""Capture only selected questions, escalating failed sentinel controls to a full run."""

from __future__ import annotations

from .adapters import Backend, capture
from .evaluate import compare_runs
from .models import Case, Plan, Report, Run, Snapshot
from .planner import build_plan


def run_candidate(
    before: Snapshot,
    after: Snapshot,
    cases: tuple[Case, ...],
    baseline: Run,
    backend: Backend,
    *,
    mode: str = "full",
    audit_controls: int = 1,
    allow_scoped_pass: bool = False,
) -> tuple[Run, Report]:
    """Use provenance to save calls; a failed/unknown control invalidates that scope.

    Each selected query is captured once. On escalation only previously skipped
    queries are added. A clean sentinel is not a proof of safety: skipped cases
    keep the default decision inconclusive.
    """
    plan = build_plan(before, after, cases, baseline, mode=mode, audit_controls=audit_controls)
    selected = set(plan.selected)
    candidate = capture(backend, after, [c for c in cases if c.case_id in selected])
    report = compare_runs(
        before, after, cases, baseline, candidate, plan, allow_scoped_pass=allow_scoped_pass
    )
    failed_controls = [
        r.case_id
        for r in report.results
        if r.case_id in plan.controls and r.after_status != "passed"
    ]
    if failed_controls and plan.skipped:
        remaining = capture(backend, after, [c for c in cases if c.case_id not in selected])
        candidate = Run(
            snapshot_digest=after.digest,
            producer=backend.name,
            observations=candidate.observations + remaining.observations,
        )
        plan = escalated_plan(before, after, cases, baseline, plan, failed_controls)
        report = compare_runs(before, after, cases, baseline, candidate, plan)
    return candidate, report


def escalated_plan(
    before: Snapshot,
    after: Snapshot,
    cases: tuple[Case, ...],
    baseline: Run,
    previous: Plan,
    failed_controls: list[str],
) -> Plan:
    full = build_plan(before, after, cases, baseline, mode="full")
    marker = "sentinel_escalation:" + ",".join(sorted(failed_controls))
    reasons = {
        key: tuple(sorted(set(value) | set(previous.reasons.get(key, ())) | {marker}))
        for key, value in full.reasons.items()
    }
    return full.model_copy(update={"controls": previous.controls, "reasons": reasons})
