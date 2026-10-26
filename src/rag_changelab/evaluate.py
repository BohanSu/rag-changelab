"""Separate answer-contract violations, evidence failures, and unknown provenance."""

from __future__ import annotations

from collections import Counter

from .models import (
    Case,
    CaseResult,
    Expectation,
    Finding,
    Observation,
    Plan,
    Report,
    Run,
    Snapshot,
    fingerprint,
)
from .planner import document_changes, validate_run


def inspect_observation(
    observation: Observation | None,
    snapshot: Snapshot,
    expected: Expectation,
    *,
    removed: frozenset[str] = frozenset(),
) -> tuple[str, tuple[Finding, ...]]:
    findings: list[Finding] = []
    if observation is None or observation.error:
        reason = observation.error if observation else "observation was not captured"
        return "unknown", (
            Finding(code="backend_or_capture_missing", level="unknown", message=reason),
        )
    documents = snapshot.by_id
    folded = observation.answer.casefold()
    for text in expected.contains:
        if text.casefold() not in folded:
            findings.append(
                Finding(
                    code="required_answer_missing",
                    level="failure",
                    message=f"Required answer text missing: {text}",
                )
            )
    for text in expected.excludes:
        if text.casefold() in folded:
            findings.append(
                Finding(
                    code="forbidden_answer_present",
                    level="failure",
                    message=f"Forbidden answer text present: {text}",
                )
            )
    cited = {citation.source_id for citation in observation.citations}
    for source_id in expected.must_cite:
        if source_id not in cited:
            findings.append(
                Finding(
                    code="required_citation_missing",
                    level="failure",
                    message=f"Required source was not cited: {source_id}",
                    source_id=source_id,
                )
            )
    if expected.evidence_required and not cited:
        findings.append(
            Finding(
                code="evidence_missing",
                level="failure",
                message="This answer requires cited evidence.",
            )
        )
    for citation in observation.citations:
        document = documents.get(citation.source_id)
        if document is None:
            deleted = citation.source_id in removed
            findings.append(
                Finding(
                    code="deleted_source_cited" if deleted else "source_outside_snapshot",
                    level="failure" if deleted else "unknown",
                    source_id=citation.source_id,
                    message=f"Cited source is absent from this snapshot: {citation.source_id}",
                )
            )
            continue
        if citation.document_hash and citation.document_hash != document.content_hash:
            findings.append(
                Finding(
                    code="stale_source_version",
                    level="failure",
                    source_id=citation.source_id,
                    message="Citation hash refers to another document version.",
                )
            )
        if citation.quote is not None and citation.quote not in document.content:
            findings.append(
                Finding(
                    code="unsupported_quote",
                    level="failure",
                    source_id=citation.source_id,
                    message="The cited quote does not occur in the current source.",
                )
            )
        if citation.document_hash is None and citation.quote is None:
            findings.append(
                Finding(
                    code="unverified_citation",
                    level="unknown",
                    source_id=citation.source_id,
                    message="Source ID alone cannot verify its version or quoted content.",
                )
            )
    levels = {finding.level for finding in findings}
    status = "failed" if "failure" in levels else "unknown" if "unknown" in levels else "passed"
    return status, tuple(findings)


def compare_runs(
    before: Snapshot,
    after: Snapshot,
    cases: tuple[Case, ...],
    baseline: Run,
    candidate: Run,
    plan: Plan,
    *,
    allow_scoped_pass: bool = False,
) -> Report:
    validate_run(before, cases, baseline)
    validate_run(after, cases, candidate)
    expected_bindings = (
        plan.before_digest == before.digest,
        plan.after_digest == after.digest,
        plan.cases_digest == fingerprint([case.model_dump(mode="json") for case in cases]),
        plan.baseline_run_digest == fingerprint(baseline.model_dump(mode="json")),
    )
    if not all(expected_bindings):
        raise ValueError("plan inputs changed; regenerate the plan")
    all_ids = {case.case_id for case in cases}
    selected, skipped = set(plan.selected), set(plan.skipped)
    if (
        len(selected) != len(plan.selected)
        or len(skipped) != len(plan.skipped)
        or selected & skipped
        or selected | skipped != all_ids
        or not set(plan.controls) <= selected
        or not set(plan.impacted) <= selected
        or set(plan.controls) & set(plan.impacted)
        or (plan.mode == "full" and skipped)
    ):
        raise ValueError("plan must partition the suite into selected and skipped cases")
    if plan.changes != document_changes(before, after):
        raise ValueError("plan source changes do not match the snapshots")
    removed = frozenset(change.source_id for change in plan.changes if change.kind == "removed")
    baseline_by_id, candidate_by_id = baseline.by_id, candidate.by_id
    results = []
    for case in cases:
        old_observation = baseline_by_id.get(case.case_id)
        new_observation = candidate_by_id.get(case.case_id)
        if case.case_id not in plan.selected:
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    before_status="not_checked",
                    after_status="not_checked",
                    transition="not_checked",
                )
            )
            continue
        old_status, old_findings = inspect_observation(old_observation, before, case.before)
        new_status, new_findings = inspect_observation(
            new_observation, after, case.after, removed=removed
        )
        if new_status == "unknown" or old_status == "unknown":
            transition = "inconclusive"
        elif old_status == "passed" and new_status == "failed":
            transition = "regression"
        elif old_status == "failed" and new_status == "passed":
            transition = "improved"
        elif new_status == "failed":
            transition = "existing_failure"
        else:
            transition = "passed"
        results.append(
            CaseResult(
                case_id=case.case_id,
                question=case.question,
                before_status=old_status,
                after_status=new_status,
                transition=transition,
                before_findings=old_findings,
                after_findings=new_findings,
                before_observation=old_observation,
                after_observation=new_observation,
            )
        )
    summary = dict(Counter(result.transition for result in results))
    summary.update(
        total=len(cases),
        checked=len(plan.selected),
        skipped=len(plan.skipped),
        controls=len(plan.controls),
        impacted=len(plan.impacted),
        candidate_failures=sum(r.after_status == "failed" for r in results),
        unknown=sum(r.transition == "inconclusive" for r in results),
    )
    if summary["candidate_failures"]:
        decision = "failed"
    elif summary["unknown"]:
        decision = "inconclusive"
    elif plan.skipped:
        decision = "scoped_pass" if allow_scoped_pass else "inconclusive"
    else:
        decision = "passed"
    return Report(
        before_name=before.name,
        after_name=after.name,
        plan=plan,
        results=tuple(results),
        decision=decision,
        summary=summary,
    )
