"""Build a provenance graph, select impacted cases, and sample sentinel controls."""

from __future__ import annotations

import networkx as nx

from .models import Case, Cases, Change, Plan, Run, Snapshot, fingerprint


def document_changes(before: Snapshot, after: Snapshot) -> tuple[Change, ...]:
    old, new = before.by_id, after.by_id
    changes = []
    for source_id in sorted(old.keys() | new.keys()):
        old_hash = old[source_id].content_hash if source_id in old else None
        new_hash = new[source_id].content_hash if source_id in new else None
        if old_hash != new_hash:
            kind = "added" if old_hash is None else "removed" if new_hash is None else "modified"
            changes.append(
                Change(source_id=source_id, kind=kind, before_hash=old_hash, after_hash=new_hash)
            )
    return tuple(changes)


def validate_run(snapshot: Snapshot, cases: tuple[Case, ...], run: Run) -> None:
    if snapshot.digest != run.snapshot_digest:
        raise ValueError("run is bound to a different snapshot")
    questions = {case.case_id: case.question for case in cases}
    for observation in run.observations:
        if observation.case_id not in questions:
            raise ValueError(f"run contains unknown case {observation.case_id}")
        if observation.question != questions[observation.case_id]:
            raise ValueError(f"question changed for case {observation.case_id}; recapture the run")


def build_plan(
    before: Snapshot,
    after: Snapshot,
    cases: tuple[Case, ...],
    baseline: Run,
    *,
    mode: str = "full",
    audit_controls: int = 1,
) -> Plan:
    Cases(cases=cases)
    validate_run(before, cases, baseline)
    if mode not in {"full", "impact"}:
        raise ValueError("mode must be full or impact")
    if audit_controls < 0:
        raise ValueError("audit_controls must be non-negative")
    changes = document_changes(before, after)
    baseline_by_id = baseline.by_id
    graph = nx.DiGraph()
    reasons: dict[str, set[str]] = {case.case_id: set() for case in cases}
    for case in cases:
        query_node = ("case", case.case_id)
        graph.add_node(query_node)
        observation = baseline_by_id.get(case.case_id)
        dependencies = set(case.before.must_cite) | set(case.after.must_cite)
        if observation:
            dependencies.update(observation.retrieved_sources)
            dependencies.update(citation.source_id for citation in observation.citations)
        if not observation or observation.error or not dependencies:
            reasons[case.case_id].add("baseline_dependency_unknown")
        for source_id in dependencies:
            graph.add_edge(("source", source_id), query_node)

    fallback = False
    for change in changes:
        source_node = ("source", change.source_id)
        if change.kind == "added" or source_node not in graph:
            # A new/unseen source can affect queries absent from the provenance graph.
            fallback = True
        if source_node in graph:
            for _, case_id in nx.descendants(graph, source_node):
                reasons[case_id].add(f"{change.kind}:{change.source_id}")
    if fallback:
        for case in cases:
            reasons[case.case_id].add("new_or_unmapped_source_requires_full_retest")
    impacted = sorted(case_id for case_id, why in reasons.items() if why)
    all_ids = sorted(reasons)
    controls: list[str] = []
    if mode == "full":
        selected = all_ids
        for case_id in all_ids:
            reasons[case_id].add("full_retest")
    else:
        remaining = sorted(
            set(all_ids) - set(impacted),
            key=lambda case_id: fingerprint([before.digest, after.digest, case_id]),
        )
        controls = remaining[:audit_controls]
        for case_id in controls:
            reasons[case_id].add("sentinel_control")
        selected = sorted(set(impacted) | set(controls))
    return Plan(
        before_digest=before.digest,
        after_digest=after.digest,
        cases_digest=fingerprint([case.model_dump(mode="json") for case in cases]),
        baseline_run_digest=fingerprint(baseline.model_dump(mode="json")),
        mode=mode,
        changes=changes,
        selected=tuple(selected),
        impacted=tuple(impacted),
        controls=tuple(controls),
        skipped=tuple(sorted(set(all_ids) - set(selected))),
        reasons={case_id: tuple(sorted(why)) for case_id, why in sorted(reasons.items())},
    )
