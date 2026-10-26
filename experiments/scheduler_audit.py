"""Seeded dependency-oracle audit: scheduling behavior, not retriever quality.

Each query has 1–3 known source dependencies. The backend returns those sources
directly, so no real retrieval/model performance is implied by the call counts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rag_changelab.adapters import capture
from rag_changelab.artifacts import write_model
from rag_changelab.models import Case, Citation, Document, Expectation, Observation, Snapshot
from rag_changelab.planner import build_plan
from rag_changelab.workflow import run_candidate


def audit(seed: int = 42, source_count: int = 50, query_count: int = 200) -> dict:
    if source_count < 10 or query_count < 20:
        raise ValueError("audit requires at least 10 sources and 20 queries")
    rng = random.Random(seed)
    source_ids = [f"source-{i:03d}.md" for i in range(source_count)]
    old = {source_id: f"{source_id} policy value is revision-0." for source_id in source_ids}
    new = {**old}
    for source_id in source_ids[:4]:
        new[source_id] = f"{source_id} policy value is revision-1."
    before = Snapshot(
        name="synthetic-before",
        documents=tuple(Document(source_id=k, content=v) for k, v in old.items()),
    )
    after = Snapshot(
        name="synthetic-after",
        documents=tuple(Document(source_id=k, content=v) for k, v in new.items()),
    )
    dependencies = {
        f"q-{i:04d}": tuple(sorted(rng.sample(source_ids, rng.randint(1, 3))))
        for i in range(query_count)
    }
    cases = tuple(
        Case(
            case_id=case_id,
            question=f"Summarize the policies for {', '.join(sources)}",
            before=Expectation(contains=tuple(old[s] for s in sources), must_cite=sources),
            after=Expectation(contains=tuple(new[s] for s in sources), must_cite=sources),
        )
        for case_id, sources in dependencies.items()
    )

    class DependencyOracle:
        name = "synthetic-dependency-oracle"

        def __init__(self, faults=()):
            self.faults = set(faults)
            self.calls = []

        def observe(self, case: Case, snapshot: Snapshot) -> Observation:
            self.calls.append(case.case_id)
            documents = [snapshot.by_id[s] for s in dependencies[case.case_id]]
            return Observation(
                case_id=case.case_id,
                question=case.question,
                answer="\n".join(d.content for d in documents),
                citations=()
                if case.case_id in self.faults
                else tuple(
                    Citation(source_id=d.source_id, document_hash=d.content_hash, quote=d.content)
                    for d in documents
                ),
                retrieved_sources=dependencies[case.case_id],
            )

    baseline = capture(DependencyOracle(), before, cases)
    plan = build_plan(before, after, cases, baseline, mode="impact", audit_controls=5)
    if not plan.impacted or not plan.controls or not plan.skipped:
        raise ValueError("this audit seed/size must leave impacted, control and skipped cases")
    scenarios = []

    def scenario(name, snapshot, faults, expected_decision, expected_calls, expected_failures):
        backend = DependencyOracle(faults)
        _, report = run_candidate(
            before, snapshot, cases, baseline, backend, mode="impact", audit_controls=5
        )
        observed = {
            "name": name,
            "decision": report.decision,
            "backend_calls": len(backend.calls),
            "unique_calls": len(set(backend.calls)),
            "unchecked": report.summary["skipped"],
            "detected_failures": report.summary["candidate_failures"],
            "injected_faults": len(faults),
            "final_mode": report.plan.mode,
        }
        observed["expected_behavior"] = (
            report.decision == expected_decision
            and len(backend.calls) == expected_calls
            and len(set(backend.calls)) == len(backend.calls)
            and report.summary["candidate_failures"] == expected_failures
        )
        scenarios.append(observed)

    scenario("localized-update", after, (), "inconclusive", len(plan.selected), 0)
    faults = plan.impacted[:3]
    scenario("faults-in-known-dependents", after, faults, "failed", len(plan.selected), len(faults))
    scenario(
        "bad-control-and-hidden-fault",
        after,
        (plan.controls[0], plan.skipped[0]),
        "failed",
        query_count,
        2,
    )
    scenario(
        "hidden-fault-outside-clean-sample",
        after,
        (plan.skipped[0],),
        "inconclusive",
        len(plan.selected),
        0,
    )
    with_new_source = Snapshot(
        name="synthetic-added-source",
        documents=after.documents
        + (Document(source_id="new-policy.md", content="A newly added source."),),
    )
    scenario("new-source-conservative-fallback", with_new_source, (), "passed", query_count, 0)
    return {
        "experiment": "synthetic dependency-oracle scheduling audit",
        "seed": seed,
        "source_count": source_count,
        "query_count": query_count,
        "modified_sources": 4,
        "initial_impacted": len(plan.impacted),
        "initial_controls": len(plan.controls),
        "initial_selected": len(plan.selected),
        "localized_call_reduction_percent": round(100 * (1 - len(plan.selected) / query_count), 1),
        "all_expected": all(s["expected_behavior"] for s in scenarios),
        "scenarios": scenarios,
        "limits": "Oracle dependencies; no measured real-world retrieval recall, latency, "
        "token cost or LLM quality. Hidden faults can evade a clean sample; "
        "unchecked coverage remains inconclusive by default.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/scheduler-audit.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = audit(seed=args.seed)
    write_model(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "scenarios"}))
    return 0 if result["all_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
