"""An executable knowledge update, not pre-baked evaluation outcomes."""

from __future__ import annotations

from pathlib import Path

from .adapters import ExtractiveBackend, capture
from .artifacts import write_model
from .evaluate import compare_runs
from .models import Case, Cases, Citation, Document, Expectation, Observation, Run, Snapshot
from .planner import build_plan
from .reporting import STYLE, esc, write_report
from .workflow import run_candidate


def fixtures() -> tuple[Snapshot, Snapshot, tuple[Case, ...]]:
    texts = {
        "retention.md": "Audit log retention is 30 days.",
        "authentication.md": "Authentication uses API tokens.",
        "legacy.md": "Legacy export route is /v1/export-legacy.",
        "upload.md": "Upload size limit is 10 MB.",
        "release.md": "SDK release version is 1.2.0.",
        "regions.md": "Data regions are eu-west-1 and us-east-1.",
        "support.md": "Support hours are 09:00 to 18:00 UTC.",
        "encryption.md": "Encryption uses AES-256.",
    }
    changed = {
        **texts,
        "retention.md": "Audit log retention is 7 days.",
        "authentication.md": "Authentication uses OIDC.",
    }
    del changed["legacy.md"]
    before = Snapshot(
        name="Product handbook v1",
        documents=tuple(
            Document(source_id=key, content=value) for key, value in sorted(texts.items())
        ),
    )
    after = Snapshot(
        name="Product handbook v2",
        documents=tuple(
            Document(source_id=key, content=value) for key, value in sorted(changed.items())
        ),
    )
    rows = [
        ("retention", "How long is audit log retention?", "retention.md", "30 days", "7 days"),
        (
            "authentication",
            "What authentication method is used?",
            "authentication.md",
            "API tokens",
            "OIDC",
        ),
        ("legacy", "What is the legacy export route?", "legacy.md", "/v1/export-legacy", None),
        ("upload", "What is the upload size limit?", "upload.md", "10 MB", "10 MB"),
        ("release", "What is the SDK release version?", "release.md", "1.2.0", "1.2.0"),
        ("regions", "Which data regions are available?", "regions.md", "eu-west-1", "eu-west-1"),
        (
            "support",
            "What are the support hours?",
            "support.md",
            "09:00 to 18:00 UTC",
            "09:00 to 18:00 UTC",
        ),
        ("encryption", "What encryption is used?", "encryption.md", "AES-256", "AES-256"),
    ]
    cases = []
    for case_id, question, source_id, old, new in rows:
        before_expected = Expectation(contains=(old,), must_cite=(source_id,))
        if new is None:
            after_expected = Expectation(
                contains=("No evidence available.",), excludes=(old,), evidence_required=False
            )
        else:
            after_expected = Expectation(
                contains=(new,), excludes=(old,) if old != new else (), must_cite=(source_id,)
            )
        cases.append(
            Case(case_id=case_id, question=question, before=before_expected, after=after_expected)
        )
    return before, after, tuple(cases)


def inject(run: Run, replacement: Observation) -> Run:
    return run.model_copy(
        update={
            "observations": tuple(
                replacement if o.case_id == replacement.case_id else o for o in run.observations
            )
        }
    )


def make_demo(directory: str | Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    before, after, cases = fixtures()
    baseline = capture(ExtractiveBackend(before), before, cases)
    healthy = capture(ExtractiveBackend(after), after, cases)
    for name, value in (
        ("before.json", before),
        ("after.json", after),
        ("cases.json", Cases(cases=cases)),
        ("baseline.json", baseline),
    ):
        write_model(directory / "fixtures" / name, value)
    for label, snapshot in (("before", before), ("after", after)):
        for document in snapshot.documents:
            path = directory / "fixtures" / label / document.source_id
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(document.content, encoding="utf-8")

    no_citation = healthy.by_id["retention"].model_copy(update={"citations": ()})
    invalid_quote = healthy.by_id["retention"].model_copy(
        update={
            "citations": (
                Citation(
                    source_id="retention.md",
                    document_hash=after.by_id["retention.md"].content_hash,
                    quote="Audit log retention is 999 years.",
                ),
            )
        }
    )
    variants = [
        ("healthy", "Intentional fact changes pass the new contracts", healthy, "passed"),
        (
            "stale-answer",
            "An old answer still cites the old document version",
            inject(healthy, baseline.by_id["retention"]),
            "failed",
        ),
        (
            "citation-loss",
            "The new answer is correct but loses its required evidence",
            inject(healthy, no_citation),
            "failed",
        ),
        (
            "deleted-source",
            "The answer still depends on a removed source",
            inject(healthy, baseline.by_id["legacy"]),
            "failed",
        ),
        (
            "unsupported-quote",
            "A citation invents text absent from the current source",
            inject(healthy, invalid_quote),
            "failed",
        ),
    ]
    outcomes = []
    for slug, title, candidate, expected in variants:
        plan = build_plan(before, after, cases, baseline)
        report = compare_runs(before, after, cases, baseline, candidate, plan)
        write_report(directory / slug, report, before, after)
        write_model(directory / slug / "candidate.json", candidate)
        write_model(
            directory / slug / "suite.json",
            {
                "before_snapshot": "../fixtures/before.json",
                "after_snapshot": "../fixtures/after.json",
                "cases": "../fixtures/cases.json",
                "before_run": "../fixtures/baseline.json",
                "after_run": "candidate.json",
            },
        )
        outcomes.append(
            {
                "scenario": slug,
                "description": title,
                "expected": expected,
                "actual": report.decision,
                "checked": report.summary["checked"],
                "regressions": report.summary.get("regression", 0),
            }
        )

    candidate, scoped = run_candidate(
        before, after, cases, baseline, ExtractiveBackend(after), mode="impact", audit_controls=1
    )
    write_report(directory / "impact-scope", scoped, before, after)
    write_model(directory / "impact-scope" / "candidate.json", candidate)
    outcomes.append(
        {
            "scenario": "impact-scope",
            "description": "Three impacted queries plus one sentinel; four cases remain unchecked",
            "expected": "inconclusive",
            "actual": scoped.decision,
            "checked": scoped.summary["checked"],
            "regressions": 0,
        }
    )

    plan = build_plan(before, after, cases, baseline, mode="impact", audit_controls=1)
    sentinel = plan.controls[0]

    class BrokenControl:
        name = "fault-injected-extractive-reference"

        def observe(self, case: Case, snapshot: Snapshot) -> Observation:
            observation = healthy.by_id[case.case_id]
            return (
                observation.model_copy(update={"citations": ()})
                if case.case_id == sentinel
                else observation
            )

    candidate, expanded = run_candidate(
        before, after, cases, baseline, BrokenControl(), mode="impact", audit_controls=1
    )
    write_report(directory / "sentinel-escalation", expanded, before, after)
    write_model(directory / "sentinel-escalation" / "candidate.json", candidate)
    outcomes.append(
        {
            "scenario": "sentinel-escalation",
            "description": "A failing sentinel expands the initial selection to all eight queries",
            "expected": "failed",
            "actual": expanded.decision,
            "checked": expanded.summary["checked"],
            "regressions": expanded.summary.get("regression", 0),
        }
    )
    result = {
        "reference_backend": "scikit-learn TF-IDF, extractive; no LLM used",
        "cases": len(cases),
        "changed_documents": len(plan.changes),
        "all_expected": all(o["expected"] == o["actual"] for o in outcomes),
        "scenarios": outcomes,
    }
    write_model(directory / "verification.json", result)
    cards = "".join(
        f'<details open><summary><a href="{o["scenario"]}/index.html">'
        f'{esc(o["scenario"])}</a> <span class="badge {o["actual"]}">{o["actual"]}</span>'
        f'</summary><p>{esc(o["description"])}</p><p class="muted">'
        f"{o['checked']}/8 cases checked · {o['regressions']} new regressions</p></details>"
        for o in outcomes
    )
    (directory / "index.html").write_text(
        f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG ChangeLab · Working demo</title><style>{STYLE}</style></head><body><main>
<div class="eyebrow">RAG ChangeLab / Reproducible demo</div>
<h1>Knowledge changes.<br>Does your RAG keep up?</h1>
<p class="muted">Two handbook versions · 8 questions · 3 document changes · 7 scenarios</p>
<div class="summary"><div class="metric"><strong>{sum(o["actual"] == o["expected"] for o in outcomes)} / {len(outcomes)}</strong><span>expected outcomes</span></div>
<div class="metric"><strong>4</strong><span>distinct evidence faults</span></div>
<div class="metric"><strong>0</strong><span>API keys or model downloads</span></div></div>
<p>The reference backend actually retrieves from each snapshot using scikit-learn TF-IDF.
Faults are injected into its captured answers. These results validate the regression workflow,
not the quality of an LLM or a live LightRAG deployment.</p>
<h2>Open an evidence comparison</h2>{cards}
<footer><a href="verification.json">Verification data</a> ·
<a href="healthy/suite.json">Replayable suite</a><br>
Impact selection is a heuristic. A clean sample does not establish a global pass.
Checks cover explicit answer text, source versions and quotes, not semantic entailment.</footer>
</main></body></html>""",
        encoding="utf-8",
    )
    return result
