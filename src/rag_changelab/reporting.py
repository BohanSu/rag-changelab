"""Standalone, escaped HTML evidence reports and machine-readable CI artifacts."""

from __future__ import annotations

import difflib
import html
from pathlib import Path

from .artifacts import write_model
from .models import Report, Snapshot

STYLE = """
:root{color-scheme:light;--ink:#152b30;--muted:#536b70;--paper:#f5f5ed;--line:#c8d1c7;
--accent:#205d59;--bad:#a53b31}*{box-sizing:border-box}body{margin:0;background:var(--paper);
color:var(--ink);font:15px/1.6 ui-sans-serif,system-ui,sans-serif}main{max-width:1180px;margin:auto;
padding:48px 30px 80px}h1{font:500 48px/1.1 Georgia,serif;margin:16px 0}h2{font:500 26px Georgia,serif}
.eyebrow{letter-spacing:.16em;font-size:12px;text-transform:uppercase;color:var(--accent)}
.muted{color:var(--muted)}.summary{display:flex;gap:30px;flex-wrap:wrap;padding:24px 0;
border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:30px 0}
.metric strong{display:block;font:500 32px Georgia,serif}.metric span{color:var(--muted);font-size:13px}
.badge{display:inline-block;border:1px solid currentColor;border-radius:4px;padding:2px 9px;
font:600 12px/1.7 ui-monospace,monospace}.failed,.regression,.existing_failure{color:var(--bad)}
.passed,.improved{color:var(--accent)}.inconclusive,.unknown,.not_checked{color:#85621e}
details{border-top:1px solid var(--line);padding:15px 0}summary{cursor:pointer;display:list-item}
summary .badge{margin-right:10px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}
.panel{min-width:0;background:#fffef8;border:1px solid var(--line);padding:18px;border-radius:5px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,SFMono-Regular,monospace}
blockquote{margin:12px 0;padding-left:14px;border-left:3px solid var(--line);white-space:pre-wrap}
code{font:12px ui-monospace,monospace;overflow-wrap:anywhere}.finding{border-left:3px solid var(--bad);
padding-left:12px;margin:12px 0}.finding.unknown{border-color:#85621e}footer{margin-top:40px;
border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted)}a{color:var(--accent)}
@media(max-width:700px){main{padding:28px 18px}h1{font-size:35px}.pair{grid-template-columns:1fr}}
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def render_report(report: Report, before: Snapshot, after: Snapshot) -> str:
    metrics = [
        (report.summary["checked"], "cases checked"),
        (report.summary.get("regression", 0), "new regressions"),
        (report.summary["unknown"], "inconclusive comparisons"),
        (report.summary["skipped"], "cases not checked"),
    ]
    metric_html = "".join(
        f'<div class="metric"><strong>{count}</strong><span>{label}</span></div>'
        for count, label in metrics
    )
    source_details = []
    for change in report.plan.changes:
        old = before.by_id.get(change.source_id)
        new = after.by_id.get(change.source_id)
        diff = "".join(
            difflib.unified_diff(
                old.content.splitlines(keepends=True) if old else [],
                new.content.splitlines(keepends=True) if new else [],
                fromfile="before/" + change.source_id,
                tofile="after/" + change.source_id,
            )
        )
        source_details.append(
            f"<details><summary>{esc(change.kind)} · {esc(change.source_id)}</summary>"
            f"<pre>{esc(diff)}</pre></details>"
        )
    cards = []
    for result in report.results:
        panels = []
        for label, status, observation, findings in (
            ("Before", result.before_status, result.before_observation, result.before_findings),
            ("After", result.after_status, result.after_observation, result.after_findings),
        ):
            answer = observation.answer if observation else "Not captured / not checked"
            citations = ""
            if observation:
                citations = "".join(
                    f"<p><code>{esc(c.source_id)}</code></p>"
                    f"<blockquote>{esc(c.quote or 'No quote supplied')}</blockquote>"
                    f"<code>{esc(c.document_hash or 'Version hash not supplied')}</code>"
                    for c in observation.citations
                )
            findings_html = "".join(
                f'<div class="finding {esc(f.level)}"><strong>{esc(f.code)}</strong><br>'
                f"{esc(f.message)}</div>"
                for f in findings
            )
            panels.append(
                f'<section class="panel"><div class="eyebrow">{label} · {esc(status)}</div>'
                f"<pre>{esc(answer)}</pre>{findings_html}{citations}</section>"
            )
        reasons = ", ".join(report.plan.reasons.get(result.case_id, ())) or "Outside selected scope"
        opened = " open" if result.after_status in {"failed", "unknown"} else ""
        cards.append(
            f'<details{opened}><summary><span class="badge {esc(result.transition)}">'
            f"{esc(result.transition)}</span>{esc(result.case_id)} · {esc(result.question)}</summary>"
            f'<p class="muted">Selection: {esc(reasons)}</p>'
            f'<div class="pair">{"".join(panels)}</div></details>'
        )
    coverage_note = (
        "All suite cases were checked."
        if not report.plan.skipped
        else "This is a scoped run. Unchecked queries can still contain regressions."
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>RAG ChangeLab</title>
<style>{STYLE}</style></head><body><main><div class="eyebrow">RAG ChangeLab / Evidence regression</div>
<h1>What changed in your knowledge?</h1><p class="muted">{esc(before.name)} → {esc(after.name)}</p>
<span class="badge {esc(report.decision)}">{esc(report.decision)}</span>
<div class="summary">{metric_html}</div><p>{esc(coverage_note)}</p>
<p class="muted">Deterministic checks cover declared answer contracts, source hashes and quotes.
They do not certify semantic entailment or answer completeness.</p>
<h2>Source changes</h2>{"".join(source_details) or "<p>No document changes.</p>"}
<h2>Answer & evidence comparisons</h2>{"".join(cards)}
<footer>Snapshot before: <code>{esc(before.digest)}</code><br>
Snapshot after: <code>{esc(after.digest)}</code><br>
Plan: {esc(report.plan.mode)} · Sentinels: {len(report.plan.controls)} · Tool: {esc(report.tool_version)}<br>
<a href="report.json">Machine-readable report</a> · <a href="plan.json">Impact plan</a></footer>
</main></body></html>"""


def write_report(directory: str | Path, report: Report, before: Snapshot, after: Snapshot) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_model(directory / "report.json", report)
    write_model(directory / "plan.json", report.plan)
    path = directory / "index.html"
    path.write_text(render_report(report, before, after), encoding="utf-8")
    return path
