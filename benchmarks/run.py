"""Run the frozen external-validity audit without model keys or invented evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import subprocess
from collections import Counter
from itertools import chain
from pathlib import Path

from contracts import boolean_fixtures, timely_fixtures
from contracts import evaluate as evaluate_contracts
from data import ragtruth, timelyqa
from metrics import summarize
from ragtruth import evaluate as evaluate_ragtruth
from releases import evaluate as evaluate_releases

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def table(headers, rows):
    def esc(value):
        return html.escape(str(value), quote=True)

    return (
        "<table><thead><tr>"
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table>"
    )


def percent(value):
    return "unscored" if value is None else f"{100 * value:.1f}%"


def report(summary, directory):
    controlled = summary["controlled_contracts"]
    temporal = summary["inventory"]["timelyqa"]
    truth = summary["inventory"]["ragtruth"]
    release = summary["release_history"]
    test = controlled["by_split"]["test"]
    corpus_rows = [
        [
            "RAGTruth",
            truth["source_instances"],
            truth["model_responses"],
            "Human-annotated recorded model outputs",
            "450 held-out source groups; six correlated responses each",
        ],
        [
            "TimelyQABench",
            temporal["unique_normalized_questions"],
            str(temporal["deduplicated_context_cases"]) + " usable temporal contexts / 4 domains",
            "External synthetic temporal QA",
            "Exact context dedup + numeric-masked query group splits",
        ],
        [
            "Real release history",
            release["total_baseline_probes"],
            "6 projects / 12 pinned snapshots",
            "Human-authored docs; generated retrieval probes",
            "Document-family groups; no generated-answer quality claim",
        ],
    ]
    family_rows = [
        [
            name,
            value["observations"],
            value["groups"],
            percent(value["status_accuracy"]),
            value["fn"],
            value["fp"],
            percent(value["unknown_rate"]),
        ]
        for name, value in controlled["by_family_test"].items()
    ]
    release_rows = [
        [
            r["project"],
            r["retriever"],
            r["queries"],
            r["selected"],
            percent(r["call_reduction"]),
            percent(r["evidence_change_coverage"]),
            "yes" if r["full_fallback"] else "no",
        ]
        for r in release["comparisons"]
        if r["condition"] == "complete_release" and r["sentinels"] == 5
    ]
    temporal_rows = [
        [
            domain,
            values["raw_questions"],
            values["unique_normalized_queries"],
            values["numeric_masked_query_groups"],
            values["future_dated_gold_questions"],
            values["unknown_gold_availability_time_questions"],
            values["duplicate_query_groups_with_different_answers"],
        ]
        for domain, values in temporal["domains"].items()
    ]
    baseline = summary["ragtruth_external"]
    incremental = summary.get("incremental_history_exploratory")
    incremental_html = ""
    if incremental:
        cohort = incremental["primary_cohort_summary"]
        incremental_html = f"""<h2>Exploratory follow-up: actual small commits</h2>
<p>This cohort was defined <em>after</em> the primary release results, to check workload representativeness.
It does not replace that frozen outcome. It uses {
            incremental["update_events"]
        } actual parent/commit pairs
from Requests and HTTPX, {incremental["unique_generated_probe_ids"]} unique probes,
and {incremental["total_baseline_probes"]} query/snapshot replays per retriever.</p>
<div class="scroll">{
            table(
                ["Metric", "Observed"],
                [
                    ["Median calls saved", percent(cohort["median_call_reduction"])],
                    [
                        "Pooled evidence-change coverage",
                        percent(cohort["pooled_evidence_change_coverage"]),
                    ],
                    [
                        "Median per-comparison coverage",
                        percent(cohort["median_evidence_change_coverage"]),
                    ],
                    [
                        "Worst single-comparison coverage",
                        percent(incremental["primary_min_evidence_change_coverage"]),
                    ],
                    [
                        "Comparisons with ≥20% savings and ≥95% coverage",
                        str(cohort["comparisons_with_20pct_saving_and_95pct_coverage"])
                        + " / "
                        + str(cohort["retriever_update_comparisons"]),
                    ],
                    [
                        "Plain reverse-index selection matches",
                        incremental["all_plain_reverse_index_selections_match"],
                    ],
                ],
            )
        }</div>
<p>Local retesting shows a useful efficiency signal, but it can miss evidence changes outside old dependencies.
This experiment does not establish semantic regression recall or a distinctive algorithmic advantage.</p>"""
    promptfoo = summary.get("promptfoo_comparison")
    promptfoo_html = ""
    if promptfoo:
        promptfoo_html = f"""<h2>Actual Promptfoo comparison</h2><p>Promptfoo {promptfoo["version"]},
<code>icontains</code>, {promptfoo["observations"]} held-out recorded-output checks:
{promptfoo["status_mismatches"]} differences from ChangeLab's required-text result.
This compares the text-contract component only, not semantic evaluators or full system capabilities.</p>"""
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ChangeLab external validity audit</title>
<style>body{{margin:0;background:#f5f5ed;color:#152b30;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:40px 26px 70px}}h1{{font:42px/1.15 Georgia,serif}}h2{{font:27px Georgia,serif;margin-top:36px}}.tag{{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:#205d59}}.warning{{padding:20px;border:1px solid #a53b31;background:#fff9f2}}.metrics{{display:flex;gap:35px;flex-wrap:wrap;margin:30px 0}}.metrics strong{{font:32px Georgia,serif;display:block}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:#fffef8;font-size:13px}}td,th{{padding:11px;border-bottom:1px solid #c8d1c7;text-align:left;vertical-align:top}}a{{color:#205d59}}code{{overflow-wrap:anywhere}}footer{{border-top:1px solid #c8d1c7;padding-top:20px;margin-top:40px}}details{{margin:20px 0}}.muted{{color:#536b70}}</style></head><body><main>
<div class="tag">RAG ChangeLab / external validity audit / frozen v1</div>
<h1>Does the implementation earn its complexity?</h1>
<div class="warning"><strong>Scheduler continuation gate: {"MET" if release["scheduler_continuation_gate"] else "NOT MET"}.</strong>
The gate requires at least 20% median call reduction with at least 95% evidence-change coverage on complete real release changes.
Semantic answer validation and model-backed LightRAG deployment remain unvalidated.</div>
<div class="metrics"><div><strong>{truth["model_responses"]:,}</strong>recorded model responses</div>
<div><strong>{temporal["unique_normalized_questions"]:,}</strong>distinct normalized temporal questions</div>
<div><strong>{controlled["all_splits"]["observations"]:,}</strong>controlled observations; correlated variants</div>
<div><strong>{release["total_baseline_probes"]:,}</strong>real-doc retrieval probes</div></div>
<p>Counts are not interchangeable. The unit of independence is a source/query group, never each fault variant or each model response.
The core implementation was held fixed at <code>{summary["implementation"]}</code>.</p>
<h2>Data coverage and provenance</h2><div class="scroll">{table(["Layer", "Independent inputs / probes", "Records / snapshots", "What it is", "Controls"], corpus_rows)}</div>
<h2>Held-out contract behavior</h2><p>{test["observations"]:,} observations from {test["groups"]:,} source/query groups.
These are controlled rule checks. They are not a measured real-model hallucination-detection rate.</p>
<div class="scroll">{table(["Family", "Observations", "Groups", "Expected-status match", "Missed failures", "False alarms", "Unknown"], family_rows)}</div>
<p class="muted">“Valid equivalent rendering” means whitespace-preserving rewrites for temporal answers and boolean paraphrases for business facts.
Negated/contradictory replies retain the required phrase. These expose the difference between literal contracts and semantic correctness.</p>
<h2>Actual release-change scheduling</h2><p>Primary setting: top-3 retrieval, five sentinel controls, complete version changes.
Median call reduction: <strong>{percent(release["primary_median_call_reduction"])}</strong>.
Plain reverse-index selection matches ChangeLab in every comparison: <strong>{release["all_plain_reverse_index_selections_match"]}</strong>.</p>
<div class="scroll">{table(["Project", "Retriever", "Probes", "Selected", "Calls saved", "Evidence-change coverage", "Full fallback"], release_rows)}</div>
<p class="muted">Evidence changes compare actual full-retest source/version signatures. A changed signature is not automatically an incorrect answer.
The JSON also contains a common-files-only ablation, zero/one/five/twenty sentinels, same-budget random and keyword baselines.</p>
{incremental_html}{promptfoo_html}<h2>RAGTruth external labels</h2><p>ChangeLab semantic score: <strong>UNSCORED / unsupported input contract</strong>.
The corpus has free-form answers without trustworthy answer-citation metadata or caller-defined answer contracts.
We do not invent citations, treat unsupported formats as hallucinations, or use gold annotation spans as assertions.</p>
<p>A simple token-novelty comparator was calibrated only on official training data; on {baseline["test"]["observations"]:,}
good-quality official test answers it has failure precision {percent(baseline["test"]["precision"])},
recall {percent(baseline["test"]["recall"])}, false-positive rate {percent(baseline["test"]["false_positive_rate"])}.
This comparator is not Ragas or Promptfoo and establishes no superiority over those systems.</p>
<h2>Data-quality quarantine</h2><div class="scroll">{table(["Domain", "Raw questions", "Unique questions", "Numeric-masked groups", "Future-dated gold", "Unknown gold time", "Repeated questions with differing answers"], temporal_rows)}</div>
<p>TimelyQABench is synthetic, not actual policy/legal history. Future-dated gold is flagged for availability review;
those rows cannot support an as-of deployment claim. Corrupt and relative timestamps stay unknown. Different temporal contexts can legitimately have different answers to the same question; they remain correlated, not independent samples.</p>
<h2>What remains open</h2><ul><li>No model-backed LightRAG deployment or actual generation token/latency costs measured.</li>
<li>No semantic verifier added to rescue the frozen implementation after seeing its results.</li>
<li>No user-adoption evidence; passing engineering tests does not satisfy independent product value.</li></ul>
<footer><a href="summary.json">Full results and group bootstrap intervals</a> · <a href="contract-index.jsonl.gz">Controlled case index</a> ·
<a href="ragtruth-index.jsonl.gz">Recorded-answer outcomes</a> · <a href="retrieval-index.jsonl.gz">Retrieval decisions</a> ·
<a href="manifest.json">Pinned archive hashes</a><br>
Sources: <a href="https://github.com/ParticleMedia/RAGTruth">RAGTruth</a> ·
<a href="https://github.com/kaist-dmlab/TimelyRAG">TimelyQABench</a> ·
<a href="https://github.com/BohanSu/rag-changelab/blob/893d08d/benchmarks/protocol.json">Frozen protocol</a> ·
<a href="incremental-index.jsonl.gz">Small-commit decisions (when executed)</a></footer></main></body></html>"""
    (directory / "index.html").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("validation"))
    args = parser.parse_args()
    directory, destination = args.data_dir.resolve(), args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            protocol["evaluated_implementation"],
            "--",
            "src/rag_changelab",
        ],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    for artifact in [
        manifest["ragtruth"],
        manifest["timelyqa"],
        *[r[k] for r in manifest["releases"] for k in ("before", "after")],
    ]:
        actual = hashlib.sha256((directory / artifact["file"]).read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise ValueError(f"archive hash mismatch: {artifact['file']}")
    write_json(destination / "manifest.json", manifest)
    sources, responses, truth_inventory = ragtruth(directory)
    temporal_rows, temporal_inventory = timelyqa(directory)
    assert (
        truth_inventory["source_instances"] == 2965 and truth_inventory["model_responses"] == 17790
    )
    assert sum(v["raw_questions"] for v in temporal_inventory["domains"].values()) == 12000
    print("inventory verified; frozen core unchanged", flush=True)
    with gzip.open(destination / "contract-index.jsonl.gz", "wt", encoding="utf-8") as output:
        outcomes = evaluate_contracts(
            chain(timely_fixtures(temporal_rows), boolean_fixtures(sources, responses)), output
        )
    controlled = summarize(outcomes)
    controlled["plain_python_mismatches"] = sum(r["actual"] != r["plain_python"] for r in outcomes)
    controlled["unique_source_cases"] = len({r["source_case"] for r in outcomes})
    controlled["category_counts_test"] = dict(
        Counter(r["domain"] + ":" + r["category"] for r in outcomes if r["split"] == "test")
    )
    print(
        f"controlled checks: {len(outcomes)} observations, {controlled['unique_source_cases']} source cases",
        flush=True,
    )
    with gzip.open(destination / "ragtruth-index.jsonl.gz", "wt", encoding="utf-8") as output:
        external = evaluate_ragtruth(sources, responses, output)
    print("external human-label comparator complete", flush=True)
    with gzip.open(destination / "retrieval-index.jsonl.gz", "wt", encoding="utf-8") as output:
        releases = evaluate_releases(directory, manifest, output)
    summary = {
        "protocol": protocol["name"],
        "implementation": protocol["evaluated_implementation"],
        "inventory": {"ragtruth": truth_inventory, "timelyqa": temporal_inventory},
        "controlled_contracts": controlled,
        "ragtruth_external": external,
        "release_history": releases,
        "model_backed_lightrag": {
            "executed": False,
            "reason": "No provider or local language model was configured for this audit; actual deployment experiment not executed",
        },
        "no_aggregate_accuracy": "Contract faults, semantic annotations and evidence-change coverage are different targets",
    }
    write_json(destination / "summary.json", summary)
    report(summary, destination)
    print(
        json.dumps(
            {
                "contract_observations": len(outcomes),
                "source_cases": controlled["unique_source_cases"],
                "heldout_contract_observations": controlled["by_split"]["test"]["observations"],
                "release_probes": releases["total_baseline_probes"],
                "scheduler_gate": releases["scheduler_continuation_gate"],
                "report": str(destination / "index.html"),
            }
        )
    )


if __name__ == "__main__":
    main()
