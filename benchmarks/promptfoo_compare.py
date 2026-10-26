"""Compare the required-text component against actual Promptfoo, without an LLM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from itertools import chain
from pathlib import Path

from contracts import boolean_fixtures, timely_fixtures
from data import ragtruth, timelyqa
from run import report, write_json

from rag_changelab.evaluate import inspect_observation
from rag_changelab.models import Expectation, Observation

ROOT = Path(__file__).resolve().parent


def export(directory, out):
    sources, responses, _ = ragtruth(directory)
    temporal, _ = timelyqa(directory)
    tests = []
    for fixture in chain(timely_fixtures(temporal), boolean_fixtures(sources, responses)):
        if fixture.split != "test":
            continue
        variants = {
            "valid_exact": fixture.answer,
            "valid_wrapper": "According to the source: " + fixture.answer,
            "valid_equivalent_rendering": fixture.equivalent,
            "negated_required_phrase": "The answer is NOT: " + fixture.answer,
            "contradictory_extra_claim": fixture.answer + "\nThe preceding answer is false.",
            "missing_required_answer": "Unspecified.",
        }
        for family, answer in variants.items():
            status, _ = inspect_observation(
                Observation(case_id=fixture.identifier, question=fixture.question, answer=answer),
                fixture.snapshot,
                Expectation(contains=(fixture.answer,), evidence_required=False),
            )
            tests.append(
                {
                    "description": fixture.identifier + ":" + family,
                    "vars": {"recorded_answer": answer, "required_text": fixture.answer},
                    "metadata": {
                        "family": family,
                        "group": fixture.group,
                        "changelab_status": status,
                    },
                    "assert": [{"type": "icontains", "value": "{{required_text}}"}],
                }
            )
    configuration = {
        "description": "Held-out text-contract parity; no LLM/model calls",
        "prompts": ["{{recorded_answer}}"],
        "providers": ["file://" + str(ROOT / "promptfoo-recorded.cjs")],
        "tests": tests,
        "sharing": False,
        "evaluateOptions": {"cache": False, "showProgressBar": False, "maxConcurrency": 8},
    }
    write_json(out, configuration)
    print(json.dumps({"exported": len(tests), "configuration": str(out)}))


def compare(raw, destination, configuration=None):
    payload = json.loads(raw.read_text())
    results = payload["results"]["results"]
    configuration = configuration or raw.parent / "promptfoo-config.json"
    tests = json.loads(configuration.read_text())["tests"]
    if len(results) != len(tests):
        raise ValueError("Promptfoo did not score every exported test exactly once")
    if payload["results"]["stats"]["errors"]:
        raise ValueError("Promptfoo returned execution errors, not assertion failures")
    indices = [result["testIdx"] for result in results]
    if set(indices) != set(range(len(tests))):
        raise ValueError("Promptfoo test indices are missing or duplicated")
    outcomes = []
    for result in results:
        original = tests[result["testIdx"]]
        if original["description"] != result["testCase"]["description"]:
            raise ValueError("Promptfoo result does not match exported test identity")
        # Promptfoo redacts hash-shaped metadata as possible secrets. Rejoin to
        # the original local test index rather than collapsing all groups.
        metadata = original["metadata"]
        actual = "passed" if result["success"] else "failed"
        outcomes.append(
            {
                "expected": metadata["changelab_status"],
                "actual": actual,
                "family": metadata["family"],
                "group": metadata["group"],
            }
        )
    comparison = {
        "version": "0.125.1",
        "component": "required-text icontains only",
        "observations": len(outcomes),
        "source_groups": len({r["group"] for r in outcomes}),
        "status_mismatches": sum(r["expected"] != r["actual"] for r in outcomes),
        "execution_errors": payload["results"]["stats"]["errors"],
        "model_tokens": payload["results"]["stats"]["tokenUsage"]["total"],
        "by_family": dict(Counter(r["family"] for r in outcomes)),
        "limits": "Recorded controlled outputs; no model grading, Ragas comparison, citation/provenance scoring, or generation benchmark",
    }
    summary = json.loads((destination / "summary.json").read_text())
    summary["promptfoo_comparison"] = comparison
    write_json(destination / "summary.json", summary)
    report(summary, destination)
    print(json.dumps(comparison))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--runtime", type=Path, help="Directory containing installed promptfoo@0.125.1"
    )
    parser.add_argument(
        "--scratch", type=Path, help="Local raw output directory outside published report"
    )
    parser.add_argument("--out", type=Path, default=Path("validation"))
    args = parser.parse_args()
    if args.runtime:
        runtime = args.runtime.resolve()
        package = json.loads((runtime / "node_modules/promptfoo/package.json").read_text())
        if package["version"] != "0.125.1":
            raise ValueError("comparison is pinned to Promptfoo 0.125.1")
        scratch = (args.scratch or args.out.parent / "promptfoo-raw").resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        configuration = scratch / "promptfoo-config.json"
        raw = scratch / "promptfoo-results.json"
        export(args.data_dir.resolve(), configuration)
        environment = {
            **os.environ,
            "PROMPTFOO_CONFIG_DIR": str(scratch / "state"),
            "PROMPTFOO_DISABLE_TELEMETRY": "1",
            "PROMPTFOO_DISABLE_UPDATE": "1",
            "FORCE_COLOR": "0",
        }
        result = subprocess.run(
            [
                str(runtime / "node_modules/.bin/promptfoo"),
                "eval",
                "-c",
                str(configuration),
                "-o",
                str(raw),
                "--no-cache",
                "--no-write",
                "--no-share",
                "--no-progress-bar",
                "--no-table",
            ],
            env=environment,
        )
        if result.returncode not in {0, 100}:
            raise ValueError(f"Promptfoo execution failed with exit {result.returncode}")
        compare(raw, args.out.resolve(), configuration)
    elif args.export:
        export(args.data_dir.resolve(), args.export.resolve())
    else:
        compare(
            args.results.resolve(),
            args.out.resolve(),
            args.config.resolve() if args.config else None,
        )


if __name__ == "__main__":
    main()
