"""Small CLI for snapshots, capture, impact planning and reproducible CI checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from .adapters import LightRAGBackend, capture
from .artifacts import read_model, snapshot_directory, write_model
from .evaluate import compare_runs
from .models import Cases, Model, Plan, Run, Snapshot, fingerprint
from .planner import build_plan
from .reporting import write_report
from .workflow import run_candidate


class Suite(Model):
    before_snapshot: str
    after_snapshot: str
    cases: str
    before_run: str
    after_run: str | None = None


def load_suite(path: Path):
    suite = read_model(path, Suite)
    root = path.parent
    return (
        suite,
        read_model(root / suite.before_snapshot, Snapshot),
        read_model(root / suite.after_snapshot, Snapshot),
        read_model(root / suite.cases, Cases).cases,
        read_model(root / suite.before_run, Run),
    )


def add_policy(parser):
    parser.add_argument("--mode", choices=("full", "impact"), default="full")
    parser.add_argument("--audit-controls", type=int, default=1)
    parser.add_argument(
        "--allow-scoped-pass",
        action="store_true",
        help="Allow exit 0 for a clean subset; report remains scoped_pass",
    )


def add_backend(parser):
    parser.add_argument("--url", required=True, help="LightRAG server base URL")
    parser.add_argument("--api-key-env", default="LIGHTRAG_API_KEY")
    parser.add_argument(
        "--retrieval-mode",
        choices=("local", "global", "hybrid", "naive", "mix", "bypass"),
        default="mix",
    )
    parser.add_argument(
        "--aliases", type=Path, help="JSON mapping of exact server file_path to source_id"
    )
    parser.add_argument("--timeout", type=float, default=120)


def backend(args):
    aliases = json.loads(args.aliases.read_text(encoding="utf-8")) if args.aliases else {}
    if not isinstance(aliases, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in aliases.items()
    ):
        raise ValueError("aliases must be a JSON object mapping paths to source ID strings")
    return LightRAGBackend(
        args.url,
        api_key=os.environ.get(args.api_key_env),
        mode=args.retrieval_mode,
        source_aliases=aliases,
        timeout=args.timeout,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rag-changelab", description=__doc__)
    root.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run seven offline reference and fault scenarios")
    demo.add_argument("--out", type=Path, default=Path("reports/demo"))
    snapshot = commands.add_parser(
        "snapshot", help="Snapshot UTF-8 .md/.txt files by relative path"
    )
    snapshot.add_argument("directory", type=Path)
    snapshot.add_argument("--name")
    snapshot.add_argument("--out", type=Path, required=True)
    for command in ("plan", "check", "run"):
        cmd = commands.add_parser(
            command,
            help={
                "plan": "Plan queries using corpus changes and baseline evidence",
                "check": "Replay captured runs and emit CI results",
                "run": "Capture current LightRAG answers with automatic sentinel escalation",
            }[command],
        )
        cmd.add_argument("suite", type=Path)
        cmd.add_argument("--out", type=Path, required=True)
        add_policy(cmd)
        if command == "run":
            add_backend(cmd)
    cap = commands.add_parser("capture", help="Read-only LightRAG /query capture")
    cap.add_argument("snapshot", type=Path)
    cap.add_argument("cases", type=Path)
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--plan", type=Path, help="Capture only this plan's selected candidate cases")
    add_backend(cap)
    return root


def execute(args) -> int:
    if args.command == "demo":
        from .demo import make_demo

        result = make_demo(args.out)
        print(
            json.dumps(
                {
                    "all_expected": result["all_expected"],
                    "scenarios": len(result["scenarios"]),
                    "report": str(args.out / "index.html"),
                }
            )
        )
        return 0 if result["all_expected"] else 1
    if args.command == "snapshot":
        snapshot = snapshot_directory(args.directory, name=args.name)
        write_model(args.out, snapshot)
        print(json.dumps({"digest": snapshot.digest, "documents": len(snapshot.documents)}))
        return 0
    if args.command == "capture":
        snapshot = read_model(args.snapshot, Snapshot)
        cases = read_model(args.cases, Cases).cases
        if args.plan:
            plan = read_model(args.plan, Plan)
            if (
                plan.after_digest != snapshot.digest
                or plan.cases_digest != fingerprint([c.model_dump(mode="json") for c in cases])
                or not set(plan.selected) <= {c.case_id for c in cases}
            ):
                raise ValueError("capture plan does not match snapshot or cases")
            cases = tuple(c for c in cases if c.case_id in plan.selected)
        run = capture(backend(args), snapshot, cases)
        write_model(args.out, run)
        errors = sum(bool(o.error) for o in run.observations)
        print(json.dumps({"captured": len(run.observations), "errors": errors}))
        return 3 if errors else 0
    suite, before, after, cases, baseline = load_suite(args.suite)
    if args.command == "plan":
        plan = build_plan(
            before, after, cases, baseline, mode=args.mode, audit_controls=args.audit_controls
        )
        write_model(args.out, plan)
        print(
            json.dumps(
                {
                    "selected": len(plan.selected),
                    "skipped": len(plan.skipped),
                    "plan": str(args.out),
                }
            )
        )
        return 0
    if args.command == "check":
        if not suite.after_run:
            raise ValueError("check requires after_run in the suite manifest")
        candidate = read_model(args.suite.parent / suite.after_run, Run)
        plan = build_plan(
            before, after, cases, baseline, mode=args.mode, audit_controls=args.audit_controls
        )
        report = compare_runs(
            before,
            after,
            cases,
            baseline,
            candidate,
            plan,
            allow_scoped_pass=args.allow_scoped_pass,
        )
    else:
        candidate, report = run_candidate(
            before,
            after,
            cases,
            baseline,
            backend(args),
            mode=args.mode,
            audit_controls=args.audit_controls,
            allow_scoped_pass=args.allow_scoped_pass,
        )
    path = write_report(args.out, report, before, after)
    if args.command == "run":
        write_model(args.out / "candidate.json", candidate)
    print(json.dumps({"decision": report.decision, **report.summary, "report": str(path)}))
    return report.exit_code


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (OSError, ValidationError, ValueError, TypeError, ImportError) as exc:
        # Input errors are distinct from failed checks and uncertain results.
        print(f"rag-changelab: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
