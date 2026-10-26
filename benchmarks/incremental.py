"""Exploratory actual-commit replay, separate from the frozen release experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

from releases import documentation, evaluate
from run import report, write_json

ROOT = Path(__file__).resolve().parent


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args])


def snapshot(repository, commit, target, repo):
    if not target.exists():
        lines = (
            git(repository, "ls-tree", "-r", "--format=%(objectname) %(path)", commit, "docs")
            .decode()
            .splitlines()
        )
        entries = [line.split(" ", 1) for line in lines]
        entries = [(sha, name) for sha, name in entries if name.endswith((".md", ".rst"))]
        blob_data = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "--batch"],
            input="".join(sha + "\n" for sha, _ in entries).encode(),
            check=True,
            capture_output=True,
        ).stdout
        stream = io.BytesIO(blob_data)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for _, name in entries:
                header = stream.readline().decode().strip().split()
                if len(header) != 3 or header[1] != "blob":
                    raise ValueError("missing expected Git blob")
                content = stream.read(int(header[2]))
                if stream.read(1) != b"\n":
                    raise ValueError("bad Git batch boundary")
                archive.writestr("source/" + name, content)
    return {
        "tag": commit[:12],
        "commit": commit,
        "file": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": target.stat().st_size,
        "url": f"https://github.com/{repo}/tree/{commit}/docs",
    }


def prepare(directory):
    original = json.loads((directory / "manifest.json").read_text())
    followup = json.loads((ROOT / "incremental-protocol.json").read_text())
    wanted = set(followup["projects"])
    updates = []
    for release in original["releases"]:
        repo = release["repo"]
        if repo not in wanted:
            continue
        history = directory / "history" / repo.split("/")[1]
        if not (history / ".git").is_dir():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-checkout",
                    "--single-branch",
                    "--depth",
                    "100",
                    "--branch",
                    release["after"]["tag"],
                    "--filter=blob:limit=200000",
                    f"https://github.com/{repo}.git",
                    str(history),
                ],
                check=True,
            )
        if git(history, "rev-parse", "HEAD").decode().strip() != release["after"]["commit"]:
            raise ValueError("history does not end at pinned release commit")
        commits = (
            git(history, "rev-list", "--first-parent", "HEAD", "--", "docs").decode().splitlines()
        )
        kept = 0
        for commit in commits:
            if kept >= 12:
                break
            parent = subprocess.run(
                ["git", "-C", str(history), "rev-parse", "--verify", commit + "^"],
                capture_output=True,
            )
            if parent.returncode:
                continue
            parent_id = parent.stdout.decode().strip()
            paths = []
            for label, revision in (("before", parent_id), ("after", commit)):
                target = (
                    directory / "history-snapshots" / f"{repo.replace('/', '--')}--{revision}.zip"
                )
                item = snapshot(history, revision, target, repo)
                item["file"] = str(target.relative_to(directory))
                paths.append(item)
            before = documentation(directory / paths[0]["file"], repo)
            after = documentation(directory / paths[1]["file"], repo)
            if before == after:
                continue
            updates.append({"repo": repo, "before": paths[0], "after": paths[1]})
            kept += 1
    result = {
        "releases": updates,
        "cohort": "exploratory actual small commits",
        "protocol": followup,
    }
    write_json(directory / "incremental-manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("validation"))
    args = parser.parse_args()
    directory, destination = args.data_dir.resolve(), args.out.resolve()
    manifest = prepare(directory)
    write_json(destination / "incremental-manifest.json", manifest)
    with gzip.open(destination / "incremental-index.jsonl.gz", "wt", encoding="utf-8") as output:
        results = evaluate(directory, manifest, output)
    results["status"] = (
        "Exploratory post-primary follow-up; does not replace full-release primary outcome"
    )
    summary = json.loads((destination / "summary.json").read_text())
    summary["incremental_history_exploratory"] = results
    write_json(destination / "summary.json", summary)
    report(summary, destination)
    print(
        json.dumps(
            {
                "update_events": results["update_events"],
                "unique_probes": results["unique_generated_probe_ids"],
                "query_snapshot_replays": results["total_baseline_probes"],
                "median_call_reduction": results["primary_median_call_reduction"],
                "min_evidence_change_coverage": results["primary_min_evidence_change_coverage"],
            }
        )
    )


if __name__ == "__main__":
    main()
