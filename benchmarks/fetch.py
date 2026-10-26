"""Fetch pinned public archives; corpus text stays outside the Git repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def archive(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or not zipfile.is_zipfile(destination):
        subprocess.run(
            [
                "curl",
                "--http1.1",
                "--location",
                "--compressed",
                "--silent",
                "--show-error",
                "--fail",
                "--retry",
                "2",
                "--connect-timeout",
                "12",
                "--max-time",
                "240",
                "--max-filesize",
                "50000000",
                url,
                "-o",
                str(destination),
            ],
            check=True,
        )
    with zipfile.ZipFile(destination) as compressed:
        if compressed.testzip() is not None:
            raise ValueError(f"incomplete archive: {destination.name}")
    return {
        "url": url,
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def release(project: dict, directory: Path) -> dict:
    repository = project["repo"]
    refs = ["refs/tags/" + project[k] for k in ("before", "after")]
    output = subprocess.check_output(
        [
            "git",
            "ls-remote",
            "https://github.com/" + repository + ".git",
            *[suffix for ref in refs for suffix in (ref, ref + "^{}")],
        ],
        text=True,
        timeout=45,
    )
    resolved = {ref: sha for sha, ref in (line.split() for line in output.splitlines())}
    snapshots = {}
    for label, ref in zip(("before", "after"), refs):
        commit = resolved.get(ref + "^{}", resolved.get(ref))
        if not commit:
            raise ValueError(f"missing release ref {repository}:{ref}")
        path = directory / (repository.replace("/", "--") + "--" + label + ".zip")
        snapshots[label] = {
            "tag": project[label],
            "commit": commit,
            **archive(f"https://codeload.github.com/{repository}/zip/{commit}", path),
        }
    return {"repo": repository, **snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--releases-only", action="store_true")
    args = parser.parse_args()
    directory = args.data_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not args.releases_only:
        manifest["ragtruth"] = archive(
            "https://codeload.github.com/ParticleMedia/RAGTruth/zip/"
            + protocol["datasets"]["ragtruth"]["commit"],
            directory / "ragtruth.zip",
        )
        manifest["timelyqa"] = archive(
            "https://raw.githubusercontent.com/kaist-dmlab/TimelyRAG/"
            + protocol["datasets"]["timelyqa"]["commit"]
            + "/datasets.zip",
            directory / "timelyqa.zip",
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    projects = protocol["datasets"]["release_history"]["projects"]
    existing = {item["repo"]: item for item in manifest.get("releases", [])}
    pending = [p for p in projects if p["repo"] not in existing]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda project: release(project, directory), pending):
            existing[result["repo"]] = result
            manifest["releases"] = [existing[k] for k in sorted(existing)]
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "downloaded": result["repo"],
                        "archives": [result[k]["bytes"] for k in ("before", "after")],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
