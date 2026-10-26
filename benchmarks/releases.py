"""Actual release documentation, generated section probes, full retrieval replay.

Retrieval signatures are measurable evidence changes, not labeled answer errors.
No LLM and no LightRAG server are substituted into these measurements.
"""

from __future__ import annotations

import re
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from data import digest, normalize, split
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_changelab.models import (
    Case,
    Citation,
    Document,
    Expectation,
    Observation,
    Run,
    Snapshot,
    fingerprint,
)
from rag_changelab.planner import build_plan, document_changes


def documentation(path: Path, repo: str) -> dict[str, str]:
    documents = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir() or "/" not in member.filename:
                continue
            relative = member.filename.split("/", 1)[1]
            if not relative.startswith("docs/") or not relative.endswith((".md", ".rst")):
                continue
            if repo == "fastapi/fastapi":
                if not relative.startswith("docs/en/docs/"):
                    continue
                relative = "docs/" + relative.removeprefix("docs/en/docs/")
            if any(
                word in relative.casefold()
                for word in (
                    "changelog",
                    "release-notes",
                    "release_notes",
                    "/changes.",
                    "/license.",
                    "/authors.",
                )
            ):
                continue
            content = archive.read(member).decode("utf-8")
            if len(content.strip()) >= 80:
                documents[relative] = content
    return documents


def headings(text: str) -> list[tuple[str, str]]:
    """Parse authored Markdown/RST section paths, ignoring fenced code headings."""
    lines = text.splitlines()
    stack = []
    result = []
    body = []
    fenced = False
    rst_levels = {}
    skip_line = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            if stack:
                body.append(line)
            continue
        title = None
        level = 0
        if not fenced:
            markdown = re.match(r"^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?$", line)
            if markdown:
                level, title = len(markdown[1]), markdown[2].strip()
            elif index + 1 < len(lines) and line.strip() and not line.startswith(" "):
                underline = lines[index + 1].strip()
                if (
                    len(underline) >= max(3, len(line.strip()))
                    and len(set(underline)) == 1
                    and underline[0] in '=-~^"'
                ):
                    character = underline[0]
                    rst_levels.setdefault(character, len(rst_levels) + 1)
                    level, title = rst_levels[character], line.strip()
                    skip_line = True
        if title:
            if stack and len("\n".join(body).strip()) >= 60:
                result.append((" / ".join(value for _, value in stack), "\n".join(body)))
            while stack and stack[-1][0] >= level:
                stack.pop()
            title = re.sub(r"\{[^}]*\}|<[^>]*>", "", title).strip()
            stack.append((level, title))
            body = []
        elif skip_line:
            skip_line = False
        elif stack:
            body.append(line)
    if stack and len("\n".join(body).strip()) >= 60:
        result.append((" / ".join(value for _, value in stack), "\n".join(body)))
    return result


def queries(documents: dict[str, str], repo: str) -> list[dict]:
    seen = set()
    result = []
    for source_id, text in sorted(documents.items()):
        for title, _ in headings(text):
            question = f"For {repo}, explain {title}."
            key = normalize(question)
            if key in seen or len(title) < 3:
                continue
            seen.add(key)
            group = digest(repo + ":" + source_id)
            result.append(
                {
                    "id": digest(repo + ":" + key)[:24],
                    "question": question,
                    "source_document_family": source_id,
                    "group": group,
                    "split": split(group),
                }
            )
    return result


def retrieve(documents: dict[str, str], probes: list[dict], kind: str):
    vectorizer = (
        TfidfVectorizer(stop_words="english", max_features=100000)
        if kind == "word"
        else TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60000)
    )
    source_ids = sorted(documents)
    matrix = vectorizer.fit_transform(documents[source_id] for source_id in source_ids)
    query_matrix = vectorizer.transform(q["question"] for q in probes)
    scores = (query_matrix @ matrix.T).toarray()
    result = {}
    for query, values in zip(probes, scores):
        # Stable tie handling is essential to compare versions meaningfully.
        indices = np.argsort(-values, kind="stable")[: min(3, len(source_ids))]
        result[query["id"]] = tuple(source_ids[i] for i in indices if values[i] > 0)
    return result, vectorizer, query_matrix


def reverse_index_plan(before, after, cases, baseline, controls):
    dependencies = {
        case.case_id: set(case.before.must_cite) | set(case.after.must_cite) for case in cases
    }
    observations = baseline.by_id
    selected = set()
    for case in cases:
        observation = observations.get(case.case_id)
        if observation:
            dependencies[case.case_id].update(observation.retrieved_sources)
            dependencies[case.case_id].update(c.source_id for c in observation.citations)
        if not observation or observation.error or not dependencies[case.case_id]:
            selected.add(case.case_id)
    mapped = set().union(*dependencies.values())
    changes = document_changes(before, after)
    if any(c.kind == "added" or c.source_id not in mapped for c in changes):
        return set(dependencies)
    changed_ids = {c.source_id for c in changes}
    selected.update(case_id for case_id, sources in dependencies.items() if sources & changed_ids)
    remaining = sorted(
        set(dependencies) - selected,
        key=lambda key: fingerprint([before.digest, after.digest, key]),
    )
    return selected | set(remaining[:controls])


def coverage(selected, affected):
    return round(len(selected & affected) / len(affected), 6) if affected else None


def cohort_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["project"], row["after_revision"]), []).append(row)
    events = list(grouped.values())
    rng = np.random.default_rng(42)
    saved, covered = [], []
    for _ in range(1000):
        sample = [
            row for index in rng.integers(0, len(events), len(events)) for row in events[index]
        ]
        saved.append(float(np.median([r["call_reduction"] for r in sample])))
        affected = sum(r["evidence_changed"] for r in sample)
        if affected:
            covered.append(
                sum(r["evidence_changed"] - r["missed_evidence_changes"] for r in sample) / affected
            )
    all_affected = sum(r["evidence_changed"] for r in rows)
    return {
        "update_events": len(events),
        "retriever_update_comparisons": len(rows),
        "query_snapshot_replays": sum(r["queries"] for r in rows),
        "selected_query_replays": sum(r["selected"] for r in rows),
        "affected_evidence_replays": all_affected,
        "covered_evidence_replays": sum(
            r["evidence_changed"] - r["missed_evidence_changes"] for r in rows
        ),
        "pooled_evidence_change_coverage": round(
            sum(r["evidence_changed"] - r["missed_evidence_changes"] for r in rows) / all_affected,
            6,
        )
        if all_affected
        else None,
        "median_evidence_change_coverage": float(
            np.median(
                [
                    r["evidence_change_coverage"]
                    for r in rows
                    if r["evidence_change_coverage"] is not None
                ]
            )
        ),
        "median_call_reduction": float(np.median([r["call_reduction"] for r in rows])),
        "comparisons_with_20pct_saving_and_95pct_coverage": sum(
            r["call_reduction"] >= 0.2
            and r["evidence_change_coverage"] is not None
            and r["evidence_change_coverage"] >= 0.95
            for r in rows
        ),
        "full_fallback_comparisons": sum(r["full_fallback"] for r in rows),
        "update_event_bootstrap": {
            "repetitions": 1000,
            "median_call_reduction_95ci": [
                round(float(v), 6) for v in np.quantile(saved, [0.025, 0.975])
            ],
            "pooled_evidence_coverage_95ci": [
                round(float(v), 6) for v in np.quantile(covered, [0.025, 0.975])
            ],
            "unit": "paired update event; both retrievers stay together; observed projects only",
        },
    }


def evaluate(directory: Path, manifest: dict, output):
    results = []
    inventory = []
    unique_probes = set()
    for release in manifest["releases"]:
        repo = release["repo"]
        old = documentation(directory / release["before"]["file"], repo)
        new = documentation(directory / release["after"]["file"], repo)
        probes = queries(old, repo)
        unique_probes.update(q["id"] for q in probes)
        all_before = Snapshot(
            name=repo + ":" + release["before"]["tag"],
            documents=tuple(Document(source_id=k, content=v) for k, v in sorted(old.items())),
        )
        all_after = Snapshot(
            name=repo + ":" + release["after"]["tag"],
            documents=tuple(Document(source_id=k, content=v) for k, v in sorted(new.items())),
        )
        changes = document_changes(all_before, all_after)
        inventory.append(
            {
                "project": repo,
                "before_tag": release["before"]["tag"],
                "after_tag": release["after"]["tag"],
                "before_documents": len(old),
                "after_documents": len(new),
                "generated_section_probes": len(probes),
                "document_family_groups": len({q["group"] for q in probes}),
                "split_probes": dict(Counter(q["split"] for q in probes)),
                "changes": dict(Counter(c.kind for c in changes)),
            }
        )
        for condition in ("complete_release", "common_files_ablation"):
            common = old.keys() & new.keys()
            before_docs = (
                old if condition == "complete_release" else {k: old[k] for k in sorted(common)}
            )
            after_docs = (
                new if condition == "complete_release" else {k: new[k] for k in sorted(common)}
            )
            selected_probes = (
                probes
                if condition == "complete_release"
                else [q for q in probes if q["source_document_family"] in common]
            )
            if not selected_probes:
                continue
            before = Snapshot(
                name=all_before.name,
                documents=tuple(
                    Document(source_id=k, content=v) for k, v in sorted(before_docs.items())
                ),
            )
            after = Snapshot(
                name=all_after.name,
                documents=tuple(
                    Document(source_id=k, content=v) for k, v in sorted(after_docs.items())
                ),
            )
            cases = tuple(
                Case(
                    case_id=q["id"],
                    question=q["question"],
                    before=Expectation(evidence_required=False),
                    after=Expectation(evidence_required=False),
                )
                for q in selected_probes
            )
            for kind in ("word", "character"):
                started = time.perf_counter()
                old_retrieved, vectorizer, query_matrix = retrieve(
                    before_docs, selected_probes, kind
                )
                new_retrieved, _, _ = retrieve(after_docs, selected_probes, kind)
                baseline = Run(
                    snapshot_digest=before.digest,
                    producer="sklearn-" + kind + "-tfidf",
                    observations=tuple(
                        Observation(
                            case_id=case.case_id,
                            question=case.question,
                            retrieved_sources=old_retrieved[case.case_id],
                            citations=tuple(
                                Citation(
                                    source_id=source_id,
                                    document_hash=before.by_id[source_id].content_hash,
                                    quote=before_docs[source_id][:800],
                                )
                                for source_id in old_retrieved[case.case_id][:1]
                            ),
                        )
                        for case in cases
                    ),
                )
                old_hashes = {k: d.content_hash for k, d in before.by_id.items()}
                new_hashes = {k: d.content_hash for k, d in after.by_id.items()}
                affected = {
                    case.case_id
                    for case in cases
                    if tuple((k, old_hashes[k]) for k in old_retrieved[case.case_id])
                    != tuple((k, new_hashes[k]) for k in new_retrieved[case.case_id])
                }
                change_ids = {c.source_id for c in document_changes(before, after)}
                change_text = "\n".join(
                    (before_docs.get(k, "") + "\n" + after_docs.get(k, ""))
                    for k in sorted(change_ids)
                )
                keyword_values = (
                    (query_matrix @ vectorizer.transform([change_text[:1000000]]).T)
                    .toarray()
                    .ravel()
                )
                keyword_order = [
                    selected_probes[i]["id"] for i in np.argsort(-keyword_values, kind="stable")
                ]
                by_id = {q["id"]: q for q in selected_probes}
                for controls in (0, 1, 5, 20):
                    plan = build_plan(
                        before, after, cases, baseline, mode="impact", audit_controls=controls
                    )
                    chosen = set(plan.selected)
                    plain = reverse_index_plan(before, after, cases, baseline, controls)
                    random_order = sorted(
                        by_id, key=lambda key: digest(f"42:{repo}:{kind}:{condition}:{key}")
                    )
                    random_chosen = set(random_order[: len(chosen)])
                    keyword_chosen = set(keyword_order[: len(chosen)])
                    test_ids = {q["id"] for q in selected_probes if q["split"] == "test"}
                    result = {
                        "project": repo,
                        "condition": condition,
                        "retriever": kind,
                        "before_revision": release["before"]["commit"],
                        "after_revision": release["after"]["commit"],
                        "sentinels": controls,
                        "queries": len(cases),
                        "selected": len(chosen),
                        "call_reduction": round(1 - len(chosen) / len(cases), 6),
                        "evidence_changed": len(affected),
                        "evidence_change_coverage": coverage(chosen, affected),
                        "missed_evidence_changes": len(affected - chosen),
                        "same_budget_random_coverage": coverage(random_chosen, affected),
                        "same_budget_keyword_coverage": coverage(keyword_chosen, affected),
                        "full_retest_coverage": 1 if affected else None,
                        "plain_reverse_index_identical_selection": chosen == plain,
                        "full_fallback": any(
                            "new_or_unmapped_source_requires_full_retest" in reasons
                            for reasons in plan.reasons.values()
                        ),
                        "unchecked": len(plan.skipped),
                        "default_clean_subset_verdict": "inconclusive"
                        if plan.skipped
                        else "passed",
                        "test_queries": len(test_ids),
                        "test_evidence_changed": len(affected & test_ids),
                        "test_evidence_change_coverage": coverage(
                            chosen & test_ids, affected & test_ids
                        ),
                    }
                    results.append(result)
                    for query in selected_probes:
                        output.write(
                            __import__("json").dumps(
                                {
                                    "project": repo,
                                    "condition": condition,
                                    "retriever": kind,
                                    "sentinels": controls,
                                    "id": query["id"],
                                    "group": query["group"],
                                    "split": query["split"],
                                    "source_document_family": query["source_document_family"],
                                    "evidence_changed": query["id"] in affected,
                                    "selected": query["id"] in chosen,
                                    "plain_selected": query["id"] in plain,
                                    "random_selected": query["id"] in random_chosen,
                                    "keyword_selected": query["id"] in keyword_chosen,
                                }
                            )
                            + "\n"
                        )
                print(
                    f"release audit: {repo}, {condition}, {kind}, {len(cases)} probes, {time.perf_counter() - started:.2f}s",
                    flush=True,
                )
    primary = [r for r in results if r["condition"] == "complete_release" and r["sentinels"] == 5]
    median_reduction = float(np.median([r["call_reduction"] for r in primary]))
    coverages = [
        r["evidence_change_coverage"] for r in primary if r["evidence_change_coverage"] is not None
    ]
    gate = median_reduction >= 0.2 and bool(coverages) and min(coverages) >= 0.95
    return {
        "query_type": "Generated authored-section retrieval probes, not human-written QA",
        "backend": "Actual sparse retrieval on release snapshots; no LLM or model-backed LightRAG",
        "ground_truth": "Full-retest top-k source/version signature changes; NOT answer correctness",
        "inventory": inventory,
        "comparisons": results,
        "primary_median_call_reduction": round(median_reduction, 6),
        "primary_min_evidence_change_coverage": min(coverages) if coverages else None,
        "all_plain_reverse_index_selections_match": all(
            r["plain_reverse_index_identical_selection"] for r in results
        ),
        "primary_cohort_summary": cohort_summary(primary),
        "scheduler_continuation_gate": gate,
        "total_baseline_probes": sum(i["generated_section_probes"] for i in inventory),
        "unique_generated_probe_ids": len(unique_probes),
        "update_events": len(inventory),
        "minimum_probe_target_met": len(unique_probes) >= 1000,
    }
