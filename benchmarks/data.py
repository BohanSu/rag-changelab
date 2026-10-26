"""Pinned corpus readers, explicit provenance and leakage-resistant grouping."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def template_group(text: str) -> str:
    return digest(re.sub(r"\d+(?:[.,:/-]\d+)*", "#", normalize(text)))


def split(group: str) -> str:
    bucket = int(group[:12], 16) % 100
    return "development" if bucket < 70 else "validation" if bucket < 85 else "test"


def timestamp(value: str):
    korean = re.fullmatch(
        r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s+(\d{1,2}):(\d{2}):(\d{2})", value.strip()
    )
    try:
        if korean:
            return datetime(*(int(part) for part in korean.groups()))
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Relative phrases, corrupt digits and invalid dates stay unknown.
        return None


def zip_jsonl(path: Path, suffix: str) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        names = [
            n for n in archive.namelist() if n.endswith(suffix) and not n.startswith("__MACOSX")
        ]
        if len(names) != 1:
            raise ValueError(f"ambiguous archive member: {suffix}")
        return [json.loads(line) for line in archive.open(names[0])]


def ragtruth(directory: Path) -> tuple[list[dict], list[dict], dict]:
    sources = zip_jsonl(directory / "ragtruth.zip", "/dataset/source_info.jsonl")
    responses = zip_jsonl(directory / "ragtruth.zip", "/dataset/response.jsonl")
    by_id = {s["source_id"]: s for s in sources}
    groups = {
        label: {r["source_id"] for r in responses if r["split"] == label}
        for label in ("train", "test")
    }
    if groups["train"] & groups["test"]:
        raise ValueError("RAGTruth official source split overlaps")
    for response in responses:
        response["task_type"] = by_id[response["source_id"]]["task_type"]
    inventory = {
        "source_instances": len(sources),
        "model_responses": len(responses),
        "models": dict(Counter(r["model"] for r in responses)),
        "source_tasks": dict(Counter(s["task_type"] for s in sources)),
        "split_responses": dict(Counter(r["split"] for r in responses)),
        "split_source_groups": {label: len(ids) for label, ids in groups.items()},
        "split_overlap": 0,
        "original_answer_citation_fields": 0,
        "native_semantic_scoring": "unsupported: no answer contracts or reliable answer-citation metadata; not scored",
        "independence_unit": "source instance, not six model responses",
    }
    return sources, responses, inventory


def source_context(source: dict) -> str:
    value = source["source_info"]
    if source["task_type"] == "QA":
        return value["passages"]
    return (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )


def timelyqa(directory: Path) -> tuple[list[dict], dict]:
    all_rows = []
    audits = {}
    with zipfile.ZipFile(directory / "timelyqa.zip") as archive:
        for filename in sorted(archive.namelist()):
            if not filename.endswith(".jsonl") or filename.startswith("__MACOSX"):
                continue
            domain = filename.removeprefix("timelyrag_").removesuffix("_dataset_v2.jsonl")
            rows = [json.loads(line) for line in archive.open(filename)]
            exact = Counter(normalize(r["query_ko"]) for r in rows)
            future = 0
            malformed = 0
            empty = 0
            unknown_time_rows = 0
            unknown_document_times = 0
            seen = set()
            groups = set()
            texts = set()
            doc_ids = set()
            conflicts = defaultdict(set)
            multihop = 0
            for row in rows:
                question = normalize(row["query_ko"])
                conflicts[question].add(normalize(row["answer"]))
                documents = row["docs_ko"]
                gold = row["gold_docs_final"]
                valid = bool(gold) and all(
                    isinstance(i, int) and 0 <= i < len(documents) for i in gold
                )
                if not row["query_ko"].strip() or not row["answer"].strip():
                    empty += 1
                    continue
                valid = valid and all(documents[i]["text"].strip() for i in gold)
                if not valid or len({d["global_doc_id"] for d in documents}) != len(documents):
                    malformed += 1
                    continue
                query_time = timestamp(row["query_insert_time"])
                document_times = [timestamp(d["doc_insert_time"]) for d in documents]
                unknown_time = query_time is None or any(document_times[i] is None for i in gold)
                unknown_time_rows += unknown_time
                unknown_document_times += sum(value is None for value in document_times)
                is_future = bool(
                    query_time
                    and any(
                        document_times[i] is not None and document_times[i] > query_time
                        for i in gold
                    )
                )
                future += is_future
                multihop += len(gold) > 1
                texts.update(digest(d["text"]) for d in documents)
                doc_ids.update(d["global_doc_id"] for d in documents)
                group = template_group(row["query_ko"])
                groups.add(group)
                record = digest(
                    json.dumps(
                        [
                            question,
                            row["query_insert_time"],
                            normalize(row["answer"]),
                            [(normalize(d["text"]), d["doc_insert_time"]) for d in documents],
                            gold,
                        ],
                        ensure_ascii=False,
                    )
                )
                if record in seen:
                    continue
                seen.add(record)
                row = {
                    **row,
                    "domain": domain,
                    "row_id": f"{domain}:{row['index']}",
                    "group": group,
                    "split": split(group),
                    "future_gold": is_future,
                    "unknown_gold_time": unknown_time,
                }
                all_rows.append(row)
            audits[domain] = {
                "raw_questions": len(rows),
                "unique_normalized_queries": len(exact),
                "numeric_masked_query_groups": len(groups),
                "duplicate_query_rows": sum(count - 1 for count in exact.values()),
                "duplicate_query_groups_with_different_answers": sum(
                    len(v) > 1 for v in conflicts.values()
                ),
                "unique_document_texts": len(texts),
                "document_ids": len(doc_ids),
                "categories": dict(Counter(r["category"] for r in rows)),
                "multi_document_gold_questions": multihop,
                "future_dated_gold_questions": future,
                "unknown_gold_availability_time_questions": unknown_time_rows,
                "unknown_document_times": unknown_document_times,
                "malformed_rows": malformed,
                "empty_reference_or_question_rows": empty,
                "deduplicated_context_cases": len(seen),
                "exact_duplicate_context_rows": len(rows) - malformed - empty - len(seen),
                "as_of_policy": "future-dated or unparseable gold times quarantined from availability claims; not automatically annotation errors",
            }
    split_groups = {
        name: {r["group"] for r in all_rows if r["split"] == name}
        for name in ("development", "validation", "test")
    }
    assert not (
        split_groups["development"] & split_groups["validation"]
        or split_groups["development"] & split_groups["test"]
        or split_groups["validation"] & split_groups["test"]
    )
    return all_rows, {
        "construction": "Externally generated synthetic Korean temporal QA, not actual version history",
        "domains": audits,
        "deduplicated_context_cases": len(all_rows),
        "unique_normalized_questions": sum(a["unique_normalized_queries"] for a in audits.values()),
        "numeric_masked_query_groups": len({r["group"] for r in all_rows}),
        "split_questions": dict(Counter(r["split"] for r in all_rows)),
        "split_groups": {name: len(ids) for name, ids in split_groups.items()},
        "split_group_overlap": 0,
        "dedup_limit": "Exact normalized context records and numeric-masked query groups only; no claim of complete semantic deduplication",
        "license": "No explicit license file in pinned repository; corpus text not redistributed by this project",
    }
