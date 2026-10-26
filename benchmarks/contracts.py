"""Controlled faults measure contract behavior, never real-model hallucination accuracy."""

from __future__ import annotations

import json
from dataclasses import dataclass

from rag_changelab.evaluate import inspect_observation
from rag_changelab.models import Citation, Document, Expectation, Observation, Snapshot


@dataclass
class Fixture:
    identifier: str
    group: str
    split: str
    domain: str
    question: str
    answer: str
    equivalent: str
    snapshot: Snapshot
    citations: tuple[Citation, ...]
    category: str
    future_dated_gold: bool = False
    unknown_gold_time: bool = False


def timely_fixtures(rows):
    for row in rows:
        documents = tuple(
            Document(source_id=f"{row['domain']}/{d['global_doc_id']}", content=d["text"])
            for d in row["docs_ko"]
        )
        snapshot = Snapshot(name=row["row_id"], documents=documents)
        citations = tuple(
            Citation(
                source_id=documents[i].source_id,
                quote=documents[i].content,
                document_hash=documents[i].content_hash,
            )
            for i in row["gold_docs_final"]
        )
        # Whitespace changes preserve words and order, not a model-generated paraphrase.
        equivalent = row["answer"].replace(" ", "\n")
        yield Fixture(
            row["row_id"],
            row["group"],
            row["split"],
            row["domain"],
            row["query_ko"],
            row["answer"],
            equivalent,
            snapshot,
            citations,
            row["category"],
            row["future_gold"],
            row["unknown_gold_time"],
        )


def boolean_fixtures(sources, responses):
    source_split = {r["source_id"]: r["split"] for r in responses}
    fields = {
        "RestaurantsReservations": "reservations",
        "OutdoorSeating": "outdoor seating",
        "RestaurantsTakeOut": "takeout",
        "RestaurantsGoodForGroups": "group dining",
    }
    for source in sources:
        if source["task_type"] != "Data2txt":
            continue
        information = source["source_info"]
        document = Document(
            source_id="ragtruth/" + source["source_id"],
            content=json.dumps(information, ensure_ascii=False, sort_keys=True),
        )
        snapshot = Snapshot(name=source["source_id"], documents=(document,))
        for field, label in fields.items():
            value = (information.get("attributes") or {}).get(field)
            if not isinstance(value, bool):
                continue
            source_id = "ragtruth:" + source["source_id"]
            yield Fixture(
                source_id + ":" + field,
                source_id,
                "test" if source_split[source["source_id"]] == "test" else "development",
                "business_boolean",
                f"Does {information['name']} offer {label}?",
                "Yes" if value else "No",
                "It does." if value else "It does not.",
                snapshot,
                (
                    Citation(
                        source_id=document.source_id,
                        quote=document.content,
                        document_hash=document.content_hash,
                    ),
                ),
                field,
            )


def simple_check(observation, snapshot, expected, removed):
    """Independent plain-Python baseline implementing the same declared contracts."""
    if observation.error:
        return "unknown"
    text = observation.answer.casefold()
    sources = {c.source_id for c in observation.citations}
    failed = (
        any(t.casefold() not in text for t in expected.contains)
        or any(t.casefold() in text for t in expected.excludes)
        or not set(expected.must_cite) <= sources
        or (expected.evidence_required and not sources)
    )
    unknown = False
    documents = snapshot.by_id
    for citation in observation.citations:
        document = documents.get(citation.source_id)
        if document is None:
            failed |= citation.source_id in removed
            unknown |= citation.source_id not in removed
        else:
            failed |= bool(
                citation.document_hash and citation.document_hash != document.content_hash
            )
            failed |= citation.quote is not None and citation.quote not in document.content
            unknown |= citation.document_hash is None and citation.quote is None
    return "failed" if failed else "unknown" if unknown else "passed"


def evaluate(fixtures, output):
    rows = []
    for fixture in fixtures:
        expected = Expectation(
            contains=(fixture.answer,), must_cite=tuple(c.source_id for c in fixture.citations)
        )
        observation = Observation(
            case_id=fixture.identifier,
            question=fixture.question,
            answer=fixture.answer,
            citations=fixture.citations,
        )
        first = fixture.citations[0]
        deleted_snapshot = Snapshot(
            name=fixture.snapshot.name + ":deleted",
            documents=tuple(
                d for d in fixture.snapshot.documents if d.source_id != first.source_id
            ),
        )
        variants = [
            ("valid_exact", "passed", observation, fixture.snapshot, expected, frozenset()),
            (
                "valid_wrapper",
                "passed",
                observation.model_copy(
                    update={"answer": "According to the source: " + fixture.answer}
                ),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "valid_equivalent_rendering",
                "passed",
                observation.model_copy(update={"answer": fixture.equivalent}),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "negated_required_phrase",
                "failed",
                observation.model_copy(update={"answer": "The answer is NOT: " + fixture.answer}),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "contradictory_extra_claim",
                "failed",
                observation.model_copy(
                    update={"answer": fixture.answer + "\nThe preceding answer is false."}
                ),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "missing_required_answer",
                "failed",
                observation.model_copy(update={"answer": "Unspecified."}),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "missing_citation",
                "failed",
                observation.model_copy(update={"citations": ()}),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "deleted_source",
                "failed",
                observation,
                deleted_snapshot,
                expected,
                frozenset({first.source_id}),
            ),
            (
                "wrong_source_hash",
                "failed",
                observation.model_copy(
                    update={
                        "citations": (first.model_copy(update={"document_hash": "0" * 64}),)
                        + fixture.citations[1:]
                    }
                ),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "unsupported_quote",
                "failed",
                observation.model_copy(
                    update={
                        "citations": (
                            first.model_copy(
                                update={"quote": "UNSUPPORTED-QUOTE-" + fixture.identifier}
                            ),
                        )
                        + fixture.citations[1:]
                    }
                ),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "id_only_evidence",
                "unknown",
                observation.model_copy(
                    update={
                        "citations": tuple(
                            Citation(source_id=c.source_id) for c in fixture.citations
                        )
                    }
                ),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
            (
                "unmapped_source",
                "unknown",
                observation.model_copy(
                    update={"citations": (Citation(source_id="unmapped:" + fixture.identifier),)}
                ),
                fixture.snapshot,
                expected.model_copy(update={"must_cite": ()}),
                frozenset(),
            ),
            (
                "backend_unavailable",
                "unknown",
                observation.model_copy(update={"error": "injected timeout"}),
                fixture.snapshot,
                expected,
                frozenset(),
            ),
        ]
        for family, label, candidate, snapshot, contract, removed in variants:
            actual, findings = inspect_observation(candidate, snapshot, contract, removed=removed)
            baseline = simple_check(candidate, snapshot, contract, removed)
            row = {
                "id": fixture.identifier + ":" + family,
                "source_case": fixture.identifier,
                "group": fixture.group,
                "split": fixture.split,
                "domain": fixture.domain,
                "category": fixture.category,
                "future_dated_gold": fixture.future_dated_gold,
                "unknown_gold_time": fixture.unknown_gold_time,
                "family": family,
                "expected": label,
                "actual": actual,
                "plain_python": baseline,
                "codes": [f.code for f in findings],
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    return rows
