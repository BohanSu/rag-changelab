"""Immutable input artifacts and content-addressed snapshots."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def unique_ids(values: tuple, attribute: str, label: str) -> None:
    duplicates = [
        key for key, count in Counter(getattr(x, attribute) for x in values).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(sorted(duplicates))}")


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Document(Model):
    source_id: Text
    content: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class Snapshot(Model):
    schema_version: Literal[1] = 1
    name: Text
    documents: tuple[Document, ...]
    digest: str = ""

    @model_validator(mode="after")
    def verify_digest(self) -> Snapshot:
        unique_ids(self.documents, "source_id", "source IDs")
        actual = fingerprint(sorted((d.source_id, d.content_hash) for d in self.documents))
        if self.digest and self.digest != actual:
            raise ValueError("snapshot digest does not match its document contents")
        object.__setattr__(self, "digest", actual)
        return self

    @property
    def by_id(self) -> dict[str, Document]:
        return {document.source_id: document for document in self.documents}


class Citation(Model):
    source_id: Text
    document_hash: Digest | None = None
    quote: Text | None = None


class Observation(Model):
    case_id: Text
    question: Text
    answer: str = ""
    citations: tuple[Citation, ...] = ()
    retrieved_sources: tuple[str, ...] = ()
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Run(Model):
    schema_version: Literal[1] = 1
    snapshot_digest: Digest
    producer: Text
    observations: tuple[Observation, ...]

    @model_validator(mode="after")
    def validate_cases(self) -> Run:
        unique_ids(self.observations, "case_id", "observation case IDs")
        return self

    @property
    def by_id(self) -> dict[str, Observation]:
        return {observation.case_id: observation for observation in self.observations}


class Expectation(Model):
    contains: tuple[Text, ...] = ()
    excludes: tuple[Text, ...] = ()
    must_cite: tuple[Text, ...] = ()
    evidence_required: bool = True


class Case(Model):
    case_id: Text
    question: Text
    before: Expectation
    after: Expectation
    tags: tuple[str, ...] = ()


class Cases(Model):
    schema_version: Literal[1] = 1
    cases: tuple[Case, ...]

    @model_validator(mode="after")
    def validate_cases(self) -> Cases:
        if not self.cases:
            raise ValueError("the suite must contain at least one case")
        unique_ids(self.cases, "case_id", "test case IDs")
        return self


class Change(Model):
    source_id: str
    kind: Literal["added", "modified", "removed"]
    before_hash: str | None = None
    after_hash: str | None = None


class Plan(Model):
    before_digest: Digest
    after_digest: Digest
    cases_digest: Digest
    baseline_run_digest: Digest
    mode: Literal["full", "impact"]
    changes: tuple[Change, ...]
    selected: tuple[str, ...]
    impacted: tuple[str, ...]
    controls: tuple[str, ...]
    skipped: tuple[str, ...]
    reasons: dict[str, tuple[str, ...]]


class Finding(Model):
    code: str
    level: Literal["failure", "unknown"]
    message: str
    source_id: str | None = None


class CaseResult(Model):
    case_id: str
    question: str
    before_status: Literal["passed", "failed", "unknown", "not_checked"]
    after_status: Literal["passed", "failed", "unknown", "not_checked"]
    transition: str
    before_findings: tuple[Finding, ...] = ()
    after_findings: tuple[Finding, ...] = ()
    before_observation: Observation | None = None
    after_observation: Observation | None = None


class Report(Model):
    schema_version: Literal[1] = 1
    tool_version: str = "0.1.0"
    before_name: str
    after_name: str
    plan: Plan
    results: tuple[CaseResult, ...]
    decision: Literal["passed", "failed", "inconclusive", "scoped_pass"]
    summary: dict[str, int]

    @property
    def exit_code(self) -> int:
        return {"passed": 0, "scoped_pass": 0, "failed": 1, "inconclusive": 3}[self.decision]
