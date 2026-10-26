"""Read-only LightRAG capture and a reproducible extractive reference backend."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from .models import Case, Citation, Observation, Run, Snapshot


class Backend(Protocol):
    name: str

    def observe(self, case: Case, snapshot: Snapshot) -> Observation: ...


def capture(backend: Backend, snapshot: Snapshot, cases: Sequence[Case]) -> Run:
    return Run(
        snapshot_digest=snapshot.digest,
        producer=backend.name,
        observations=tuple(backend.observe(case, snapshot) for case in cases),
    )


class LightRAGBackend:
    """Consume documented POST /query JSON, requesting reference chunk contents.

    The supplied snapshot is the expected corpus, not a server-side attestation.
    Retrieved references and answer citations are recorded separately. No document
    or index mutation is performed. API keys never enter an Observation or Run.
    """

    name = "lightrag-http"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        mode: str = "mix",
        source_aliases: dict[str, str] | None = None,
        timeout: float = 120,
        client: httpx.Client | None = None,
    ):
        parsed = httpx.URL(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("LightRAG URL must use http or https")
        if parsed.userinfo or parsed.query or parsed.fragment:
            raise ValueError(
                "use an API-key environment variable, not URL credentials or query strings"
            )
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.aliases = dict(source_aliases or {})
        self.timeout = timeout
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self._client = client

    def observe(self, case: Case, snapshot: Snapshot) -> Observation:
        body = {
            "query": case.question,
            "mode": self.mode,
            "stream": False,
            "include_references": True,
            "include_chunk_content": True,
        }
        try:
            if self._client is not None:
                response = self._client.post(
                    self.base_url + "/query", json=body, headers=self.headers, timeout=self.timeout
                )
            else:
                with httpx.Client(follow_redirects=False) as client:
                    response = client.post(
                        self.base_url + "/query",
                        json=body,
                        headers=self.headers,
                        timeout=self.timeout,
                    )
            response.raise_for_status()
            return self.normalize(case, response.json())
        except httpx.HTTPStatusError as exc:
            error = f"LightRAG returned HTTP {exc.response.status_code}"
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            # Avoid persisting headers, credentials, endpoint query strings, or raw bodies.
            error = f"LightRAG capture failed: {type(exc).__name__}"
        return Observation(case_id=case.case_id, question=case.question, error=error)

    def normalize(self, case: Case, payload: dict) -> Observation:
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise ValueError("expected LightRAG /query JSON with a string response")
        references = payload.get("references") or []
        if not isinstance(references, list):
            raise ValueError("references must be an array")
        by_reference: dict[str, dict] = {}
        retrieved = []
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError("each reference must be an object")
            reference_id, path = reference.get("reference_id"), reference.get("file_path")
            if not isinstance(reference_id, str) or not isinstance(path, str) or not path:
                raise ValueError("reference_id and file_path must be nonempty strings")
            if reference_id in by_reference:
                raise ValueError("duplicate reference_id")
            by_reference[reference_id] = reference
            retrieved.append(self.aliases.get(path, path))
        # The default LightRAG prompt uses [n] citations. A returned reference
        # inventory alone is NOT evidence that the answer cited those sources.
        cited_ids = []
        for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", payload["response"]):
            cited_ids.extend(piece.strip() for piece in group.split(","))
        citations = []
        for reference_id in dict.fromkeys(cited_ids):
            reference = by_reference.get(reference_id)
            if reference is None:
                citations.append(Citation(source_id=f"unresolved-reference:{reference_id}"))
                continue
            path = reference["file_path"]
            source_id = self.aliases.get(path, path)
            chunks = reference.get("content")
            if chunks is not None and (
                not isinstance(chunks, list) or any(not isinstance(c, str) for c in chunks)
            ):
                raise ValueError("reference content must be an array of strings")
            quotes = [chunk for chunk in (chunks or []) if chunk]
            if quotes:
                citations.extend(Citation(source_id=source_id, quote=quote) for quote in quotes)
            else:
                citations.append(Citation(source_id=source_id))
        return Observation(
            case_id=case.case_id,
            question=case.question,
            answer=payload["response"],
            citations=tuple(citations),
            retrieved_sources=tuple(dict.fromkeys(retrieved)),
            metadata={
                "citation_style": "numeric_brackets",
                "llm_generated": payload.get("llm_generated"),
            },
        )


class ExtractiveBackend:
    """TF-IDF retrieval and quoted source text for the offline control demo.

    This is a reference backend, not an LLM or a measure of generation quality.
    It reuses scikit-learn and needs no tokenizer/model downloads.
    """

    name = "sklearn-extractive-reference"

    def __init__(self, snapshot: Snapshot):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.snapshot = snapshot
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(d.content for d in snapshot.documents)

    def observe(self, case: Case, snapshot: Snapshot) -> Observation:
        if snapshot.digest != self.snapshot.digest:
            raise ValueError("reference backend was initialized for a different snapshot")
        scores = (self.matrix @ self.vectorizer.transform([case.question]).T).toarray().ravel()
        index = int(scores.argmax())
        if scores[index] <= 0:
            return Observation(
                case_id=case.case_id, question=case.question, answer="No evidence available."
            )
        document = snapshot.documents[index]
        return Observation(
            case_id=case.case_id,
            question=case.question,
            answer=document.content + " [1]",
            citations=(
                Citation(
                    source_id=document.source_id,
                    document_hash=document.content_hash,
                    quote=document.content,
                ),
            ),
            retrieved_sources=(document.source_id,),
        )
