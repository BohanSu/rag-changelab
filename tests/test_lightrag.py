import json

import httpx
import pytest

from rag_changelab.adapters import LightRAGBackend, capture
from rag_changelab.evaluate import inspect_observation


def test_request_contract_and_only_answer_cited_references(knowledge_update):
    _, after, cases, _, _ = knowledge_update
    case = next(c for c in cases if c.case_id == "retention")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "response": "Audit log retention is 7 days. [1]",
                "references": [
                    {
                        "reference_id": "1",
                        "file_path": "/server/retention.md",
                        "content": ["Audit log retention is 7 days."],
                    },
                    {
                        "reference_id": "2",
                        "file_path": "/server/upload.md",
                        "content": ["Upload size limit is 10 MB."],
                    },
                ],
                "llm_generated": True,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        backend = LightRAGBackend(
            "https://example.invalid/rag/",
            api_key="secret-test-key",
            source_aliases={
                "/server/retention.md": "retention.md",
                "/server/upload.md": "upload.md",
            },
            client=client,
        )
        run = capture(backend, after, [case])
    request = requests[0]
    assert str(request.url) == "https://example.invalid/rag/query"
    assert request.headers["x-api-key"] == "secret-test-key"
    assert json.loads(request.content) == {
        "query": case.question,
        "mode": "mix",
        "stream": False,
        "include_references": True,
        "include_chunk_content": True,
    }
    observation = run.observations[0]
    assert observation.retrieved_sources == ("retention.md", "upload.md")
    assert [c.source_id for c in observation.citations] == ["retention.md"]
    assert observation.citations[0].document_hash is None  # Do not invent backend attestation.
    assert "secret-test-key" not in run.model_dump_json()
    assert inspect_observation(observation, after, case.after)[0] == "passed"


def test_reference_inventory_without_inline_citations_is_not_evidence(knowledge_update):
    _, after, cases, _, _ = knowledge_update
    case = cases[0]
    observation = LightRAGBackend("http://localhost:9621").normalize(
        case,
        {
            "response": "7 days",
            "references": [
                {"reference_id": "1", "file_path": "retention.md", "content": ["7 days"]}
            ],
        },
    )
    assert observation.retrieved_sources == ("retention.md",)
    assert not observation.citations
    assert inspect_observation(observation, after, case.after)[0] == "failed"


def test_grouped_and_repeated_numeric_citations_keep_every_chunk(knowledge_update):
    _, _, cases, _, _ = knowledge_update
    observation = LightRAGBackend("http://localhost").normalize(
        cases[0],
        {
            "response": "Evidence [1, 2] and again [1].",
            "references": [
                {"reference_id": "1", "file_path": "a.md", "content": ["a1", "a2"]},
                {"reference_id": "2", "file_path": "b.md"},
            ],
        },
    )
    assert [(c.source_id, c.quote) for c in observation.citations] == [
        ("a.md", "a1"),
        ("a.md", "a2"),
        ("b.md", None),
    ]


def test_unresolved_reference_stays_visible(knowledge_update):
    _, _, cases, _, _ = knowledge_update
    observation = LightRAGBackend("http://localhost").normalize(
        cases[0], {"response": "A missing citation [99]", "references": []}
    )
    assert observation.citations[0].source_id == "unresolved-reference:99"


@pytest.mark.parametrize("failure", ["http", "timeout", "bad-json", "bad-shape"])
def test_capture_failures_are_unknown_and_do_not_persist_secrets(knowledge_update, failure):
    _, after, cases, _, _ = knowledge_update

    def handler(request):
        if failure == "http":
            return httpx.Response(503, text="private debug body secret-test-key")
        if failure == "timeout":
            raise httpx.ReadTimeout("private detail secret-test-key", request=request)
        if failure == "bad-json":
            return httpx.Response(200, text="secret-test-key")
        return httpx.Response(200, json={"response": 7, "secret": "secret-test-key"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = capture(
            LightRAGBackend("https://example.invalid", api_key="secret-test-key", client=client),
            after,
            cases[:1],
        )
    assert run.observations[0].error
    assert "secret-test-key" not in run.model_dump_json()
    assert inspect_observation(run.observations[0], after, cases[0].after)[0] == "unknown"


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/rag",
        "https://name:password@example.org",
        "https://example.org?key=private",
        "https://example.org/#private",
    ],
)
def test_credentials_and_non_http_urls_are_rejected(url):
    with pytest.raises(ValueError):
        LightRAGBackend(url)


@pytest.mark.parametrize(
    "references",
    [
        "not-an-array",
        [None],
        [{"reference_id": "1", "file_path": ""}],
        [{"reference_id": "1", "file_path": "a"}, {"reference_id": "1", "file_path": "b"}],
        [{"reference_id": "1", "file_path": "a", "content": "not-an-array"}],
    ],
)
def test_malformed_cited_references_are_rejected(knowledge_update, references):
    _, _, cases, _, _ = knowledge_update
    with pytest.raises(ValueError):
        LightRAGBackend("http://localhost").normalize(
            cases[0], {"response": "A cited answer [1]", "references": references}
        )
