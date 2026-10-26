"""Exercise real loopback HTTP + CLI artifacts against the public LightRAG wire shape.

This fixture is not a model-backed LightRAG server.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rag_changelab.artifacts import write_model
from rag_changelab.cli import main
from rag_changelab.models import Cases
from rag_changelab.planner import build_plan


def test_real_http_capture_and_sentinel_escalation(knowledge_update, tmp_path, monkeypatch):
    before, after, cases, baseline, healthy = knowledge_update
    initial = build_plan(before, after, cases, baseline, mode="impact", audit_controls=1)
    control = initial.controls[0]
    failures = {control, initial.skipped[0]}
    calls = []
    by_question = {c.question: c for c in cases}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/query"
            assert self.headers["X-API-Key"] == "local-test-only-secret"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["include_chunk_content"] is True
            assert payload["include_references"] is True
            assert payload["stream"] is False
            case = by_question[payload["query"]]
            calls.append(case.case_id)
            observation = healthy.by_id[case.case_id]
            answer = observation.answer
            if case.case_id in failures:
                answer = answer.replace(" [1]", "")
            references = [
                {
                    "reference_id": str(i),
                    "file_path": "/server/" + citation.source_id,
                    "content": [citation.quote],
                }
                for i, citation in enumerate(observation.citations, 1)
            ]
            body = json.dumps(
                {"response": answer, "references": references, "llm_generated": True}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for name, value in (
            ("before.json", before),
            ("after.json", after),
            ("baseline.json", baseline),
            ("cases.json", Cases(cases=cases)),
        ):
            write_model(tmp_path / name, value)
        write_model(
            tmp_path / "suite.json",
            {
                "before_snapshot": "before.json",
                "after_snapshot": "after.json",
                "cases": "cases.json",
                "before_run": "baseline.json",
            },
        )
        write_model(
            tmp_path / "aliases.json",
            {"/server/" + d.source_id: d.source_id for d in after.documents},
        )
        monkeypatch.setenv("CHANGELAB_TEST_KEY", "local-test-only-secret")
        code = main(
            [
                "run",
                str(tmp_path / "suite.json"),
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--aliases",
                str(tmp_path / "aliases.json"),
                "--api-key-env",
                "CHANGELAB_TEST_KEY",
                "--mode",
                "impact",
                "--out",
                str(tmp_path / "report"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert code == 1
    assert len(calls) == len(set(calls)) == len(cases)
    result = json.loads((tmp_path / "report/report.json").read_text())
    assert result["summary"]["candidate_failures"] == 2
    assert result["plan"]["mode"] == "full"
    assert result["summary"]["skipped"] == 0
    for path in (tmp_path / "report").iterdir():
        assert "local-test-only-secret" not in path.read_text()
