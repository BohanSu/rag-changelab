"""Adapt your RAG; replace the illustrative query function with your own client.

Returning the expected source's hash without receiving it from your backend would
fabricate evidence. Pass only hashes/quotes actually produced by the integration.
"""

from rag_changelab.models import Case, Citation, Observation, Snapshot


class MyRAG:
    name = "my-rag"

    def __init__(self, query):
        self.query = query

    def observe(self, case: Case, snapshot: Snapshot) -> Observation:
        response = self.query(case.question)
        return Observation(
            case_id=case.case_id,
            question=case.question,
            answer=response["answer"],
            citations=tuple(
                Citation(
                    source_id=item["source_id"],
                    quote=item.get("quote"),
                    document_hash=item.get("document_hash"),
                )
                for item in response["answer_citations"]
            ),
            retrieved_sources=tuple(response["retrieved_sources"]),
        )
