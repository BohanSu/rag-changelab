import pytest

from rag_changelab.adapters import ExtractiveBackend, capture
from rag_changelab.demo import fixtures


@pytest.fixture
def knowledge_update():
    before, after, cases = fixtures()
    baseline = capture(ExtractiveBackend(before), before, cases)
    candidate = capture(ExtractiveBackend(after), after, cases)
    return before, after, cases, baseline, candidate
