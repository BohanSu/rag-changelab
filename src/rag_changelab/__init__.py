"""Change-aware, evidence-versioned RAG regression tests."""

__version__ = "0.1.0"

from .evaluate import compare_runs
from .models import Case, Citation, Document, Expectation, Observation, Run, Snapshot
from .planner import build_plan

__all__ = [
    "Case",
    "Citation",
    "Document",
    "Expectation",
    "Observation",
    "Run",
    "Snapshot",
    "__version__",
    "build_plan",
    "compare_runs",
]
