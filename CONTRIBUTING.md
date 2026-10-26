# Contributing

Use Python 3.11+ and install `pip install -e '.[demo,dev]'`. Run `pytest -q` and
`ruff check .` before submitting changes. New adapters must preserve the distinction
between retrieved sources and answer citations, and must not fabricate source hashes.

Useful contributions include captured public API fixtures, additional backend
normalizers, multilingual citation formats, and reproducible real-corpus studies.
Include the deployment/version and expected normalized artifact with an adapter
change. Do not include API keys, private corpora or provider debug responses.

Evaluation changes should test an observable failure case and a legitimate knowledge
update. Planner changes must preserve explicit unchecked coverage, input binding,
and the conservative fallback for newly added or unmapped sources.

An evaluation result is limited to the checks actually executed. Avoid claiming
semantic faithfulness from quote membership, global safety from sentinel sampling,
or production quality from the synthetic demo.
