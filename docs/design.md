# Design and artifact contract

## Data flow

```
before corpus ─ snapshot ─ baseline capture ─┐
                                          ├─ dependency graph ─ impact plan
after corpus ── snapshot ──────────────────┘                       │
                          candidate backend ← selected questions ─┘
                                  │
                     checks + sentinel failures?
                          │                  │
                      no expansion       capture skipped cases
                          └────────┬─────────┘
                          JSON + HTML + CI exit
```

All JSON models reject unknown fields. Source IDs are stable logical names; relative
paths are the snapshot CLI's convention. Content hashes use SHA-256 over exact UTF-8
text. Snapshot digests cover sorted `(source_id, content_hash)` pairs, independent
of the snapshot display name and filesystem root. Run digests and suite digests
use canonical sorted-key JSON. Plans bind both snapshots, the suite and the baseline.
These hashes catch accidental mismatches; they are not signatures or access control.

## Selection

An edge `source → case` exists for each baseline retrieved source, baseline answer
citation, or before/after required source. Descendants of changed/removed sources
are impacted. Missing baseline observations, backend errors and absent dependency
information always select the affected case.

Added sources, or changes to a source absent from the graph, force the whole suite.
Otherwise impact mode selects impacted cases plus N controls from the remainder,
ordered by SHA-256 of `(before_digest, after_digest, case_id)`. The default N is 1;
it is an operational knob, not a statistical confidence guarantee.

For live `run`, a control with failed or unknown candidate status triggers full
selection. Only skipped cases are then queried. Full-mode reports retain the
control IDs and an explicit `sentinel_escalation` reason. Replay-only `check` does
not perform new HTTP requests. Full mode is the default for both commands.

## Checks

Each case declares independent `before` and `after` expectations. `contains` and
`excludes` use casefolded substring matching, while quote membership is exact.
`must_cite` names required source IDs. `evidence_required` defaults to true.

Each cited item is checked independently. A deleted source, mismatched supplied
hash, or quote absent from the expected text fails. A source outside the snapshot,
or an ID-only citation, is unknown. A valid hash without a quote verifies the
reported document version only; it does not verify any claim. A matching quote
without a hash verifies content compatibility only.

Candidate failures take CI precedence over unknown comparisons. Missing/error
captures are unknown. If either baseline or candidate status is unknown, the
comparison is inconclusive. Otherwise transitions are pass→pass, regression,
improved, or existing_failure. Existing failures still fail the candidate gate.

All cases must be partitioned into selected and skipped; duplicate selections,
stale source-change lists, and mismatched input hashes are rejected. Unchecked
cases cannot produce a default global pass. `--allow-scoped-pass` is an explicit
policy opt-in that retains the `scoped_pass` label in JSON/HTML.

## LightRAG contract

Checked against the official
[query route definitions](https://github.com/HKUDS/LightRAG/blob/aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a/lightrag/api/routers/query_routes.py)
on 2026-10-26. The request is non-streaming `POST /query` with reference chunk
contents enabled. Response references contain `reference_id`, `file_path`, and
optional `content: list[str]`. Only numeric IDs referenced in the answer are
normalized as citations; all returned paths are retained as retrieval dependencies.

Paths are mapped through an exact alias map. Multiple chunks remain multiple
citation records. Unknown numeric reference IDs remain visible as unresolved
sources. HTTP, JSON and shape errors produce sanitized unknown observations.
API keys, headers and raw error bodies are not persisted.

The test suite uses documented HTTP fixtures and failure injection. It does not
start a model-backed LightRAG service. Custom prompts, normalized ingestion text,
page extraction or different citation syntax can require a custom adapter. The
expected snapshot is supplied by the caller, not attested by the server.

Verified upstream commit: `aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a`.
The inspected route file SHA-256 is
`61ec3b5b04f10e5dd1511bbc6cd6a1ba9c152552f09da21686814f1cce2c6931`.

## Deliberate v0.1 boundaries

No ingestion service, automatic question generator, LLM judge, semantic claim
verification, cache mutation, or GPU workload is included. The reusable unit is
a knowledge-update test workflow with portable artifacts. Stable source identity
and meaningful human-written contracts are required.
