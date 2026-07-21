# Egomotion4D Mental-Model Source

This directory tracks the reproducible source for the project-specific
Hindsight mental-model workflow. Runtime state remains outside Git under
`~/.hermes/mental-models/egomotion4d/`.

Tracked source:

- `specs/`: generation and completeness contracts;
- `sources/*_current_evidence.md`: reproducible, replaceable build inputs derived
  from current KG/evidence. They must be registered in
  `sources/derived-build-inputs.json` and must never become an independently
  maintained truth store;
- `benchmark/`: frozen questions and A/B runner;
- `pitfall_writer.py`: sole writer for the canonical Pitfall index/catalog;
- `recreate_models.py`: explicit model-slot bootstrap utility;
- `tests/`: governance, completeness, adjudication and lifecycle tests.

Do not commit runtime registry state, accepted model contents, generated
reports, manifests, benchmark results, conflicts, backups, caches, or API
credentials. The operational scripts under `local/hermes-scripts/` continue to
read and write the runtime directory.

Knowledge changes flow in one direction: update KG/current evidence first,
rebuild the source snapshot and SHA-bound candidate, then publish only after
adjudication returns `PASS_PUBLISH`. A failed or conflicting candidate cannot
rewrite KG and cannot replace the previous accepted revision.
