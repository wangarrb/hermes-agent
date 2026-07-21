# Egomotion4D Mental-Model Source

This directory tracks the reproducible source for the project-specific
Hindsight mental-model workflow. Runtime state remains outside Git under
`~/.hermes/mental-models/egomotion4d/`.

Tracked source:

- `specs/` and `sources/`: generation contracts and curated evidence;
- `benchmark/`: frozen questions and A/B runner;
- `pitfall_writer.py`: sole writer for the canonical Pitfall index/catalog;
- `recreate_models.py`: explicit model-slot bootstrap utility;
- `tests/`: governance, completeness, adjudication and lifecycle tests.

Do not commit runtime registry state, accepted model contents, generated
reports, manifests, benchmark results, conflicts, backups, caches, or API
credentials. The operational scripts under `local/hermes-scripts/` continue to
read and write the runtime directory.
