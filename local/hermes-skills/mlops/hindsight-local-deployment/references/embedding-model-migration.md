# Hindsight embedding model migration

Use this reference when changing Hindsight's local embedding model for an existing production bank.

## Key rule

Do not mix embeddings from different models inside the same vector index/bank.

Observed example:
- `BAAI/bge-small-en-v1.5` uses `hidden_size=384` and existing `memory_units.embedding` may be `vector(384)`.
- `BAAI/bge-m3` uses `hidden_size=1024`.

A model switch from bge-small to bge-m3 is therefore not a simple config-only change for an existing populated bank. Existing rows are in the old vector dimension and semantic space.

## Recommended workflow

1. Verify the old and new model dimensions from model config or the embedding service, not by memory.
   - HuggingFace config `hidden_size` is a quick proxy for BGE dense embedding dim.
2. Check the DB vector column dimension, e.g. `memory_units.embedding` type (`vector(384)`, `vector(1024)`, etc.).
3. If the bank already contains production data and the user wants a unified bge-m3 bank:
   - backup the current/raw partial DB first;
   - reset or create a new bank/schema with the correct vector dimension;
   - rerun native retain / re-embed the full cleaned manifest from scratch.
4. If the user only wants future imports to use the new model:
   - keep the old bank separate, or explicitly accept mixed-era retrieval limitations;
   - do not silently append new-dimension embeddings to the old vector column.

## Pitfalls

- Changing `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-m3` alone does not convert old embeddings.
- A `vector(384)` column cannot store 1024-dim bge-m3 vectors without schema reset/migration.
- Even if dimensions matched, mixing embedding models degrades retrieval comparability because vectors live in different semantic spaces.
- After editing the local embedding model, restart the actual Hindsight container/service and verify the runtime environment/config; do not assume the new env is active just because a file changed.
- Hindsight `/health` may become OK while migrations or schema/vector dimensions are still wrong; verify DB schema (`memory_units.embedding`) separately after reset/migration.
- bge-m3 can take longer to load and may need HuggingFace/proxy access on first use; allow for delayed health checks and inspect logs before declaring startup failure.
- In Docker wrapper deployments, do not rely on the container writable layer for HuggingFace cache: recreate/remove loses downloaded model files. Mount persistent host caches such as `$HOME/.cache/huggingface:$HOME/.cache/huggingface` and `$HOME/.cache/torch:$HOME/.cache/torch`, and set `HINDSIGHT_API_STARTUP_WAIT_SECONDS` above the default 300s (e.g. 1200) for first bge-m3 load.
- For paid/full retain runs, preserve the user's established manifest hygiene: user input + Hermes assistant output only; exclude tool/command/search/thinking traces.

## Runtime verification after restart

Use a two-layer check before re-retain:

1. Runtime env/config points at bge-m3:
   - `docker inspect <hindsight_container> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL`
   - or check the wrapper-generated env/start command and `/banks/<bank>/config` if exposed by the API.
2. DB/vector schema matches bge-m3 after reset/migration:
   - Hindsight's base SQLAlchemy model/default migration can still create `vector(384)` because `DEFAULT_EMBEDDING_DIMENSION=384`.
   - after DB reset, explicitly run/admin-enforce embedding dimension 1024, e.g. `hindsight-api run-db-migration --embedding-dimension 1024` or equivalent `ensure_embedding_dimension(..., 1024)` path.
   - inspect `memory_units.embedding` type; for bge-m3 it should be `vector(1024)`, not `vector(384)`.
3. Optional smoke: create/retain one tiny test document and verify no embedding-dimension error appears in container logs.

Only after all three are clean should a production reset + full cleaned native retain proceed.
