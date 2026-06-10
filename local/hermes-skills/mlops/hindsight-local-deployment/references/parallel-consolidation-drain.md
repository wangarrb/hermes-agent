- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- `async_operations` shows long-running `processing:consolidation` jobs.
- `memory_units` backlog (`consolidated_at is null`, `consolidation_failed_at is null`, base units) drains slowly while `failed_base=0`.
- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

1. Check live state before patching/restart:
   - container health: `curl -s http://127.0.0.1:8888/health`
   - operations: query `async_operations` grouped by `status, operation_type`.
   - backlog: count unconsolidated base `memory_units` and failed base units.
2. Patch code on a host-side copy first; run `python3 -m py_compile` before touching the container.
3. Copy patched `consolidator.py` into the live container only after syntax passes.
4. Do not interrupt an active `processing:consolidation` unless the user explicitly authorizes it. Prefer an idle watcher:
   - poll until `processing:consolidation == 0` and `pending:consolidation == 0`.
   - then `docker restart -t 30 hindsight`.
   - wait for `/health` to become healthy.
   - then start the drain script again.
- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Claim/select a consolidation set once, then split into smaller LLM batches.
- Run multiple LLM batches concurrently (`asyncio.gather`), but cap total in-flight LLM/recall passes with one shared job-level semaphore. Do not nest `PARALLEL_BATCHES` and per-batch scope semaphores without a global cap, or you get accidental `8 × 8` oversubscription.
- Use an `asyncio.Lock` around all write paths: create, update, delete observation actions.
- Add row-level locking or exact-match dedupe for writes where possible:
  - update/delete should lock the target observation row before mutation.
  - create should dedupe by exact text + tags/scope before insertion to avoid duplicate observations from parallel batches.
- Make concurrency configurable, e.g. `HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES`, default conservatively (3) rather than maxing paid concurrency.
- For large drains, expose fetch/job sizing through a small local tuning file rather than editing `.env`: keep `LLM_BATCH_SIZE` at the quality ceiling, raise fetch/job window to expose more runnable chunks, and keep the global semaphore at the paid concurrency ceiling.
- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- `python3 -m py_compile` on host copy and container copy.
- After restart, confirm the new code path is active by watching:
  - `async_operations` transitions,
  - backlog decreasing,
  - observations increasing,
  - `failed_base` staying 0,
  - no duplicate-observation explosion.
- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.

- Do not use global HTTP(S)_PROXY for Hindsight unless specifically needed; Docker-internal proxy assumptions can break direct providers.
- Do not auto-edit `.env` or insert keys.
- Do not restart while a current consolidation write is active unless user accepts possible partial/failed job cleanup.
- Under heavy parallel drain, external `psql`/monitor queries can hit `FATAL: sorry, too many clients already`. In this setup the API process can also expand its own DB pool to ~100 idle connections, so the error may persist until the pool naturally shrinks or the service is restarted. Treat it as a real capacity symptom, not just a flaky query, and make drain/monitor loops survive it instead of exiting.
