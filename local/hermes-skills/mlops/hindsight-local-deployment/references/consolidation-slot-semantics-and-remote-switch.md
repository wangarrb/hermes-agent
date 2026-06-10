# Hindsight consolidation slot semantics and remote-switch notes

Session-derived note for future recovery/debugging.

## What we observed

- During the 2026-05-11 recovery run, `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=1` and `HINDSIGHT_API_WORKER_MAX_SLOTS=9` were in effect.
- The worker poller reserves consolidation capacity, but same-bank consolidation is still effectively serialized:
  - async submit uses `dedupe_by_bank=True` for consolidation
  - worker claim logic refuses a second consolidation when one is already `processing` for the same bank
- The consolidator itself processes `memory_units` in ordered batches and can split batches on LLM failure, so one job may stay in `llm.openai.consolidation+structured` for minutes.
- The bottleneck is therefore usually not queue starvation but the active LLM batch + DB update path for one bank.

## Practical conclusion

- Distinguish three knobs:
  - `HINDSIGHT_API_WORKER_MAX_SLOTS`: total worker task slots.
  - `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS`: reserved consolidation slots.
  - `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT`: in-task LLM concurrency.
- The upstream v0.6.1 default for `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS` is 2 (`config.py` reservation default), but the local paid/minimax wrapper intentionally set it to 1 when observations are enabled: retain concurrency 8 + one extra consolidation slot → `WORKER_MAX_SLOTS=9`, `WORKER_CONSOLIDATION_MAX_SLOTS=1`. This was a conservative safety choice to avoid consolidation starving retain/observation work.
- Setting consolidation reserved slots from 1 → 2 is reasonable for normal runtime if provider/DB capacity is healthy; pair it with `WORKER_MAX_SLOTS = retain_concurrency + consolidation_reserved_slots` (e.g. 8 + 2 = 10) so retain capacity is not stolen.
- Do not expect a slot increase to speed up an already-running single-bank consolidation: Hindsight claim logic excludes banks that already have `operation_type='consolidation' AND status='processing'`, and async submit uses `dedupe_by_bank=True` for pending consolidation. More slots mainly help multiple banks or mixed queued work, not one active `hermes` consolidation job.
- Running multiple same-bank consolidation jobs in parallel is unsafe unless Hindsight adds an explicit bank-level partition/claim design; it risks duplicate work and repeated observation generation.
- If consolidation is slow but healthy (`failed_base=0`, `processing=1`), the safest move is to wait, keep a monitor running, and only change slots/provider/proxy after the queue drains or with explicit interruption approval.

## Remote-switch lesson

- `normal-local` may be used as a command name even when the actual target is precision remote mode.
- In the 2026-05-11 run, switching to bailian/glm-5 succeeded only after the in-flight consolidation reached idle, then the container was recreated and the global proxy env was removed.
- Keep temporary in-container proxy bridges only as a bridge for in-flight work; remove them after idle/restart.
