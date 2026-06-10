# Offline zero-unit documents and observations repair

Use when Hindsight offline reflect/consolidation documents exist but `memory_units` are missing, or when `observations=0` / native consolidation is not producing high-level observations after a paid/offline backfill.

## Signals

- `documents` contains many `hermes-offline-consolidation::...` rows but `memory_units.document_id` has few/no matching rows.
- Weekly/history-through documents are present but have `0` units.
- Native consolidation jobs complete or process slowly while `fact_type='observation'` stays at `0`.
- Container logs or async operation errors suggest provider connection trouble, even though the host can reach the provider.

## Safe inspection queries

Use production psql path unless overridden:

```bash
PSQL=${HINDSIGHT_PSQL:-$HOME/.hindsight-docker/installation/18.1.0/bin/psql}
$PSQL -h /tmp -p 5432 -U hindsight -d hindsight -q -t -A -F $'\t' -c "
with offline_docs as (
  select d.id, coalesce(count(m.id),0) units
  from documents d
  left join memory_units m on m.bank_id=d.bank_id and m.document_id=d.id
  where d.bank_id='hermes' and d.id like 'hermes-offline-consolidation::%'
  group by d.id
)
select count(*) total,
       count(*) filter(where units=0) zero,
       count(*) filter(where units>0) with_units,
       sum(units) units
from offline_docs;

select count(*) filter(where fact_type='observation') observations,
       count(*) filter(where fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null) unconsolidated,
       count(*) filter(where fact_type in ('world','experience') and consolidation_failed_at is not null) failed
from memory_units where bank_id='hermes';
"
```

Before repair, take a local snapshot of affected documents:

```bash
mkdir -p $HOME/.hermes/hindsight/snapshots
$PSQL -h /tmp -p 5432 -U hindsight -d hindsight -q -t -A -c "
copy (
  select jsonb_agg(to_jsonb(d))
  from documents d
  where d.bank_id='hermes' and d.id like 'hermes-offline-consolidation::%'
) to stdout
" > $HOME/.hermes/hindsight/snapshots/$(date +%Y%m%d)-offline-docs-before-repair.json
```

## Zero-unit repair approach

Root cause observed in production: offline consolidation documents may be written successfully while retain/fact extraction stores no `memory_units`, especially when the bank/provider path uses a verbatim or batch mode that returns no facts for these synthetic documents. Treat this as a document-to-unit archiving failure, not a missing offline document.

Preferred repair:

1. Enable observations only after confirming the LLM provider path is reachable.
2. For repairing synthetic offline summaries, set/override `retain_extraction_mode=chunks` so the already-summarized offline text is archived deterministically as memory chunks rather than relying on another LLM fact extraction pass.
3. Smoke one specific weekly/history-through document synchronously and verify `memory_units.document_id` increases.
4. Batch repair the rest; keep a JSON report and DB snapshot.
5. Re-run the offline-doc unit query; remaining zero docs are often empty/no substantive content and should be inspected individually.

Useful host-side artifacts from the proven workflow:

- `hindsight_offline_zero_unit_repair.py`: discover zero-unit offline docs, rebuild retain payload from `documents.original_text`, submit sync/async retain, and report results.
- `hindsight_bank_quality_audit.py`: ensure API lineage sparse cases fall back to PostgreSQL for accurate `docs_without_units`.

## Observations / native consolidation repair

After units exist:

1. Confirm bank config has `enable_observations=true` and an explicit observation mission focused on durable user/project facts, decisions, experiments, environment facts, workflows, risks, and open questions; exclude credentials, raw command logs, and transient progress chatter.
2. Trigger native consolidation with `POST /v1/default/banks/hermes/consolidate`.
3. Monitor `fact_type='observation'`, unconsolidated base units, failed base units, and async consolidation status.
4. If failures were marked before fixing provider/network, call `POST /v1/default/banks/hermes/consolidation/recover` and then trigger consolidation again.

## Docker proxy pitfall

For this user's Hindsight deployment, first check the network mode: it commonly runs with `NetworkMode=host`. In host-network mode, `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:7890` points at the host loopback. If the host Clash/proxy is not actually listening on 7890, provider calls can stall/fail even though direct internet egress would work.

Important user preference/architecture rule: do not keep a global proxy in the Hindsight container unless it is actually needed. Direct egress has been verified for DeepSeek, MiniMax, and DashScope; opencode-go or other special providers may need a proxy, but that should be provider/task-specific rather than container-global.

Diagnosis:

```bash
# Check whether the container has global proxy env and whether it is host-networked.
sg docker -c "docker inspect hindsight --format 'NetworkMode={{.HostConfig.NetworkMode}}'"
sg docker -c "docker inspect hindsight --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -i proxy || true"

# Test provider direct egress with proxy env removed. HTTP 401/403 from /models usually means network is reachable but auth is missing.
cat >/tmp/test_hindsight_direct_connectivity.sh <<'SH'
#!/bin/sh
for u in \
  https://api.deepseek.com/v1/models \
  https://api.minimax.chat/v1/models \
  https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
 do
  code=$(curl -sS -o /dev/null -w "%{http_code} %{time_total}" --max-time 10 "$u" 2>/dev/null || echo FAIL)
  echo "$u $code"
done
SH
sg docker -c "docker cp /tmp/test_hindsight_direct_connectivity.sh hindsight:/tmp/test_hindsight_direct_connectivity.sh && docker exec hindsight env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy sh /tmp/test_hindsight_direct_connectivity.sh"
```

If direct egress works when proxy env is removed, native consolidation failures are likely self-inflicted by stale/global proxy env. Preferred long-term fix: remove `HTTP_PROXY`/`HTTPS_PROXY` from the Hindsight container startup environment and only inject proxy env for provider calls that truly need it. Do not silently modify `.env`; get confirmation first.

Safe temporary workaround while a long consolidation is already running: run a minimal HTTP CONNECT proxy bound to `127.0.0.1:7890` in the same host-network namespace so the already-running Hindsight API/worker process can finish. This is temporary and should be removed after drain; it survives neither container recreate nor should be treated as desired steady state.

## Verification before reporting done

- Async queue has no failed/pending retain/consolidation jobs, or remaining processing is explicitly reported.
- Offline docs have expected unit coverage (`zero` near 0; inspect any remaining zeros).
- `observations` is nonzero and increasing under consolidation.
- `consolidation_failed_at` for base facts is 0 or has been recovered/retried.
- Reports/snapshots are written under `$HOME/.hermes/hindsight/reports/` and `$HOME/.hermes/hindsight/snapshots/`.
