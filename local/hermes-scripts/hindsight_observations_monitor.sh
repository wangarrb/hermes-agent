#!/usr/bin/env bash
set -euo pipefail
PSQL="${HINDSIGHT_PSQL:-/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql}"
LOG="/home/wyr/.hermes/logs/hindsight-observations/20260511-181633-drain.jsonl"
while true; do
  clear || true
  date '+%F %T %z'
  echo "Hindsight observations / consolidation monitor"
  echo "log: $LOG"
  echo
  "$PSQL" -h /tmp -p 5432 -U hindsight -d hindsight -q -t -A -F $'\t' -c "
    select
      count(*) filter(where fact_type='observation') as observations,
      coalesce(max(created_at) filter(where fact_type='observation')::text,'') as last_observation,
      count(*) filter(where fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null) as unconsolidated_base,
      count(*) filter(where fact_type in ('world','experience') and consolidation_failed_at is not null) as failed_base
    from memory_units where bank_id='hermes';
    select status, operation_type, count(*) from async_operations where bank_id='hermes' group by status,operation_type order by status,operation_type;
    with offline_docs as (
      select d.id,coalesce(count(m.id),0) units
      from documents d left join memory_units m on m.bank_id=d.bank_id and m.document_id=d.id
      where d.bank_id='hermes' and d.id like 'hermes-offline-consolidation::%'
      group by d.id
    ) select 'offline_docs', count(*) total, count(*) filter(where units=0) zero, count(*) filter(where units>0) with_units, coalesce(sum(units),0) units from offline_docs;
  " | sed 's/\t/    /g'
  echo
  python3 - <<'PY'
import json, datetime, pathlib
p=pathlib.Path('/home/wyr/.hermes/logs/hindsight-observations/20260511-181633-drain.jsonl')
rows=[]
if p.exists():
    for line in p.read_text().splitlines():
        line=line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except Exception: pass
print('monitor_samples:', len(rows))
if len(rows) >= 2:
    def parse(ts): return datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
    for n in (10,20,60,len(rows)):
        sub=rows[-n:]
        if len(sub)<2: continue
        mins=(parse(sub[-1]['ts'])-parse(sub[0]['ts'])).total_seconds()/60
        if mins<=0: continue
        du=sub[0].get('unconsolidated_base',0)-sub[-1].get('unconsolidated_base',0)
        do=sub[-1].get('observations',0)-sub[0].get('observations',0)
        rate=du/mins
        latest=sub[-1].get('unconsolidated_base',0)
        eta='unknown'
        if rate>0:
            eta_min=latest/rate
            eta=f'{eta_min:.1f} min ({eta_min/60:.1f} h)'
        print(f'window={n:>3} mins={mins:5.1f} uncon_delta={du:4} rate={rate:5.2f}/min obs_delta={do:4} obs_rate={do/mins:5.2f}/min ETA={eta}')
PY
  echo
  echo "Refresh: 30s. Attach: tmux attach -t hindsight-obs-monitor"
  sleep 30
done
