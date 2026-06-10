#!/usr/bin/env bash
set -euo pipefail

# Install packaged Hindsight pipeline helper scripts into a Hermes home.
# Safe by default: existing files are backed up, not overwritten destructively.
# Does not read or edit .env.

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$SKILL_DIR/scripts"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
DST_DIR="${1:-$HERMES_HOME/scripts}"
BACKUP_ROOT="$DST_DIR/backups/hindsight-local-deployment-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$DST_DIR" "$BACKUP_ROOT" "$DST_DIR/tests"

files=(
  hindsight_pipeline_common.py
  hindsight_pipeline_preflight.py
  hindsight_memory_pipeline.py
  hindsight_session_manifest.py
  hindsight_session_retain_runner.py
  hindsight_minimax_import.py
  patch_hindsight_consolidator_parallel.py
  hindsight_consolidator_parallel_patched.py
  patch_hindsight_retain_temporal_fk_guard.py
  offline_hindsight_reflect_consolidate.py
  hindsight_offline_v2_rebuild.py
  hindsight_conflict_core.py
  hindsight_conflict_audit.py
  hindsight_repair_proposal_build.py
  hindsight_proposal_review.py
  hindsight_native_client.py
  hindsight_consolidation_status.py
  hindsight_wait_native_consolidation.py
  import_sqlite_to_hindsight.py
  hindsight_progress_bar.py
  hindsight_progress_live.sh
  import_sessions_to_hindsight.py
)

test_files=(
  tests/test_hindsight_memory_pipeline.py
  tests/test_hindsight_pipeline_preflight.py
  tests/test_hindsight_native_client.py
  tests/test_hindsight_session_manifest.py
  tests/test_hindsight_minimax_import_session_manifest.py
  tests/test_hindsight_repair_proposal_build.py
  tests/test_hindsight_proposal_review.py
)

copy_one() {
  local rel="$1"
  local src="$SRC_DIR/$rel"
  local dst="$DST_DIR/$rel"
  if [[ ! -f "$src" ]]; then
    echo "missing packaged file: $src" >&2
    return 1
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "$rel")"
    cp -p "$dst" "$BACKUP_ROOT/$rel"
  fi
  cp -p "$src" "$dst"
}

for f in "${files[@]}"; do copy_one "$f"; done
for f in "${test_files[@]}"; do copy_one "$f"; done

chmod +x "$DST_DIR/hindsight_progress_live.sh" 2>/dev/null || true

# Syntax-check ALL deployed .py scripts (auto-discovers, no manual list maintenance).
if compgen -G "$DST_DIR/*.py" >/dev/null 2>&1; then
  python3 -m py_compile "$DST_DIR"/*.py
fi
if compgen -G "$DST_DIR/tests/*.py" >/dev/null 2>&1; then
  python3 -m py_compile "$DST_DIR"/tests/*.py
fi

if [[ ! -f "$HERMES_HOME/hindsight/pipeline_config.json" ]]; then
  python3 "$DST_DIR/hindsight_pipeline_preflight.py" --init-config --json
fi
python3 "$DST_DIR/hindsight_pipeline_preflight.py" --write-tuning --json >/tmp/hindsight_pipeline_preflight_install.json

cat <<EOF
Installed Hindsight pipeline scripts to: $DST_DIR
Backups of overwritten files, if any: $BACKUP_ROOT
Preflight output: /tmp/hindsight_pipeline_preflight_install.json
Next verification:
  python3 $DST_DIR/hindsight_pipeline_preflight.py --strict-runtime --json
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest $DST_DIR/tests/test_hindsight_memory_pipeline.py $DST_DIR/tests/test_hindsight_pipeline_preflight.py $DST_DIR/tests/test_hindsight_native_client.py $DST_DIR/tests/test_hindsight_session_manifest.py $DST_DIR/tests/test_hindsight_minimax_import_session_manifest.py $DST_DIR/tests/test_hindsight_repair_proposal_build.py $DST_DIR/tests/test_hindsight_proposal_review.py -q
EOF
