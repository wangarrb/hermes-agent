#!/usr/bin/env bash
set -euo pipefail

# Install read-only Hindsight consolidation operations helpers.
# Safe: backs up overwritten files and never edits .env.

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$SKILL_DIR/scripts"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
DST_DIR="${1:-$HERMES_HOME/scripts}"
BACKUP_ROOT="$DST_DIR/backups/hindsight-consolidation-operations-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$DST_DIR" "$BACKUP_ROOT"

copy_one() {
  local rel="$1"
  local src="$SRC_DIR/$rel"
  local dst="$DST_DIR/$rel"
  if [[ ! -f "$src" ]]; then
    echo "missing packaged file: $src" >&2
    return 1
  fi
  if [[ -f "$dst" ]]; then
    cp -p "$dst" "$BACKUP_ROOT/$rel"
  fi
  cp -p "$src" "$dst"
}

copy_one hindsight_consolidation_status.py
chmod +x "$DST_DIR/hindsight_consolidation_status.py"
python3 -m py_compile "$DST_DIR/hindsight_consolidation_status.py"

cat <<EOF
Installed read-only Hindsight consolidation helper to: $DST_DIR/hindsight_consolidation_status.py
Backups of overwritten files, if any: $BACKUP_ROOT
Verify API-only mode:
  python3 $DST_DIR/hindsight_consolidation_status.py --skip-psql --json
EOF
