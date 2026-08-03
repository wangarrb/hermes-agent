#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$ROOT/local/lib/owner_project_switch.sh"
SWITCH="$ROOT/local/bin/hermes-kanban-switch-owner-project"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
assert_contains() { [[ "$1" == *"$2"* ]] || fail "missing '$2' in: $1"; }

# shellcheck source=/dev/null
source "$LIB"
for role in planner designer coordinator; do owner_switch_validate_role "$role"; done
if owner_switch_validate_role reviewer >/dev/null 2>&1; then fail "reviewer accepted as owner"; fi

mkdir -p "$TMP/bin" "$TMP/Project"
git init -q -b master "$TMP/Project"
git -C "$TMP/Project" config user.email test@example.com
git -C "$TMP/Project" config user.name Test
git -C "$TMP/Project" commit --allow-empty -qm base

cat >"$TMP/bin/hermes" <<EOF
#!/usr/bin/env bash
if [[ "\$1 \$2" == "project show" ]]; then
  cat <<OUT
target [p_1]
  name: Target
  board: target-board
  primary: $TMP/Project
OUT
  exit 0
fi
exit 2
EOF
chmod +x "$TMP/bin/hermes"

for role in planner designer coordinator; do
    out="$(PATH="$TMP/bin:$PATH" "$SWITCH" --role "$role" --project target --source-board source-board --dry-run)"
    assert_contains "$out" "role=$role"
    assert_contains "$out" "target_board=target-board"
    case "$role" in
        planner) expected="$TMP/Project" ;;
        *) expected="$TMP/Project-$role" ;;
    esac
    assert_contains "$out" "target_workspace=$expected"
    assert_contains "$out" "HERMES_KANBAN_FRESH=1"
    assert_contains "$out" "-p $role"
done

if PATH="$TMP/bin:$PATH" "$SWITCH" --role planner --project target \
    --target-board wrong --source-board source-board --dry-run >/dev/null 2>&1; then
    fail "project/board mismatch accepted"
fi
if PATH="$TMP/bin:$PATH" "$SWITCH" --role planner --project target \
    --target-workspace "$TMP/Wrong" --source-board source-board --dry-run >/dev/null 2>&1; then
    fail "project/workspace mismatch accepted"
fi

unbound="$(PATH="$TMP/bin:$PATH" "$SWITCH" --role planner \
    --target-board explicit-board --target-workspace "$TMP/Project" \
    --source-board source-board --dry-run)"
assert_contains "$unbound" "binding=UNBOUND_EXPLICIT"

# Two-phase safety: preparation may finish, but a changed source pane identity
# must prevent replacement while preserving the prepared target.
cat >"$TMP/bin/hermes" <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == "kanban" ]]; then
  printf '[]\n'
  exit 0
fi
exit 2
EOF
cat >"$TMP/bin/zellij" <<'EOF'
#!/usr/bin/env bash
state="${FAKE_ZELLIJ_STATE:?}"
if [[ "$*" == *"list-panes"* ]]; then
  n=0; [[ ! -f "$state" ]] || n="$(<"$state")"; n=$((n + 1)); printf '%s' "$n" >"$state"
  if (( n >= 4 )); then id=8; else id=7; fi
  printf '[{"id":%s,"name":"","terminal_command":"bash -lc hermes-kanban-continue -p designer","is_plugin":false,"is_focused":true,"pane_x":0,"pane_y":0,"pane_rows":20,"pane_columns":80}]\n' "$id"
  exit 0
fi
if [[ "$*" == *"dump-screen"* ]]; then
  printf 'previous output\n\n› \n\n  idle\n'
  exit 0
fi
echo "unexpected zellij call: $*" >&2
exit 9
EOF
chmod +x "$TMP/bin/hermes" "$TMP/bin/zellij"
bash -c 'exec -a "listener --profile reviewer --board target-board" sleep 20' &
reviewer_pid=$!
set +e
switch_error="$(FAKE_ZELLIJ_STATE="$TMP/zellij-state" PATH="$TMP/bin:$PATH" \
    "$SWITCH" --role designer --source-board source-board \
    --target-board target-board --target-workspace "$TMP/Project" \
    --session source-session --confirm-background-work-clear \
    --checks 1 --interval-s 0 --timeout-s 2 --worker 2>&1)"
switch_rc=$?
set -e
kill "$reviewer_pid" 2>/dev/null || true
wait "$reviewer_pid" 2>/dev/null || true
[[ "$switch_rc" == "4" ]] || fail "expected PREPARED_NOT_SWITCHED rc=4, got $switch_rc: $switch_error"
assert_contains "$switch_error" "PREPARED_NOT_SWITCHED"
[[ -d "$TMP/Project-designer" ]] || fail "prepared target was removed after final recheck failure"
assert_contains "$(git -C "$TMP/Project" worktree list --porcelain)" "worktree $TMP/Project-designer"

source_text="$(<"$SWITCH")"
assert_contains "$source_text" "--confirm-background-work-clear"
assert_contains "$source_text" "PREPARED_NOT_SWITCHED"
assert_contains "$source_text" "owner_switch_has_live_reviewer"
assert_contains "$source_text" "owner_switch_has_running_task"
assert_contains "$source_text" "wait_for_reviewer_switch_boundary"
assert_contains "$source_text" "owner_switch_pane_fingerprint"
assert_contains "$source_text" "--close-replaced-pane"
assert_contains "$source_text" "action focus-pane-id"
if [[ "$source_text" == *"action focus-pane --pane-id"* ]]; then
    fail "unsupported zellij focus-pane subcommand remains"
fi

echo "PASS: owner project switch contract"
