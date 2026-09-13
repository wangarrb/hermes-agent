#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="$ROOT/local/bin/hermes-kanban-owner-workspace"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
assert_eq() { [[ "$1" == "$2" ]] || fail "expected '$2', got '$1'"; }
assert_contains() { [[ "$1" == *"$2"* ]] || fail "missing '$2' in: $1"; }

new_repo() {
    local path="$1"
    git init -q -b master "$path"
    git -C "$path" config user.email test@example.com
    git -C "$path" config user.name Test
    printf 'base\n' >"$path/base.txt"
    git -C "$path" add base.txt
    git -C "$path" commit -qm base
}

PRIMARY="$TMP/AnyRepo"
new_repo "$PRIMARY"
BASE="$(git -C "$PRIMARY" rev-parse HEAD)"

dry="$($HELPER prepare --role designer --workspace "$PRIMARY" --dry-run)"
assert_contains "$dry" "workspace=$TMP/AnyRepo-designer"
assert_contains "$dry" "branch=designer/mainline"
assert_contains "$dry" "base=$BASE"
assert_contains "$dry" "integration_branch=master"
[[ ! -e "$TMP/AnyRepo-designer" ]] || fail "dry-run created a worktree"
! git -C "$PRIMARY" show-ref --verify --quiet refs/heads/designer/mainline || fail "dry-run created a branch"

$HELPER prepare --role designer --workspace "$PRIMARY" >/dev/null
assert_eq "$(git -C "$TMP/AnyRepo-designer" branch --show-current)" "designer/mainline"
git -C "$PRIMARY" worktree list --porcelain | grep -Fq "worktree $TMP/AnyRepo-designer" || fail "designer worktree not linked"

printf 'role\n' >"$TMP/AnyRepo-designer/role.txt"
git -C "$TMP/AnyRepo-designer" add role.txt
git -C "$TMP/AnyRepo-designer" commit -qm role-only
ROLE_ONLY="$(git -C "$TMP/AnyRepo-designer" rev-parse HEAD)"
printf 'main\n' >"$PRIMARY/main.txt"
git -C "$PRIMARY" add main.txt
git -C "$PRIMARY" commit -qm main-advance
MAIN_ADVANCE="$(git -C "$PRIMARY" rev-parse HEAD)"
$HELPER sync --role designer --workspace "$PRIMARY" >/dev/null
git -C "$TMP/AnyRepo-designer" merge-base --is-ancestor "$ROLE_ONLY" HEAD || fail "role-only commit was lost"
git -C "$TMP/AnyRepo-designer" merge-base --is-ancestor "$MAIN_ADVANCE" HEAD || fail "integration commit not merged"

$HELPER prepare --role coordinator --workspace "$PRIMARY" >/dev/null
assert_eq "$(git -C "$TMP/AnyRepo-coordinator" branch --show-current)" "coordinator/mainline"

printf 'dirty\n' >>"$TMP/AnyRepo-coordinator/base.txt"
prepare_dirty="$($HELPER prepare --role coordinator --workspace "$PRIMARY")"
assert_contains "$prepare_dirty" "status=DIRTY_NOT_SYNCED"
if $HELPER sync --role coordinator --workspace "$PRIMARY" >/dev/null 2>&1; then
    fail "dirty worktree was accepted"
fi
git -C "$TMP/AnyRepo-coordinator" restore base.txt

git -C "$TMP/AnyRepo-coordinator" checkout -qb coordinator/wrong
if $HELPER sync --role coordinator --workspace "$PRIMARY" >/dev/null 2>&1; then
    fail "unexpected branch was accepted"
fi
git -C "$TMP/AnyRepo-coordinator" checkout -q coordinator/mainline

UNRELATED_PRIMARY="$TMP/OtherRoot/AnyRepo"
mkdir -p "$TMP/OtherRoot"
new_repo "$UNRELATED_PRIMARY"
UNRELATED_TARGET="$TMP/OtherRoot/AnyRepo-designer"
new_repo "$UNRELATED_TARGET"
before="$(git -C "$UNRELATED_TARGET" rev-parse HEAD):$(git -C "$UNRELATED_TARGET" status --porcelain)"
if $HELPER prepare --role designer --workspace "$UNRELATED_PRIMARY" >/dev/null 2>&1; then
    fail "unrelated repository target was accepted"
fi
after="$(git -C "$UNRELATED_TARGET" rev-parse HEAD):$(git -C "$UNRELATED_TARGET" status --porcelain)"
assert_eq "$after" "$before"

FAIL_PRIMARY="$TMP/FailRepo"
new_repo "$FAIL_PRIMARY"
if HERMES_OWNER_WORKSPACE_FAIL_AFTER_CREATE_FOR_TEST=1 \
    $HELPER prepare --role designer --workspace "$FAIL_PRIMARY" >/dev/null 2>&1; then
    fail "forced post-create failure unexpectedly succeeded"
fi
[[ ! -e "$TMP/FailRepo-designer" ]] || fail "partial worktree was not cleaned"
! git -C "$FAIL_PRIMARY" show-ref --verify --quiet refs/heads/designer/mainline || fail "partial branch was not cleaned"

CONFLICT_PRIMARY="$TMP/ConflictRepo"
new_repo "$CONFLICT_PRIMARY"
$HELPER prepare --role designer --workspace "$CONFLICT_PRIMARY" >/dev/null
printf 'designer\n' >"$TMP/ConflictRepo-designer/base.txt"
git -C "$TMP/ConflictRepo-designer" commit -qam designer-conflict
CONFLICT_BEFORE="$(git -C "$TMP/ConflictRepo-designer" rev-parse HEAD)"
printf 'master\n' >"$CONFLICT_PRIMARY/base.txt"
git -C "$CONFLICT_PRIMARY" commit -qam master-conflict
if $HELPER sync --role designer --workspace "$CONFLICT_PRIMARY" >/dev/null 2>&1; then
    fail "merge conflict unexpectedly succeeded"
fi
assert_eq "$(git -C "$TMP/ConflictRepo-designer" rev-parse HEAD)" "$CONFLICT_BEFORE"
[[ ! -e "$TMP/ConflictRepo-designer/.git/MERGE_HEAD" ]] || fail "merge conflict was not aborted"

OWNER_PRIMARY="$TMP/OwnerRepo"
new_repo "$OWNER_PRIMARY"
mkdir -p "$OWNER_PRIMARY/docs/roadmap/owners"
printf 'log v1\n' >"$OWNER_PRIMARY/docs/roadmap/owners/designer.md"
git -C "$OWNER_PRIMARY" add docs/roadmap/owners/designer.md
git -C "$OWNER_PRIMARY" commit -qm owner-log-base
$HELPER prepare --role designer --workspace "$OWNER_PRIMARY" >/dev/null
printf 'role update\n' >>"$TMP/OwnerRepo-designer/docs/roadmap/owners/designer.md"
git -C "$TMP/OwnerRepo-designer" commit -qam role-log-update
printf 'main update\n' >>"$OWNER_PRIMARY/docs/roadmap/owners/designer.md"
git -C "$OWNER_PRIMARY" commit -qam main-log-update
$HELPER sync --role designer --workspace "$OWNER_PRIMARY" >/dev/null 2>&1 || fail "owner-log conflict blocked sync"
OWNER_LOG="$(cat "$TMP/OwnerRepo-designer/docs/roadmap/owners/designer.md")"
assert_contains "$OWNER_LOG" "role update"
assert_contains "$OWNER_LOG" "main update"
[[ ! -e "$TMP/OwnerRepo-designer/.git/MERGE_HEAD" ]] || fail "owner-log merge left MERGE_HEAD"

# With the union merge driver present in .gitattributes the same two-sided log
# update merges natively (no script fallback needed).
ATTR_PRIMARY="$TMP/AttrRepo"
new_repo "$ATTR_PRIMARY"
mkdir -p "$ATTR_PRIMARY/docs/roadmap/owners"
printf 'docs/roadmap/owners/*.md merge=union\n' >"$ATTR_PRIMARY/.gitattributes"
printf 'log v1\n' >"$ATTR_PRIMARY/docs/roadmap/owners/designer.md"
git -C "$ATTR_PRIMARY" add .gitattributes docs/roadmap/owners/designer.md
git -C "$ATTR_PRIMARY" commit -qm attr-base
$HELPER prepare --role designer --workspace "$ATTR_PRIMARY" >/dev/null
printf 'role update\n' >>"$TMP/AttrRepo-designer/docs/roadmap/owners/designer.md"
git -C "$TMP/AttrRepo-designer" commit -qam role-log-update
printf 'main update\n' >>"$ATTR_PRIMARY/docs/roadmap/owners/designer.md"
git -C "$ATTR_PRIMARY" commit -qam main-log-update
$HELPER sync --role designer --workspace "$ATTR_PRIMARY" >/dev/null 2>&1 || fail "union-attr owner-log merge blocked sync"
ATTR_LOG="$(cat "$TMP/AttrRepo-designer/docs/roadmap/owners/designer.md")"
assert_contains "$ATTR_LOG" "role update"
assert_contains "$ATTR_LOG" "main update"

echo "PASS: owner workspace contract"
