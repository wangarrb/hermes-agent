#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="$ROOT/local/lib/reviewer_mode.sh"

assert_mode() {
    local mode="$1"
    local expected="$2"
    local actual
    actual="$(bash -c 'source "$1"; resolve_reviewer_mode "$2"' reviewer-mode-test "$HELPER" "$mode")"
    if [[ "$actual" != "$expected" ]]; then
        echo "mode=$mode expected=$expected actual=$actual" >&2
        return 1
    fi
}

assert_mode economy $'economy\tgpt-5.6-luna\tmax'
assert_mode balanced $'balanced\tgpt-5.6-terra\tmax'
assert_mode performance $'performance\tgpt-5.6-sol\tmax'

if bash -c 'source "$1"; resolve_reviewer_mode "$2"' reviewer-mode-test "$HELPER" fastest >/dev/null 2>&1; then
    echo "unknown reviewer mode unexpectedly passed" >&2
    exit 1
fi

echo "reviewer mode mapping: PASS"
