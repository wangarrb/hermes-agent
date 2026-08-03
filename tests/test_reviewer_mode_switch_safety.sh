#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="$ROOT/local/lib/reviewer_mode_switch.sh"
SWITCH="$ROOT/local/bin/hermes-kanban-switch-reviewer-mode"

idle_screen=$'previous output\n\n› Run /review on my current changes\n\n  gpt-5.6-sol high · master · Context 20% used'
busy_screen=$'• Working (20s • esc to interrupt)\n\n› Run /review on my current changes\n\n  gpt-5.6-sol high · master · Context 20% used'
no_prompt_screen=$'previous output\n\n  gpt-5.6-sol high · master · Context 20% used'
hermes_idle_screen=$'previous output\n\ncoordinator ❯'
hermes_busy_screen=$'⚙ wait proc_abc 180s\n\n⚕ ❯ msg=interrupt · /queue · /bg'

source "$HELPER"

reviewer_screen_is_idle "$idle_screen"
reviewer_screen_is_idle "$hermes_idle_screen"
if reviewer_screen_is_idle "$busy_screen"; then
    echo "busy screen unexpectedly accepted" >&2
    exit 1
fi
if reviewer_screen_is_idle "$hermes_busy_screen"; then
    echo "busy Hermes screen unexpectedly accepted" >&2
    exit 1
fi
if reviewer_screen_is_idle "$no_prompt_screen"; then
    echo "screen without composer unexpectedly accepted" >&2
    exit 1
fi

sig1="$(reviewer_screen_signature "$idle_screen")"
sig2="$(reviewer_screen_signature "$idle_screen")"
sig3="$(reviewer_screen_signature $'previous output\n\n› typed text\n\n  gpt-5.6-sol high · master · Context 20% used')"
[[ "$sig1" == "$sig2" ]]
[[ "$sig1" != "$sig3" ]]

dry_run="$(bash "$SWITCH" --board egomotion4d --mode balanced --dry-run)"
[[ "$dry_run" == *"mode=balanced model=gpt-5.6-sol effort=high"* ]]
[[ "$dry_run" == *"HERMES_REVIEWER_MODE=balanced"* ]]
[[ "$dry_run" == *"--model gpt-5.6-sol"* ]]
[[ "$dry_run" == *"model_reasoning_effort="* ]]

echo "reviewer mode switch safety: PASS"
