#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="$ROOT/local/lib/reviewer_mode_switch.sh"
SWITCH="$ROOT/local/bin/hermes-kanban-switch-reviewer-mode"

idle_screen=$'previous output\n\n› Run /review on my current changes\n\n  gpt-5.6-sol high · master · Context 20% used'
idle_ansi_screen=$'previous output\n\n\033[1m›\033[m \033[2mRun /review on my current changes\033[m\n\n  gpt-5.6-sol high · master · Context 20% used'
busy_screen=$'• Working (20s • esc to interrupt)\n\n› Run /review on my current changes\n\n  gpt-5.6-sol high · master · Context 20% used'
typed_screen=$'previous output\n\n› partially typed request\n\n  gpt-5.6-sol high · master · Context 20% used'
typed_ansi_screen=$'previous output\n\n\033[1m›\033[m partially typed request\n\n  gpt-5.6-sol high · master · Context 20% used'
no_prompt_screen=$'previous output\n\n  gpt-5.6-sol high · master · Context 20% used'
hermes_idle_screen=$'previous output\n\ncoordinator ❯'
hermes_busy_screen=$'⚙ wait proc_abc 180s\n\n⚕ ❯ msg=interrupt · /queue · /bg'

source "$HELPER"

reviewer_screen_is_idle "$idle_screen" "$idle_ansi_screen"
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
if reviewer_screen_is_idle "$typed_screen" "$typed_ansi_screen"; then
    echo "non-empty Codex composer unexpectedly accepted" >&2
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
switch_source="$(<"$SWITCH")"
[[ "$switch_source" == *"action focus-pane-id"* ]]
[[ "$switch_source" != *"action focus-pane --pane-id"* ]]
[[ "$switch_source" == *'action focus-pane-id "$PANE_ID" 2>/dev/null || true'* ]]
[[ "$switch_source" == *'nohup setsid "$SELF"'* ]]
[[ "$switch_source" == *"CHECKS=1"* ]]
[[ "$switch_source" == *"reviewer_screen_is_idle \"\$FINAL_SCREEN\" \"\$FINAL_ANSI_SCREEN\""* ]]

echo "reviewer mode switch safety: PASS"
