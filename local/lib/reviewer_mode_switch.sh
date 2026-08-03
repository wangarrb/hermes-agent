#!/usr/bin/env bash

reviewer_screen_is_idle() {
    local screen="$1"
    local tail
    tail="$(printf '%s\n' "$screen" | tail -n 12)"
    if printf '%s\n' "$tail" | grep -Eiq '•[[:space:]]*(working|thinking|running)|preparing process|wait proc_|msg=interrupt|[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]'; then
        return 1
    fi
    printf '%s\n' "$tail" | grep -Eq '(^[[:space:]]*›([[:space:]]|$))|(^[[:space:]]*([[:alnum:]_.-]+[[:space:]]+)?❯[[:space:]]*$)'
}

reviewer_screen_signature() {
    local screen="$1"
    printf '%s\n' "$screen" | tail -n 12 | sha256sum | awk '{print $1}'
}

find_reviewer_pane_id() {
    local session="$1"
    local panes count
    panes="$(zellij --session "$session" action list-panes --all --json)" || return 1
    count="$(printf '%s' "$panes" | jq '[.[] | select(.is_plugin == false) | select((.terminal_command // "") | contains("--profile reviewer"))] | length')"
    if [[ "$count" != "1" ]]; then
        echo "expected exactly one reviewer pane in $session, found $count" >&2
        return 1
    fi
    printf '%s' "$panes" | jq -r '.[] | select(.is_plugin == false) | select((.terminal_command // "") | contains("--profile reviewer")) | .id'
}

wait_for_reviewer_switch_boundary() {
    local session="$1"
    local pane_id="$2"
    local timeout_s="${3:-600}"
    local interval_s="${4:-10}"
    local required="${5:-3}"
    local deadline=$((SECONDS + timeout_s))
    local stable=0 previous="" screen signature

    while (( SECONDS < deadline )); do
        screen="$(zellij --session "$session" action dump-screen --pane-id "$pane_id" 2>/dev/null || true)"
        if reviewer_screen_is_idle "$screen"; then
            signature="$(reviewer_screen_signature "$screen")"
            if [[ -n "$previous" && "$signature" == "$previous" ]]; then
                stable=$((stable + 1))
            else
                stable=1
            fi
            previous="$signature"
            if (( stable >= required )); then
                return 0
            fi
        else
            stable=0
            previous=""
        fi
        sleep "$interval_s"
    done
    echo "reviewer pane did not reach a stable safe boundary within ${timeout_s}s" >&2
    return 1
}
