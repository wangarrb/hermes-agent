#!/usr/bin/env bash

reviewer_codex_composer_is_empty() {
    local ansi_screen="$1"
    local line plain prompt_line="" plain_prompt=""
    while IFS= read -r line; do
        plain="$(printf '%s' "$line" | sed -E $'s/\\x1B\\[[0-9;?]*[ -\\/]*[@-~]//g')"
        if [[ "$plain" =~ ^[[:space:]]*›([[:space:]]|$) ]]; then
            prompt_line="$line"
            plain_prompt="$plain"
        fi
    done <<<"$ansi_screen"
    [[ -n "$prompt_line" ]] || return 1

    plain_prompt="${plain_prompt#*›}"
    plain_prompt="${plain_prompt#${plain_prompt%%[![:space:]]*}}"
    [[ -z "$plain_prompt" ]] && return 0

    # Codex renders placeholder suggestions dim. User-entered composer text is
    # normal intensity, so it must block a destructive pane replacement.
    [[ "$prompt_line" == *$'\033[2m'* ]]
}

reviewer_screen_is_idle() {
    local screen="$1"
    local ansi_screen="${2:-}"
    local tail
    tail="$(printf '%s\n' "$screen" | tail -n 12)"
    if printf '%s\n' "$tail" | grep -Eiq '•[[:space:]]*(working|thinking|running)|preparing process|wait proc_|msg=interrupt|[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]'; then
        return 1
    fi
    if printf '%s\n' "$tail" | grep -Eiq 'gpt-[^[:space:]]+.*context.*%'; then
        [[ -n "$ansi_screen" ]] || return 1
        reviewer_codex_composer_is_empty "$ansi_screen" || return 1
        return 0
    fi
    printf '%s\n' "$tail" | grep -Eq '(^[[:space:]]*([[:alnum:]_.-]+[[:space:]]+)?❯[[:space:]]*$)'
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
    local stable=0 previous="" screen ansi_screen signature

    while (( SECONDS < deadline )); do
        screen="$(zellij --session "$session" action dump-screen --pane-id "$pane_id" 2>/dev/null || true)"
        ansi_screen="$(zellij --session "$session" action dump-screen --ansi --pane-id "$pane_id" 2>/dev/null || true)"
        if reviewer_screen_is_idle "$screen" "$ansi_screen"; then
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
