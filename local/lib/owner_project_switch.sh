#!/usr/bin/env bash

owner_switch_validate_role() {
    case "${1:-}" in
        planner|designer|coordinator) return 0 ;;
        *) echo "invalid owner role: ${1:-<empty>}" >&2; return 2 ;;
    esac
}

owner_switch_resolve_project() {
    local project="$1" explicit_board="${2:-}" explicit_workspace="${3:-}"
    local details board primary
    details="$(hermes project show "$project")" || return
    board="$(printf '%s\n' "$details" | sed -n 's/^  board:[[:space:]]*//p' | head -n 1)"
    primary="$(printf '%s\n' "$details" | sed -n 's/^  primary:[[:space:]]*//p' | head -n 1)"
    [[ -n "$board" && -n "$primary" ]] || {
        echo "project must have both board and primary bindings: $project" >&2
        return 2
    }
    primary="$(readlink -m "$primary")"
    if [[ -n "$explicit_board" && "$explicit_board" != "$board" ]]; then
        echo "target board does not match project binding: $explicit_board != $board" >&2
        return 2
    fi
    if [[ -n "$explicit_workspace" && "$(readlink -m "$explicit_workspace")" != "$primary" ]]; then
        echo "target workspace does not match project binding" >&2
        return 2
    fi
    printf '%s\t%s\tBOUND\n' "$board" "$primary"
}

owner_switch_find_pane_id() {
    local session="$1" role="$2" panes count
    panes="$(zellij --session "$session" action list-panes --all --json)" || return 1
    count="$(printf '%s' "$panes" | jq --arg role "$role" '[.[] | select(.is_plugin == false) | select(((.name // "") | startswith($role + "-")) or ((.terminal_command // "") | contains("--profile " + $role)) or ((.terminal_command // "") | contains("-p " + $role)))] | length')"
    [[ "$count" == "1" ]] || {
        echo "expected exactly one $role pane in $session, found $count" >&2
        return 1
    }
    printf '%s' "$panes" | jq -r --arg role "$role" '.[] | select(.is_plugin == false) | select(((.name // "") | startswith($role + "-")) or ((.terminal_command // "") | contains("--profile " + $role)) or ((.terminal_command // "") | contains("-p " + $role))) | .id'
}

owner_switch_pane_fingerprint() {
    local session="$1" pane_id="$2"
    zellij --session "$session" action list-panes --all --json |
        jq -c --argjson pane "$pane_id" '.[] | select(.id == $pane) | [.id, .name, .terminal_command, .pane_x, .pane_y, .pane_rows, .pane_columns]'
}

owner_switch_has_live_reviewer() {
    local board="$1" pid args
    while read -r pid args; do
        [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
        if [[ "$args" == *"--profile reviewer"* && "$args" == *"--board $board"* ]]; then
            return 0
        fi
    done < <(ps -eo pid=,args=)
    return 1
}

owner_switch_has_running_task() {
    local board="$1" role="$2" payload
    payload="$(hermes kanban --board "$board" list --status running --assignee "$role" --json)" || return 0
    [[ "$(printf '%s' "$payload" | jq 'length')" != "0" ]]
}
