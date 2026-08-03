#!/usr/bin/env bash

owner_workspace_validate_role() {
    case "${1:-}" in
        planner|designer|coordinator) return 0 ;;
        *) echo "invalid owner role: ${1:-<empty>}" >&2; return 2 ;;
    esac
}

owner_workspace_primary() {
    git -C "$1" rev-parse --show-toplevel 2>/dev/null
}

owner_workspace_path() {
    local primary="$1" role="$2"
    owner_workspace_validate_role "$role" || return
    if [[ "$role" == "planner" ]]; then
        printf '%s\n' "$primary"
    else
        printf '%s-%s\n' "$primary" "$role"
    fi
}

owner_workspace_branch() {
    local primary="$1" role="$2"
    owner_workspace_validate_role "$role" || return
    if [[ "$role" == "planner" ]]; then
        git -C "$primary" branch --show-current
    else
        printf '%s/mainline\n' "$role"
    fi
}

owner_workspace_common_dir() {
    git -C "$1" rev-parse --path-format=absolute --git-common-dir 2>/dev/null
}

owner_workspace_is_linked_to() {
    local primary="$1" target="$2" primary_common target_common
    primary_common="$(owner_workspace_common_dir "$primary")" || return 1
    target_common="$(owner_workspace_common_dir "$target")" || return 1
    [[ "$primary_common" == "$target_common" ]] || return 1
    git -C "$primary" worktree list --porcelain |
        awk '/^worktree / {sub(/^worktree /, ""); print}' |
        grep -Fxq "$target"
}

owner_workspace_print_contract() {
    local primary="$1" role="$2" target="$3" branch="$4" base="$5" integration_branch="$6"
    printf 'role=%s\nworkspace=%s\nbranch=%s\nbase=%s\nintegration_branch=%s\nprimary=%s\n' \
        "$role" "$target" "$branch" "$base" "$integration_branch" "$primary"
}

owner_workspace_prepare() {
    local action="$1" role="$2" workspace="$3" dry_run="${4:-0}" target_override="${5:-}"
    local primary target branch base integration_branch created_worktree=0 created_branch=0

    owner_workspace_validate_role "$role" || return
    primary="$(owner_workspace_primary "$workspace")" || {
        echo "not a Git worktree: $workspace" >&2
        return 2
    }
    integration_branch="$(git -C "$primary" branch --show-current)"
    [[ -n "$integration_branch" ]] || {
        echo "primary workspace is detached: $primary" >&2
        return 2
    }
    base="$(git -C "$primary" rev-parse HEAD)" || return
    if [[ -n "$target_override" ]]; then
        target="$(readlink -m "$target_override")"
    else
        target="$(owner_workspace_path "$primary" "$role")" || return
    fi
    branch="$(owner_workspace_branch "$primary" "$role")" || return

    owner_workspace_print_contract "$primary" "$role" "$target" "$branch" "$base" "$integration_branch"
    [[ "$dry_run" == "1" ]] && return 0

    if [[ "$role" == "planner" ]]; then
        [[ -z "$(git -C "$primary" status --porcelain)" ]] || {
            echo "planner workspace is dirty: $primary" >&2
            return 3
        }
        return 0
    fi

    if [[ -e "$target" ]]; then
        owner_workspace_is_linked_to "$primary" "$target" || {
            echo "target exists but is not a linked worktree of primary: $target" >&2
            return 3
        }
    elif [[ "$action" == "sync" ]]; then
        echo "owner worktree does not exist: $target" >&2
        return 3
    else
        if git -C "$primary" show-ref --verify --quiet "refs/heads/$branch"; then
            git -C "$primary" worktree add "$target" "$branch" >/dev/null || return
        else
            git -C "$primary" worktree add -b "$branch" "$target" "$base" >/dev/null || return
            created_branch=1
        fi
        created_worktree=1
        if [[ "${HERMES_OWNER_WORKSPACE_FAIL_AFTER_CREATE_FOR_TEST:-0}" == "1" ]]; then
            git -C "$primary" worktree remove "$target" >/dev/null 2>&1 || true
            (( created_branch == 0 )) || git -C "$primary" branch -D "$branch" >/dev/null 2>&1 || true
            echo "forced post-create failure" >&2
            return 99
        fi
    fi

    if ! owner_workspace_is_linked_to "$primary" "$target"; then
        echo "owner worktree identity check failed: $target" >&2
        return 3
    fi
    if [[ "$(git -C "$target" branch --show-current)" != "$branch" ]]; then
        echo "owner worktree is on unexpected branch: $target" >&2
        return 3
    fi
    if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
        if [[ "$action" == "prepare" && "$created_worktree" == "0" ]]; then
            echo "status=DIRTY_NOT_SYNCED"
            return 0
        fi
        echo "owner worktree is dirty: $target" >&2
        return 3
    fi

    if ! git -C "$target" merge-base --is-ancestor "$base" HEAD; then
        if ! git -C "$target" merge --no-edit "$base" >/dev/null; then
            git -C "$target" merge --abort >/dev/null 2>&1 || true
            if (( created_worktree )); then
                git -C "$primary" worktree remove "$target" >/dev/null 2>&1 || true
                (( created_branch == 0 )) || git -C "$primary" branch -D "$branch" >/dev/null 2>&1 || true
            fi
            echo "failed to merge integration base $base into $branch" >&2
            return 4
        fi
    fi
}
