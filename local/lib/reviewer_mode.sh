#!/usr/bin/env bash

resolve_reviewer_mode() {
    local requested="${1:-balanced}"
    case "$requested" in
        economy)
            printf 'economy\tgpt-5.6-luna\tmax\n'
            ;;
        balanced|efficiency)
            printf 'balanced\tgpt-5.6-terra\tmax\n'
            ;;
        performance|high_precision)
            printf 'performance\tgpt-5.6-sol\tmax\n'
            ;;
        *)
            echo "reviewer mode must be economy, balanced, or performance: $requested" >&2
            return 2
            ;;
    esac
}
