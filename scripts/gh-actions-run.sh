#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# gh-actions-run.sh — GitHub CLI helper for GitHub Actions runs
#
# Usage:
#   ./scripts/gh-actions-run.sh [OPTIONS] <run-id-or-url>
#
# Options:
#   -R, --repo <owner/repo>   Target repository (overrides default)
#   -l, --log                 Print full run logs
#   -j, --json                Print run details as JSON
#   -h, --help                Show this help message
#
# Examples:
#   ./scripts/gh-actions-run.sh 23652704571 -R QueenFi703/amazon-iap-kotlin
#   ./scripts/gh-actions-run.sh --log 23652704571 -R QueenFi703/amazon-iap-kotlin
#   ./scripts/gh-actions-run.sh --json 23652704571 -R QueenFi703/amazon-iap-kotlin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────
REPO=""
SHOW_LOG=false
SHOW_JSON=false
RUN_REF=""

# ─────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <run-id-or-url>

GitHub CLI helper for querying GitHub Actions run status, logs, and JSON.

OPTIONS
  -R, --repo <owner/repo>   Target repository (e.g. QueenFi703/amazon-iap-kotlin)
  -l, --log                 Print full run logs
  -j, --json                Print run details as JSON
  -h, --help                Show this help

ARGUMENTS
  <run-id-or-url>           Numeric run id (e.g. 23652704571)
                            or run URL  (e.g. https://github.com/owner/repo/actions/runs/123)

PREREQUISITES
  gh must be installed and authenticated:
    gh auth login
    gh auth status

EXAMPLES
  # Summary for a specific run
  $(basename "$0") 23652704571 -R QueenFi703/amazon-iap-kotlin

  # Full logs
  $(basename "$0") --log 23652704571 -R QueenFi703/amazon-iap-kotlin

  # JSON output
  $(basename "$0") --json 23652704571 -R QueenFi703/amazon-iap-kotlin

  # Using a run URL directly
  $(basename "$0") https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571

OTHER COMMON OPERATIONS (run manually with gh):

  # List recent runs on main
  gh run list -R QueenFi703/amazon-iap-kotlin --branch main --limit 10

  # Re-run only the failed jobs
  gh run rerun 23652704571 -R QueenFi703/amazon-iap-kotlin --failed

  # Trigger a workflow_dispatch run
  gh workflow run "AWS Gradle Pipeline (S3 + CodeArtifact + Device Farm)" \\
    -R QueenFi703/amazon-iap-kotlin --ref main
EOF
}

# ─────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -R|--repo)
            REPO="$2"
            shift 2
            ;;
        -l|--log)
            SHOW_LOG=true
            shift
            ;;
        -j|--json)
            SHOW_JSON=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$RUN_REF" ]]; then
                echo "Error: unexpected argument '$1'" >&2
                usage >&2
                exit 1
            fi
            RUN_REF="$1"
            shift
            ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────
# Validate
# ─────────────────────────────────────────────────────────────────────
if [[ -z "$RUN_REF" ]]; then
    echo "Error: run id or URL is required." >&2
    usage >&2
    exit 1
fi

if ! command -v gh &>/dev/null; then
    echo "Error: 'gh' (GitHub CLI) is not installed." >&2
    echo "Install it from https://cli.github.com and run 'gh auth login'." >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# Extract run id and repo from a full URL if provided
# e.g. https://github.com/owner/repo/actions/runs/12345
# ─────────────────────────────────────────────────────────────────────
if [[ "$RUN_REF" =~ ^https://github\.com/([^/]+/[^/]+)/actions/runs/([0-9]+) ]]; then
    # Prefer the URL-embedded repo unless the caller explicitly passed -R
    if [[ -z "$REPO" ]]; then
        REPO="${BASH_REMATCH[1]}"
    fi
    RUN_ID="${BASH_REMATCH[2]}"
elif [[ "$RUN_REF" =~ ^[0-9]+$ ]]; then
    RUN_ID="$RUN_REF"
else
    echo "Error: '$RUN_REF' is not a valid run id or GitHub Actions run URL." >&2
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# Build common gh flags
# ─────────────────────────────────────────────────────────────────────
REPO_FLAGS=()
if [[ -n "$REPO" ]]; then
    REPO_FLAGS=(-R "$REPO")
fi

# ─────────────────────────────────────────────────────────────────────
# Execute
# ─────────────────────────────────────────────────────────────────────
if $SHOW_JSON; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Run JSON — id: $RUN_ID${REPO:+  repo: $REPO}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    gh run view "$RUN_ID" "${REPO_FLAGS[@]}" \
        --json status,conclusion,event,headBranch,headSha,createdAt,updatedAt,url,name,workflowName
    echo ""
    echo "Jobs:"
    gh run view "$RUN_ID" "${REPO_FLAGS[@]}" \
        --json jobs -q '.jobs[] | {name, status, conclusion, url}'
elif $SHOW_LOG; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Run logs — id: $RUN_ID${REPO:+  repo: $REPO}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    gh run view "$RUN_ID" "${REPO_FLAGS[@]}" --log
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Run summary — id: $RUN_ID${REPO:+  repo: $REPO}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    gh run view "$RUN_ID" "${REPO_FLAGS[@]}"
fi
