#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# gh-actions-run.sh
# GitHub CLI helper — view, log, and query a GitHub Actions run.
#
# Usage:
#   scripts/gh-actions-run.sh [OPTIONS] <run-id>
#
# Options:
#   -R, --repo  owner/repo   Target repository (default: QueenFi703/amazon-iap-kotlin)
#       --log                Print full run logs
#       --json               Print run status as JSON
#   -h, --help               Show this help
#
# Examples:
#   scripts/gh-actions-run.sh 23652704571
#   scripts/gh-actions-run.sh -R QueenFi703/amazon-iap-kotlin 23652704571
#   scripts/gh-actions-run.sh --log 23652704571
#   scripts/gh-actions-run.sh --json 23652704571
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────
DEFAULT_REPO="QueenFi703/amazon-iap-kotlin"
REPO="${DEFAULT_REPO}"
SHOW_LOG=false
SHOW_JSON=false
RUN_ID=""

# ─────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────
usage() {
  sed -n '/^# Usage:/,/^# ━/{ /^# ━/d; s/^# //; s/^#$//; p }' "$0"
  exit 0
}

# ─────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -R|--repo)
      REPO="${2:?'-R/--repo requires an owner/repo argument'}"
      shift 2
      ;;
    --log)
      SHOW_LOG=true
      shift
      ;;
    --json)
      SHOW_JSON=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    -*)
      echo "❌  Unknown option: $1" >&2
      echo "Run with --help for usage." >&2
      exit 1
      ;;
    *)
      if [[ -n "${RUN_ID}" ]]; then
        echo "❌  Unexpected argument: $1" >&2
        exit 1
      fi
      RUN_ID="$1"
      shift
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  echo "❌  A run id is required." >&2
  echo "Run with --help for usage." >&2
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# Verify gh is available and authenticated
# ─────────────────────────────────────────────────────────────────────
if ! command -v gh &>/dev/null; then
  echo "❌  GitHub CLI (gh) is not installed." >&2
  echo "    Install it: https://cli.github.com/manual/installation" >&2
  exit 1
fi

if ! gh auth status &>/dev/null; then
  echo "❌  Not authenticated. Run: gh auth login" >&2
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍  GitHub Actions Run: ${RUN_ID}  (repo: ${REPO})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if "${SHOW_JSON}"; then
  echo "📄  JSON status:"
  gh run view "${RUN_ID}" -R "${REPO}" \
    --json status,conclusion,event,headBranch,headSha,createdAt,updatedAt,url,name
  echo ""
else
  echo "📋  Run summary:"
  gh run view "${RUN_ID}" -R "${REPO}"
  echo ""
fi

if "${SHOW_LOG}"; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📜  Run logs:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  gh run view "${RUN_ID}" -R "${REPO}" --log
fi
