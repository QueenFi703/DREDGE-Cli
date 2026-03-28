#!/usr/bin/env bash
# check-run.sh — Query GitHub Actions run status via the inspector app
#
# Usage:
#   ./scripts/check-run.sh <run_url>
#   ./scripts/check-run.sh --owner <owner> --repo <repo> --run-id <run_id>
#
# Examples:
#   ./scripts/check-run.sh \
#     "https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571"
#
#   ./scripts/check-run.sh \
#     --owner QueenFi703 --repo amazon-iap-kotlin --run-id 23652704571

set -euo pipefail

BASE_URL="${INSPECTOR_URL:-http://localhost:3003}"

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

if [[ $# -eq 0 ]]; then
  usage
fi

# ── Parse arguments ────────────────────────────────────────────────────────────
RUN_URL=""
OWNER=""
REPO=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner) OWNER="$2"; shift 2 ;;
    --repo)  REPO="$2";  shift 2 ;;
    --run-id|--run_id) RUN_ID="$2"; shift 2 ;;
    http*) RUN_URL="$1"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

# ── Build query string ─────────────────────────────────────────────────────────
if [[ -n "$RUN_URL" ]]; then
  QUERY="run_url=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote_plus(sys.argv[1]))" "$RUN_URL")"
elif [[ -n "$OWNER" && -n "$REPO" && -n "$RUN_ID" ]]; then
  QUERY="owner=${OWNER}&repo=${REPO}&run_id=${RUN_ID}"
else
  echo "Error: provide either a run URL or --owner, --repo, and --run-id."
  usage
fi

# ── Call the inspector ─────────────────────────────────────────────────────────
echo "Querying: ${BASE_URL}/actions/run?${QUERY}"
echo

curl -s --fail-with-body "${BASE_URL}/actions/run?${QUERY}" \
  | python3 -m json.tool
