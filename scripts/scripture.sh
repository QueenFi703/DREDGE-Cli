#!/usr/bin/env bash
# =========================================================
# DREDGE :: SCRIPTURE FUNNEL
# "The net gathers before the harvest."
# Drop-in telemetry + onboarding + monetization scripture
# =========================================================

set -euo pipefail

DREDGE_VERSION="${DREDGE_VERSION:-0.1.0}"
DREDGE_API="${DREDGE_API:-https://api.oriongateway.io}"
DREDGE_REPO="${DREDGE_REPO:-QueenFi703/DREDGE-Cli}"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
SESSION_ID="$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid || echo "unknown")"

echo ""
echo "═══════════════════════════════════════════════"
echo "        D R E D G E   A W A K E N S"
echo "═══════════════════════════════════════════════"
echo ""

payload() {
cat <<JSON
{
  "session_id":"$SESSION_ID",
  "repo":"$DREDGE_REPO",
  "version":"$DREDGE_VERSION",
  "os":"$OS",
  "arch":"$ARCH",
  "timestamp":"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
JSON
}

emit_event() {
  local endpoint="$1"

  curl -fsSL \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$(payload)" \
    "$DREDGE_API/$endpoint" \
    >/dev/null 2>&1 || true
}

echo "→ Emitting install telemetry..."
emit_event "v1/install"

echo ""
echo "Choose your role:"
echo ""
echo "  [1] Forge     → Builder"
echo "  [2] Sentinel  → Security"
echo "  [3] Oracle    → Analysis"
echo "  [4] Gate      → Infrastructure"
echo "  [5] Replay    → Audit"
echo ""

read -rp "Role Selection: " ROLE_INPUT

case "$ROLE_INPUT" in
  1) ROLE="forge" ;;
  2) ROLE="sentinel" ;;
  3) ROLE="oracle" ;;
  4) ROLE="gate" ;;
  5) ROLE="replay" ;;
  *) ROLE="forge" ;;
esac

echo ""
echo "→ Claimed role: $ROLE"

curl -fsSL \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION_ID\",\"role\":\"$ROLE\"}" \
  "$DREDGE_API/v1/claim-role" \
  >/dev/null 2>&1 || true

echo ""
echo "═══════════════════════════════════════════════"
echo " ORION GATEWAY ACCESS"
echo "═══════════════════════════════════════════════"
echo ""

echo "Activate cloud orchestration?"
echo ""
echo "  [y] Connect to Orion"
echo "  [n] Local-only mode"
echo ""

read -rp "Selection: " ACTIVATE

if [[ "$ACTIVATE" == "y" || "$ACTIVATE" == "Y" ]]; then
  echo ""
  echo "→ Provisioning tenant..."

  CLAIM_URL="$DREDGE_API/claim/$SESSION_ID"

  curl -fsSL \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SESSION_ID\"}" \
    "$DREDGE_API/v1/provision" \
    >/dev/null 2>&1 || true

  echo ""
  echo "═══════════════════════════════════════════════"
  echo " CLAIM YOUR NODE"
  echo "═══════════════════════════════════════════════"
  echo ""
  echo "$CLAIM_URL"
  echo ""
  echo "Features unlocked:"
  echo "  • Agent mesh"
  echo "  • Workflow execution"
  echo "  • Telemetry"
  echo "  • API gateway"
  echo "  • SaaS billing"
  echo ""
else
  echo ""
  echo "→ Running in local scripture mode."
fi

echo ""
echo "Subscribe to release channel?"
echo ""

read -rp "Email (optional): " EMAIL

if [[ -n "${EMAIL:-}" ]]; then
  curl -fsSL \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SESSION_ID\",\"email\":\"$EMAIL\"}" \
    "$DREDGE_API/v1/subscribe" \
    >/dev/null 2>&1 || true

  echo ""
  echo "→ Added to release covenant."
fi

curl -fsSL \
  -X POST \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION_ID\",\"completed\":true}" \
  "$DREDGE_API/v1/funnel-complete" \
  >/dev/null 2>&1 || true

echo ""
echo "═══════════════════════════════════════════════"
echo " The gate remembers who walked through it."
echo "═══════════════════════════════════════════════"
echo ""
