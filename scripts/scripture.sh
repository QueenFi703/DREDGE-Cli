#!/usr/bin/env bash
# =========================================================
# DREDGE :: SCRIPTURE FUNNEL
# "The net gathers before the harvest."
# Multi-layer funnel: capture + onboarding + retention + billing
# =========================================================

set -euo pipefail

DREDGE_VERSION="${DREDGE_VERSION:-0.1.0}"
DREDGE_API="${DREDGE_API:-https://api.oriongateway.io}"
DREDGE_REPO="${DREDGE_REPO:-QueenFi703/DREDGE-Cli}"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
SESSION_ID="$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid || echo "unknown")"
TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo ""
echo "═══════════════════════════════════════════════"
echo "        D R E D G E   A W A K E N S"
echo "═══════════════════════════════════════════════"
echo ""

device_payload() {
cat <<JSON
{
  "session_id":"$SESSION_ID",
  "repo":"$DREDGE_REPO",
  "version":"$DREDGE_VERSION",
  "os":"$OS",
  "arch":"$ARCH",
  "timestamp":"$TIMESTAMP"
}
JSON
}

emit_json() {
  local endpoint="$1"
  local payload="$2"

  curl -fsSL \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$DREDGE_API/$endpoint" \
    >/dev/null 2>&1 || true
}

# ---------------------------------------------------------
# [CREATE] CAPTURE LAYER
# ---------------------------------------------------------

echo "→ Capture layer: install telemetry"
emit_json "v1/install" "$(device_payload)"

# ---------------------------------------------------------
# [CREATE] ONBOARDING LAYER
# ---------------------------------------------------------

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
echo "→ Onboarding layer: claimed role '$ROLE'"
emit_json "v1/claim-role" "{\"session_id\":\"$SESSION_ID\",\"role\":\"$ROLE\",\"timestamp\":\"$TIMESTAMP\"}"

echo ""
echo "Activate cloud orchestration?"
echo ""
echo "  [y] Connect to Orion"
echo "  [n] Local-only mode"
echo ""

read -rp "Selection: " ACTIVATE

if [[ "$ACTIVATE" == "y" || "$ACTIVATE" == "Y" ]]; then
  CLAIM_URL="$DREDGE_API/claim/$SESSION_ID"

  echo ""
  echo "→ Onboarding layer: provisioning tenant..."
  emit_json "v1/provision" "{\"session_id\":\"$SESSION_ID\",\"role\":\"$ROLE\",\"timestamp\":\"$TIMESTAMP\"}"

  echo ""
  echo "Claim your node:"
  echo "$CLAIM_URL"
  echo ""
else
  echo ""
  echo "→ Onboarding layer: local-only mode"
fi

# ---------------------------------------------------------
# [CREATE] RETENTION LAYER
# ---------------------------------------------------------

echo "Subscribe to release channel?"
read -rp "Email (optional): " EMAIL

if [[ -n "${EMAIL:-}" ]]; then
  echo "→ Retention layer: subscribing $EMAIL"
  emit_json "v1/subscribe" "{\"session_id\":\"$SESSION_ID\",\"email\":\"$EMAIL\",\"timestamp\":\"$TIMESTAMP\"}"
fi

# ---------------------------------------------------------
# [CREATE] BILLING LAYER
# ---------------------------------------------------------

echo ""
echo "Enable billing setup now?"
echo "  [y] Start SaaS billing activation"
echo "  [n] Skip for now"
read -rp "Selection: " BILLING

if [[ "$BILLING" == "y" || "$BILLING" == "Y" ]]; then
  echo "→ Billing layer: creating billing intent"
  emit_json "v1/billing/intent" "{\"session_id\":\"$SESSION_ID\",\"repo\":\"$DREDGE_REPO\",\"role\":\"$ROLE\",\"timestamp\":\"$TIMESTAMP\"}"
fi

emit_json "v1/funnel-complete" "{\"session_id\":\"$SESSION_ID\",\"completed\":true,\"timestamp\":\"$TIMESTAMP\"}"

echo ""
echo "═══════════════════════════════════════════════"
echo " The gate remembers who walked through it."
echo "═══════════════════════════════════════════════"
echo ""
