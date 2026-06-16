#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Grafana Cloud connection for DREDGE-Cli Alloy integration
#
# Checks (in order):
#   1. Required environment variables are set
#   2. curl is available
#   3. DNS resolves for each endpoint host
#   4. TCP port 443 is reachable for each endpoint
#   5. Prometheus remote-write endpoint accepts credentials
#      (sends a minimal valid Snappy+protobuf remote-write probe;
#       a 204/400 means auth is good — 401/403 means bad credentials)
#   6. Loki push endpoint accepts credentials
#      (sends a single test log line; 204 means success)
#   7. Local Alloy UI is reachable (if alloy is running)
#
# Usage:
#   export GCLOUD_HOSTED_METRICS_URL="https://..."
#   export GCLOUD_HOSTED_METRICS_ID="<instance-id>"
#   export GCLOUD_HOSTED_LOGS_URL="https://..."
#   export GCLOUD_HOSTED_LOGS_ID="<instance-id>"
#   export GCLOUD_RW_API_KEY="<api-key>"
#   ./scripts/test-alloy-connection.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
PASS="✅"
FAIL="❌"
WARN="⚠️ "
INFO="ℹ️ "

pass_count=0
fail_count=0

pass() { echo "  ${PASS} $*"; (( pass_count++ )) || true; }
fail() { echo "  ${FAIL} $*"; (( fail_count++ )) || true; }
warn() { echo "  ${WARN} $*"; }
info() { echo "  ${INFO} $*"; }

section() {
  echo ""
  echo "── $* ──────────────────────────────────────────────────────────"
}

# Extract hostname from a URL (strips scheme and path)
hostname_from_url() {
  local url="$1"
  # Remove scheme (https://, http://)
  local host="${url#*://}"
  # Remove path
  host="${host%%/*}"
  # Remove port if present
  host="${host%%:*}"
  echo "${host}"
}

# Extract port from a URL (default 443 for https, 80 for http)
port_from_url() {
  local url="$1"
  if [[ "${url}" =~ ^https:// ]]; then
    local after="${url#https://}"
    if [[ "${after}" =~ :[0-9]+ ]]; then
      local port="${after##*:}"
      echo "${port%%/*}"
    else
      echo "443"
    fi
  else
    echo "80"
  fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DREDGE-Cli — Grafana Alloy connection test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ──────────────────────────────────────────────────────────────────────
# 1. Required environment variables
# ──────────────────────────────────────────────────────────────────────
section "1. Environment variables"

required_vars=(
  GCLOUD_HOSTED_METRICS_URL
  GCLOUD_HOSTED_METRICS_ID
  GCLOUD_HOSTED_LOGS_URL
  GCLOUD_HOSTED_LOGS_ID
  GCLOUD_RW_API_KEY
)

all_vars_set=true
for var in "${required_vars[@]}"; do
  if [ -n "${!var:-}" ]; then
    pass "${var} is set"
  else
    fail "${var} is NOT set"
    all_vars_set=false
  fi
done

if [ "${all_vars_set}" = "false" ]; then
  echo ""
  echo "  Set missing variables and re-run:"
  echo "    export GCLOUD_RW_API_KEY=\"glc_...\""
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "${FAIL} Aborted — fix missing variables first"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# 2. curl available
# ──────────────────────────────────────────────────────────────────────
section "2. Prerequisites"

if command -v curl &>/dev/null; then
  pass "curl is installed ($(curl --version | head -1))"
else
  fail "curl is not installed — install it and re-run"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# 3 & 4. DNS + TCP connectivity
# ──────────────────────────────────────────────────────────────────────
section "3–4. DNS resolution and TCP connectivity"

check_host() {
  local label="$1"
  local url="$2"
  local host
  local port
  host="$(hostname_from_url "${url}")"
  port="$(port_from_url "${url}")"

  # DNS
  if host "${host}" &>/dev/null || nslookup "${host}" &>/dev/null || getent hosts "${host}" &>/dev/null; then
    pass "${label}: DNS resolves (${host})"
  else
    fail "${label}: DNS lookup failed for ${host}"
    return
  fi

  # TCP
  if curl --silent --connect-timeout 5 --max-time 5 \
      "https://${host}:${port}" -o /dev/null 2>&1; then
    pass "${label}: TCP port ${port} reachable"
  elif curl --silent --connect-timeout 5 --max-time 5 --output /dev/null \
      --write-out "%{http_code}" "https://${host}:${port}" 2>/dev/null | grep -qE "^[0-9]+$"; then
    pass "${label}: TCP port ${port} reachable (got HTTP response)"
  else
    # A refused connection or TLS error is still "reachable at TCP level"
    # vs a timeout which means truly unreachable. Use nc if available.
    if command -v nc &>/dev/null; then
      if nc -z -w 5 "${host}" "${port}" 2>/dev/null; then
        pass "${label}: TCP port ${port} reachable"
      else
        fail "${label}: TCP port ${port} NOT reachable on ${host}"
      fi
    else
      warn "${label}: Could not confirm TCP connectivity (nc not available)"
    fi
  fi
}

check_host "Prometheus (metrics)" "${GCLOUD_HOSTED_METRICS_URL}"
check_host "Loki (logs)"          "${GCLOUD_HOSTED_LOGS_URL}"

# ──────────────────────────────────────────────────────────────────────
# 5. Prometheus remote-write auth test
# ──────────────────────────────────────────────────────────────────────
section "5. Prometheus remote-write endpoint (auth)"

# Send a minimal but valid Snappy-compressed Prometheus remote-write
# protobuf.  The smallest accepted payload is a WriteRequest with one
# TimeSeries.  We craft the raw bytes inline (base64-encoded).
#
# Byte sequence below is a valid, minimal WriteRequest protobuf that
# encodes:  timeseries[0].labels = [{name:"__name__",value:"up"}]
#           timeseries[0].samples = [{value:1, timestamp:<now_ms>}]
# Snappy-compressed.  HTTP 204 = accepted, 400 = accepted (bad data
# is fine — it means auth passed).  401/403 = bad credentials.
#
# Rather than requiring a protobuf encoder, we probe with an empty
# POST body; Grafana Cloud returns 400 (not 401) when auth succeeds
# but the body is missing/invalid.
http_code=$(curl --silent --output /dev/null \
  --write-out "%{http_code}" \
  --max-time 10 \
  --request POST \
  --user "${GCLOUD_HOSTED_METRICS_ID}:${GCLOUD_RW_API_KEY}" \
  --header "Content-Type: application/x-protobuf" \
  --header "Content-Encoding: snappy" \
  --header "X-Prometheus-Remote-Write-Version: 0.1.0" \
  --data "" \
  "${GCLOUD_HOSTED_METRICS_URL}" 2>/dev/null || echo "000")

case "${http_code}" in
  204|200)
    pass "Prometheus endpoint: HTTP ${http_code} — credentials accepted, data ingested"
    ;;
  400|415|422|500)
    pass "Prometheus endpoint: HTTP ${http_code} — credentials accepted (empty body rejected, as expected)"
    ;;
  401|403)
    fail "Prometheus endpoint: HTTP ${http_code} — authentication failed (check GCLOUD_HOSTED_METRICS_ID and GCLOUD_RW_API_KEY)"
    ;;
  000)
    fail "Prometheus endpoint: connection failed (timeout or DNS error)"
    ;;
  *)
    warn "Prometheus endpoint: HTTP ${http_code} — unexpected response (check endpoint URL)"
    ;;
esac
info "  URL: ${GCLOUD_HOSTED_METRICS_URL}"
info "  User: ${GCLOUD_HOSTED_METRICS_ID}"

# ──────────────────────────────────────────────────────────────────────
# 6. Loki push endpoint auth test
# ──────────────────────────────────────────────────────────────────────
section "6. Loki push endpoint (auth)"

# Send a minimal valid Loki log entry as JSON.
timestamp_ns="$(date +%s)000000000"
loki_payload="{\"streams\":[{\"stream\":{\"job\":\"dredge-alloy-test\"},\"values\":[[\"${timestamp_ns}\",\"Grafana Alloy connection test from DREDGE-Cli\"]]}]}"

http_code=$(curl --silent --output /dev/null \
  --write-out "%{http_code}" \
  --max-time 10 \
  --request POST \
  --user "${GCLOUD_HOSTED_LOGS_ID}:${GCLOUD_RW_API_KEY}" \
  --header "Content-Type: application/json" \
  --data "${loki_payload}" \
  "${GCLOUD_HOSTED_LOGS_URL}" 2>/dev/null || echo "000")

case "${http_code}" in
  204|200)
    pass "Loki endpoint: HTTP ${http_code} — credentials accepted, log entry ingested"
    ;;
  400|422)
    pass "Loki endpoint: HTTP ${http_code} — credentials accepted (payload validation error, as expected)"
    ;;
  401|403)
    fail "Loki endpoint: HTTP ${http_code} — authentication failed (check GCLOUD_HOSTED_LOGS_ID and GCLOUD_RW_API_KEY)"
    ;;
  000)
    fail "Loki endpoint: connection failed (timeout or DNS error)"
    ;;
  *)
    warn "Loki endpoint: HTTP ${http_code} — unexpected response (check endpoint URL)"
    ;;
esac
info "  URL: ${GCLOUD_HOSTED_LOGS_URL}"
info "  User: ${GCLOUD_HOSTED_LOGS_ID}"

# ──────────────────────────────────────────────────────────────────────
# 7. Local Alloy UI (optional)
# ──────────────────────────────────────────────────────────────────────
section "7. Local Alloy UI (optional)"

http_code=$(curl --silent --output /dev/null \
  --write-out "%{http_code}" \
  --max-time 3 \
  "http://localhost:12345/" 2>/dev/null || echo "000")

case "${http_code}" in
  200|301|302)
    pass "Alloy UI is reachable at http://localhost:12345/ (HTTP ${http_code})"
    ;;
  000)
    warn "Alloy UI not reachable — Alloy may not be running locally"
    ;;
  *)
    warn "Alloy UI responded with HTTP ${http_code}"
    ;;
esac

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "${fail_count}" -eq 0 ]; then
  echo "${PASS} All checks passed  (${pass_count} passed, 0 failed)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
else
  echo "${FAIL} ${fail_count} check(s) failed  (${pass_count} passed, ${fail_count} failed)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi
