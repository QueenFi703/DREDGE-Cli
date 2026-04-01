#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Install and configure Grafana Alloy on Linux (binary install)
# Targets: amd64 and arm64
#
# Usage:
#   Export the required variables then run this script as root:
#
#   export GCLOUD_HOSTED_METRICS_URL="https://..."
#   export GCLOUD_HOSTED_METRICS_ID="<instance-id>"
#   export GCLOUD_HOSTED_LOGS_URL="https://..."
#   export GCLOUD_HOSTED_LOGS_ID="<instance-id>"
#   export GCLOUD_RW_API_KEY="<api-key>"
#   export GCLOUD_SCRAPE_INTERVAL="60s"   # optional, defaults to 60s
#   sudo -E ./scripts/install-alloy-linux.sh
#
# The script writes credentials to /etc/alloy/alloy.env (chmod 600,
# owned by root) and references that file from the systemd unit via
# EnvironmentFile=.  Credentials are never embedded in the Alloy
# config file itself.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Must run as root
# ──────────────────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
  echo "❌ This script must be run as root (use sudo -E)."
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# Validate required environment variables
# ──────────────────────────────────────────────────────────────────────
required_vars=(
  GCLOUD_HOSTED_METRICS_URL
  GCLOUD_HOSTED_METRICS_ID
  GCLOUD_HOSTED_LOGS_URL
  GCLOUD_HOSTED_LOGS_ID
  GCLOUD_RW_API_KEY
)

missing=()
for var in "${required_vars[@]}"; do
  if [ -z "${!var:-}" ]; then
    missing+=("$var")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "❌ Missing required environment variables:"
  for var in "${missing[@]}"; do
    echo "   $var"
  done
  echo ""
  echo "Set them and re-run, e.g.:"
  echo "   export GCLOUD_RW_API_KEY=\"glc_...\""
  echo "   sudo -E ./scripts/install-alloy-linux.sh"
  exit 1
fi

GCLOUD_SCRAPE_INTERVAL="${GCLOUD_SCRAPE_INTERVAL:-60s}"

# ──────────────────────────────────────────────────────────────────────
# Detect architecture
# ──────────────────────────────────────────────────────────────────────
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64)  ALLOY_ARCH="amd64" ;;
  aarch64) ALLOY_ARCH="arm64" ;;
  *)
    echo "❌ Unsupported architecture: ${ARCH}"
    exit 1
    ;;
esac

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DREDGE-Cli — Grafana Alloy setup (Linux/${ALLOY_ARCH})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ──────────────────────────────────────────────────────────────────────
# Install Alloy via the official Grafana Cloud onboarding script
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "📦 Installing Grafana Alloy..."

# Stop any running instance before reconfiguring
systemctl stop alloy 2>/dev/null || true

ARCH="${ALLOY_ARCH}" \
GCLOUD_HOSTED_METRICS_URL="${GCLOUD_HOSTED_METRICS_URL}" \
GCLOUD_HOSTED_METRICS_ID="${GCLOUD_HOSTED_METRICS_ID}" \
GCLOUD_SCRAPE_INTERVAL="${GCLOUD_SCRAPE_INTERVAL}" \
GCLOUD_HOSTED_LOGS_URL="${GCLOUD_HOSTED_LOGS_URL}" \
GCLOUD_HOSTED_LOGS_ID="${GCLOUD_HOSTED_LOGS_ID}" \
GCLOUD_RW_API_KEY="${GCLOUD_RW_API_KEY}" \
  /bin/sh -c "$(curl -fsSL https://storage.googleapis.com/cloud-onboarding/alloy/scripts/install-linux-binary.sh)"

# ──────────────────────────────────────────────────────────────────────
# Write Alloy configuration (no secrets — resolved at runtime via env)
# ──────────────────────────────────────────────────────────────────────
CONFIG_DIR="/etc/alloy"
CONFIG_PATH="${CONFIG_DIR}/config.alloy"

echo ""
echo "⚙️  Writing Alloy configuration to ${CONFIG_PATH}..."
mkdir -p "${CONFIG_DIR}"
rm -f "${CONFIG_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_CONFIG="${SCRIPT_DIR}/../monitoring/alloy/config-linux.alloy"

if [ ! -f "${REPO_CONFIG}" ]; then
  echo "❌ Config template not found at ${REPO_CONFIG}"
  exit 1
fi

# Copy the template; Alloy's env() resolves variables at process start.
cp "${REPO_CONFIG}" "${CONFIG_PATH}"
chmod 644 "${CONFIG_PATH}"

# ──────────────────────────────────────────────────────────────────────
# Write credentials to a root-only EnvironmentFile
# (never embedded in the config itself)
# ──────────────────────────────────────────────────────────────────────
ENV_FILE="${CONFIG_DIR}/alloy.env"

echo ""
echo "🔐 Writing credentials to ${ENV_FILE} (chmod 600)..."

cat > "${ENV_FILE}" <<ENV
GCLOUD_HOSTED_METRICS_URL=${GCLOUD_HOSTED_METRICS_URL}
GCLOUD_HOSTED_METRICS_ID=${GCLOUD_HOSTED_METRICS_ID}
GCLOUD_HOSTED_LOGS_URL=${GCLOUD_HOSTED_LOGS_URL}
GCLOUD_HOSTED_LOGS_ID=${GCLOUD_HOSTED_LOGS_ID}
GCLOUD_RW_API_KEY=${GCLOUD_RW_API_KEY}
GCLOUD_SCRAPE_INTERVAL=${GCLOUD_SCRAPE_INTERVAL}
ENV

chmod 600 "${ENV_FILE}"
chown root:root "${ENV_FILE}"

# ──────────────────────────────────────────────────────────────────────
# Configure systemd to load the EnvironmentFile
# ──────────────────────────────────────────────────────────────────────
DROPIN_DIR="/etc/systemd/system/alloy.service.d"
DROPIN_PATH="${DROPIN_DIR}/env-file.conf"

echo ""
echo "🔧 Configuring systemd drop-in at ${DROPIN_PATH}..."
mkdir -p "${DROPIN_DIR}"

cat > "${DROPIN_PATH}" <<DROPIN
[Service]
EnvironmentFile=${ENV_FILE}
DROPIN

chmod 644 "${DROPIN_PATH}"

# ──────────────────────────────────────────────────────────────────────
# Start Alloy
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "🚀 Starting Grafana Alloy..."
systemctl daemon-reload
systemctl enable alloy
systemctl restart alloy

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Grafana Alloy installed and started"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Config:       ${CONFIG_PATH}"
echo "  Credentials:  ${ENV_FILE}  (root-only, chmod 600)"
echo "  systemd unit: alloy.service"
echo ""
echo "  Check status:  systemctl status alloy"
echo "  View logs:     journalctl -u alloy -f"
echo "  Stop service:  systemctl stop alloy"
echo "  View UI:       http://localhost:12345"
echo ""
