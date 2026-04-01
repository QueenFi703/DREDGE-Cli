#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Install and configure Grafana Alloy on macOS (Homebrew)
# Targets: Apple Silicon (arm64) and Intel (x86_64)
#
# Usage:
#   Export the required variables then run this script:
#
#   export GCLOUD_HOSTED_METRICS_URL="https://..."
#   export GCLOUD_HOSTED_METRICS_ID="<instance-id>"
#   export GCLOUD_HOSTED_LOGS_URL="https://..."
#   export GCLOUD_HOSTED_LOGS_ID="<instance-id>"
#   export GCLOUD_RW_API_KEY="<api-key>"
#   export GCLOUD_SCRAPE_INTERVAL="60s"   # optional, defaults to 60s
#   ./scripts/install-alloy-macos.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

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
  exit 1
fi

GCLOUD_SCRAPE_INTERVAL="${GCLOUD_SCRAPE_INTERVAL:-60s}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DREDGE-Cli — Grafana Alloy setup (macOS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ──────────────────────────────────────────────────────────────────────
# Install Alloy via Homebrew
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "📦 Installing Grafana Alloy..."
if ! command -v brew &>/dev/null; then
  echo "❌ Homebrew is not installed. Install it first: https://brew.sh"
  exit 1
fi

brew install grafana/grafana/alloy

# Stop any running instance before reconfiguring
brew services stop alloy 2>/dev/null || true

# ──────────────────────────────────────────────────────────────────────
# Write Alloy configuration
# ──────────────────────────────────────────────────────────────────────
CONFIG_DIR="$(brew --prefix)/etc/alloy"
CONFIG_PATH="${CONFIG_DIR}/config.alloy"

echo ""
echo "⚙️  Writing Alloy configuration to ${CONFIG_PATH}..."
mkdir -p "${CONFIG_DIR}"
rm -f "${CONFIG_PATH}"

# Resolve the config template shipped with this repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_CONFIG="${SCRIPT_DIR}/../monitoring/alloy/config.alloy"

if [ ! -f "${REPO_CONFIG}" ]; then
  echo "❌ Config template not found at ${REPO_CONFIG}"
  exit 1
fi

# Substitute environment variables into the deployed config
# Alloy's env() function reads from the process environment at runtime,
# so we just copy the template; the variables must be present when alloy
# starts (see launchd plist below).
cp "${REPO_CONFIG}" "${CONFIG_PATH}"

# ──────────────────────────────────────────────────────────────────────
# Persist environment variables for the Homebrew service (launchd)
# ──────────────────────────────────────────────────────────────────────
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/homebrew.mxcl.alloy.plist"

echo ""
echo "🔧 Configuring launchd plist with environment variables..."
mkdir -p "${PLIST_DIR}"

# Build the plist; credentials are written only to a user-owned file.
cat > "${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>homebrew.mxcl.alloy</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(brew --prefix)/bin/alloy</string>
    <string>run</string>
    <string>${CONFIG_PATH}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>GCLOUD_HOSTED_METRICS_URL</key>
    <string>${GCLOUD_HOSTED_METRICS_URL}</string>
    <key>GCLOUD_HOSTED_METRICS_ID</key>
    <string>${GCLOUD_HOSTED_METRICS_ID}</string>
    <key>GCLOUD_HOSTED_LOGS_URL</key>
    <string>${GCLOUD_HOSTED_LOGS_URL}</string>
    <key>GCLOUD_HOSTED_LOGS_ID</key>
    <string>${GCLOUD_HOSTED_LOGS_ID}</string>
    <key>GCLOUD_RW_API_KEY</key>
    <string>${GCLOUD_RW_API_KEY}</string>
    <key>GCLOUD_SCRAPE_INTERVAL</key>
    <string>${GCLOUD_SCRAPE_INTERVAL}</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${HOME}/Library/Logs/alloy/alloy.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/Library/Logs/alloy/alloy.err.log</string>
</dict>
</plist>
PLIST

chmod 600 "${PLIST_PATH}"

# ──────────────────────────────────────────────────────────────────────
# Start Alloy
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "🚀 Starting Grafana Alloy..."
mkdir -p "${HOME}/Library/Logs/alloy"
launchctl load "${PLIST_PATH}" 2>/dev/null || true
brew services start alloy

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Grafana Alloy installed and started"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Config:  ${CONFIG_PATH}"
echo "  Logs:    ${HOME}/Library/Logs/alloy/alloy.log"
echo "  Errors:  ${HOME}/Library/Logs/alloy/alloy.err.log"
echo ""
echo "  Check status:  brew services info alloy"
echo "  Stop service:  brew services stop alloy"
echo "  View UI:       http://localhost:12345"
echo ""
