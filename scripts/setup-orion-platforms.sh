#!/usr/bin/env bash
set -euo pipefail

# Configure GitHub Actions secrets for Orion and link Vercel/Railway projects.
# Required tools: gh, vercel, railway

required_cmds=(gh vercel railway)
for cmd in "${required_cmds[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: '$cmd' CLI is required but not installed." >&2
    exit 1
  fi
done

if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
  echo "Error: set GITHUB_REPOSITORY (owner/repo), e.g. QueenFi703/DREDGE-Cli" >&2
  exit 1
fi

# Orion secrets map: ENV var name => GitHub secret name
# Add/remove based on your deployment needs.
ORION_SECRET_KEYS=(
  ORION_API_KEY
  ORION_BASE_URL
  DATABASE_URL
  REDIS_URL
  JWT_SECRET
  STRIPE_SECRET_KEY
)

missing=()
for secret in "${ORION_SECRET_KEYS[@]}"; do
  if [[ -z "${!secret:-}" ]]; then
    missing+=("$secret")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing required environment variables:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "Export them, then rerun this script." >&2
  exit 1
fi

echo "Setting GitHub secrets in ${GITHUB_REPOSITORY}..."
for secret in "${ORION_SECRET_KEYS[@]}"; do
  printf '%s' "${!secret}" | gh secret set "$secret" --repo "$GITHUB_REPOSITORY"
  echo "  ✓ $secret"
done

if [[ -n "${VERCEL_PROJECT_NAME:-}" ]]; then
  echo "Linking Vercel project '${VERCEL_PROJECT_NAME}'..."
  vercel link --yes --project "$VERCEL_PROJECT_NAME"
else
  echo "VERCEL_PROJECT_NAME is not set; running interactive Vercel link."
  vercel link
fi

if [[ -n "${RAILWAY_PROJECT_ID:-}" ]]; then
  echo "Linking Railway project '${RAILWAY_PROJECT_ID}'..."
  railway link "$RAILWAY_PROJECT_ID"
else
  echo "RAILWAY_PROJECT_ID is not set; running interactive Railway link."
  railway link
fi

echo "Done. Orion secrets configured and Vercel/Railway linked."
