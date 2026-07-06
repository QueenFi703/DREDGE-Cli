#!/usr/bin/env bash
set -euo pipefail

PUBLIC_REGISTRY="https://registry.npmjs.org/"

npm config set registry "$PUBLIC_REGISTRY"

echo "Using registry: $(npm config get registry)"

npm ping || true
npm cache clean --force
