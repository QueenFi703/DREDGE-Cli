#!/bin/bash
# Resolve DREDGE Swift dependencies using Docker dev image
# This script runs inside the Docker container with Swift compiler

set -e

cd /workspace || cd /app || cd $(pwd)

echo "📦 Resolving DREDGE Swift dependencies..."
echo ""

# Show current directory
echo "📍 Working directory: $(pwd)"
echo ""

# Resolve dependencies
echo "🔍 Running: swift package update"
swift package update

echo ""
echo "✅ Dependencies resolved!"
echo ""

# Show resolved dependencies
echo "📋 Package information:"
swift package describe

echo ""
echo "🏗️  Building DREDGE..."
swift build

echo ""
echo "✨ Done! DREDGE Swift toolchain is ready."
