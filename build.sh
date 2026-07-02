#!/bin/bash
# Vercel Build Script - Explicitly use pip, not uv

echo "🔨 Building DREDGE Auth Gateway"
echo "================================"

# Install dependencies using pip (not uv)
echo "📦 Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "✅ Build complete"
