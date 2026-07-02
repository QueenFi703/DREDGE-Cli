# PowerShell Diagnostic Script for app.py
# Run this to diagnose the errno 2 issue

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  DREDGE app.py Diagnostic Script" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check current directory
Write-Host "📁 Current Directory:" -ForegroundColor Yellow
$currentDir = Get-Location
Write-Host "   $currentDir" -ForegroundColor Gray
Write-Host ""

# Check if app.py exists
Write-Host "📋 Checking Files:" -ForegroundColor Yellow
if (Test-Path "app.py") {
    Write-Host "   ✅ app.py exists" -ForegroundColor Green
    $fileSize = (Get-Item "app.py").Length
    Write-Host "      Size: $fileSize bytes" -ForegroundColor Gray
} else {
    Write-Host "   ❌ app.py NOT FOUND" -ForegroundColor Red
    Write-Host "      Current directory: $(Get-Location)" -ForegroundColor Red
    exit 1
}

if (Test-Path "unified_auth_gateway.py") {
    Write-Host "   ✅ unified_auth_gateway.py exists" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  unified_auth_gateway.py NOT FOUND" -ForegroundColor Yellow
}

if (Test-Path "orion_gateway_authenticated.py") {
    Write-Host "   ✅ orion_gateway_authenticated.py exists" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  orion_gateway_authenticated.py NOT FOUND" -ForegroundColor Yellow
}

Write-Host ""

# Check Python
Write-Host "🐍 Python Check:" -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python not found" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test Python import
Write-Host "📦 Testing Imports:" -ForegroundColor Yellow

# Test if FastAPI is installed
Write-Host "   Checking FastAPI..." -ForegroundColor Gray
$fastapiTest = python -c "import fastapi; print('OK')" 2>&1
if ($fastapiTest -eq "OK") {
    Write-Host "   ✅ FastAPI installed" -ForegroundColor Green
} else {
    Write-Host "   ❌ FastAPI not installed" -ForegroundColor Red
    Write-Host "      Run: pip install fastapi uvicorn" -ForegroundColor Yellow
}

# Test if uvicorn is installed
Write-Host "   Checking Uvicorn..." -ForegroundColor Gray
$uvicornTest = python -c "import uvicorn; print('OK')" 2>&1
if ($uvicornTest -eq "OK") {
    Write-Host "   ✅ Uvicorn installed" -ForegroundColor Green
} else {
    Write-Host "   ❌ Uvicorn not installed" -ForegroundColor Red
    Write-Host "      Run: pip install uvicorn" -ForegroundColor Yellow
}

# Test unified_auth_gateway import
Write-Host "   Checking unified_auth_gateway import..." -ForegroundColor Gray
$importTest = python -c "from unified_auth_gateway import app; print('OK')" 2>&1
if ($importTest -eq "OK") {
    Write-Host "   ✅ unified_auth_gateway imports successfully" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  unified_auth_gateway import failed" -ForegroundColor Yellow
    Write-Host "      Error: $importTest" -ForegroundColor Gray
}

Write-Host ""

# Test app.py import
Write-Host "🚀 Testing app.py:" -ForegroundColor Yellow

Write-Host "   Running: python -c \"from app import app; print('OK')\"" -ForegroundColor Gray
$appTest = python -c "from app import app; print('OK')" 2>&1
if ($appTest -eq "OK") {
    Write-Host "   ✅ app.py imports successfully" -ForegroundColor Green
} else {
    Write-Host "   ❌ app.py import failed:" -ForegroundColor Red
    Write-Host "      $appTest" -ForegroundColor Red
}

Write-Host ""

# Try to start app with timeout
Write-Host "🔧 Attempting to Start app.py..." -ForegroundColor Yellow
Write-Host "   (This will timeout after 5 seconds)" -ForegroundColor Gray
Write-Host ""

# Create a simple test that checks if app starts
$testScript = @'
import sys
sys.path.insert(0, '.')
from app import app
print("App instance created successfully")
print(f"App title: {app.title}")
print(f"App routes: {len(app.routes)}")
'@

$appTestOutput = python -c $testScript 2>&1
Write-Host "   Output:" -ForegroundColor Gray
$appTestOutput | ForEach-Object { Write-Host "      $_" -ForegroundColor Gray }

Write-Host ""

# Recommendations
Write-Host "✅ Recommendations:" -ForegroundColor Cyan
Write-Host ""

if ($fastapiTest -ne "OK" -or $uvicornTest -ne "OK") {
    Write-Host "1. Install missing packages:" -ForegroundColor Yellow
    Write-Host "   pip install -r requirements.txt" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "2. Try starting the app with error output:" -ForegroundColor Yellow
Write-Host "   python app.py" -ForegroundColor Gray
Write-Host ""

Write-Host "3. If errno 2 persists, check:" -ForegroundColor Yellow
Write-Host "   - All required files exist" -ForegroundColor Gray
Write-Host "   - Current directory is correct" -ForegroundColor Gray
Write-Host "   - All imports resolve correctly" -ForegroundColor Gray
Write-Host ""

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Diagnostic Complete" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
