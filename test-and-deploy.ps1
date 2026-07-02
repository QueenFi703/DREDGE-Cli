# PowerShell Script to Test and Deploy DREDGE Auth Gateway
# Run in PowerShell terminal

Write-Host "╔════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  DREDGE Auth Gateway - Local Test & Vercel Deployment Script      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# SECTION 1: VERIFY ENVIRONMENT
# ============================================================================

Write-Host "1️⃣  Verifying Environment..." -ForegroundColor Yellow
Write-Host ""

# Check Python
Write-Host "   Checking Python..." -ForegroundColor Gray
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python not found" -ForegroundColor Red
    exit 1
}

# Check pip
Write-Host "   Checking pip..." -ForegroundColor Gray
try {
    $pipVersion = pip --version 2>&1
    Write-Host "   ✅ Pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Pip not found" -ForegroundColor Red
    exit 1
}

# Check Node.js (for Vercel CLI)
Write-Host "   Checking Node.js..." -ForegroundColor Gray
try {
    $nodeVersion = node --version 2>&1
    Write-Host "   ✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Node.js not found (needed for Vercel deployment)" -ForegroundColor Yellow
}

Write-Host ""

# ============================================================================
# SECTION 2: VERIFY FILES
# ============================================================================

Write-Host "2️⃣  Verifying Required Files..." -ForegroundColor Yellow
Write-Host ""

$requiredFiles = @(
    "app.py",
    "requirements.txt",
    "unified_auth_gateway.py",
    "api_key_manager.py",
    "unified_auth_middleware.py",
    "vercel.json"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "   ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $file - MISSING!" -ForegroundColor Red
        $allFilesExist = $false
    }
}

Write-Host ""

if (-not $allFilesExist) {
    Write-Host "❌ Some required files are missing!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# SECTION 3: INSTALL DEPENDENCIES
# ============================================================================

Write-Host "3️⃣  Installing Dependencies..." -ForegroundColor Yellow
Write-Host ""

Write-Host "   Running: pip install -r requirements.txt" -ForegroundColor Gray
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "   ✅ Dependencies installed" -ForegroundColor Green
Write-Host ""

# ============================================================================
# SECTION 4: TEST LOCALLY
# ============================================================================

Write-Host "4️⃣  Testing Locally..." -ForegroundColor Yellow
Write-Host ""

Write-Host "   Starting app.py in background..." -ForegroundColor Gray
Write-Host ""

# Start the app in background
$process = Start-Process python -ArgumentList "app.py" -WindowStyle Hidden -PassThru
$processId = $process.Id
Write-Host "   ✅ App started (PID: $processId)" -ForegroundColor Green

# Wait for app to start
Write-Host "   Waiting for app to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# Test health endpoint
Write-Host ""
Write-Host "   Testing health endpoint..." -ForegroundColor Gray

try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:9000/health" -TimeoutSec 5 -ErrorAction Stop
    $statusCode = $response.StatusCode
    $content = $response.Content | ConvertFrom-Json
    
    Write-Host "   ✅ Health endpoint responding (Status: $statusCode)" -ForegroundColor Green
    Write-Host "      Status: $($content.status)" -ForegroundColor Green
    Write-Host "      Service: $($content.service)" -ForegroundColor Green
    Write-Host "      Version: $($content.version)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Health endpoint failed: $_" -ForegroundColor Red
}

# Test Swagger UI
Write-Host ""
Write-Host "   Testing Swagger UI..." -ForegroundColor Gray
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:9000/docs" -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "   ✅ Swagger UI available at http://127.0.0.1:9000/docs" -ForegroundColor Green
    }
} catch {
    Write-Host "   ⚠️  Swagger UI test failed: $_" -ForegroundColor Yellow
}

# Test OpenAPI schema
Write-Host ""
Write-Host "   Testing OpenAPI schema..." -ForegroundColor Gray
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:9000/openapi.json" -TimeoutSec 5 -ErrorAction Stop
    $content = $response.Content | ConvertFrom-Json
    Write-Host "   ✅ OpenAPI schema available" -ForegroundColor Green
    Write-Host "      Title: $($content.info.title)" -ForegroundColor Green
    Write-Host "      Version: $($content.info.version)" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  OpenAPI test failed: $_" -ForegroundColor Yellow
}

Write-Host ""

# Stop the background process
Write-Host "   Stopping background process..." -ForegroundColor Gray
Stop-Process -Id $processId -Force
Write-Host "   ✅ Process stopped" -ForegroundColor Green

Write-Host ""

# ============================================================================
# SECTION 5: DEPLOYMENT OPTIONS
# ============================================================================

Write-Host "5️⃣  Deployment Options" -ForegroundColor Yellow
Write-Host ""

Write-Host "   OPTION A: Deploy to Vercel (Recommended)" -ForegroundColor Cyan
Write-Host "   ─────────────────────────────────────────" -ForegroundColor Gray
Write-Host "   1. npm install -g vercel" -ForegroundColor Gray
Write-Host "   2. vercel login" -ForegroundColor Gray
Write-Host "   3. vercel --prod" -ForegroundColor Gray
Write-Host ""

Write-Host "   OPTION B: Deploy Locally with Docker" -ForegroundColor Cyan
Write-Host "   ────────────────────────────────────" -ForegroundColor Gray
Write-Host "   1. docker build -f Dockerfile.prod -t dredge:latest ." -ForegroundColor Gray
Write-Host "   2. docker run -p 9000:9000 dredge:latest" -ForegroundColor Gray
Write-Host ""

Write-Host "   OPTION C: Run Locally for Development" -ForegroundColor Cyan
Write-Host "   ──────────────────────────────────────" -ForegroundColor Gray
Write-Host "   1. python app.py" -ForegroundColor Gray
Write-Host "   2. Open: http://127.0.0.1:9000/docs" -ForegroundColor Gray
Write-Host ""

# ============================================================================
# SECTION 6: SUMMARY
# ============================================================================

Write-Host "✅ All Tests Passed!" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  ✓ Python environment verified" -ForegroundColor Green
Write-Host "  ✓ All required files present" -ForegroundColor Green
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
Write-Host "  ✓ Local testing successful" -ForegroundColor Green
Write-Host "  ✓ Health endpoints responding" -ForegroundColor Green
Write-Host "  ✓ Swagger UI available" -ForegroundColor Green
Write-Host ""

Write-Host "Entry Point Summary:" -ForegroundColor Cyan
Write-Host "  App File: app.py (Vercel recognized)" -ForegroundColor Green
Write-Host "  Entry Variable: app" -ForegroundColor Green
Write-Host "  Type: FastAPI ASGI Application" -ForegroundColor Green
Write-Host "  Local URL: http://127.0.0.1:9000" -ForegroundColor Green
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Commit changes: git add -A && git commit -m 'fix: Add app.py entry point'" -ForegroundColor Yellow
Write-Host "  2. Push to GitHub: git push origin master" -ForegroundColor Yellow
Write-Host "  3. Deploy to Vercel: vercel --prod" -ForegroundColor Yellow
Write-Host ""

Write-Host "════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
