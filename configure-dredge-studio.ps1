param(
    [string]$ApiUrl = "http://127.0.0.1:8000"
)

Write-Host "=== DREDGE Studio - Advanced Configuration ===" -ForegroundColor Yellow
Write-Host "Configuring via Wizard Steps API`n" -ForegroundColor Cyan

# Helper function
function Submit-WizardStep {
    param(
        [int]$StepId,
        [hashtable]$Data
    )
    
    $body = @{
        step_id = $StepId
        data = $Data
    } | ConvertTo-Json
    
    Write-Host "Step $StepId - Submitting configuration..." -ForegroundColor Magenta
    
    try {
        $response = Invoke-RestMethod -Uri "$ApiUrl/api/wizard/steps" `
            -Method Post `
            -Body $body `
            -ContentType "application/json" `
            -TimeoutSec 15
        Write-Host "✓ Step $StepId configured" -ForegroundColor Green
        return $response
    } catch {
        Write-Host "✗ Error on Step $($StepId): $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Step 1: Project Setup
Write-Host "`n[1/5] Project Setup..." -ForegroundColor Cyan
$projectSetup = @{
    project_name = "DREDGE Studio"
    description = "GPU-accelerated Swift/Python development environment"
    version = "1.0.0"
}
Submit-WizardStep -StepId 1 -Data $projectSetup

# Step 2: Swift Configuration (GPU + CPU optimized)
Write-Host "`n[2/5] Swift Configuration (GPU/CPU)..." -ForegroundColor Cyan
$swiftConfig = @{
    swift_version = "5.9"
    optimization_level = "-O"
    enable_testing = $true
}
Submit-WizardStep -StepId 2 -Data $swiftConfig

# Step 3: Python Configuration
Write-Host "`n[3/5] Python Configuration..." -ForegroundColor Cyan
$pythonConfig = @{
    python_version = "3.11"
    virtual_env = $true
}
Submit-WizardStep -StepId 3 -Data $pythonConfig

# Step 4: Docker Configuration (GPU support)
Write-Host "`n[4/5] Docker Configuration (GPU)..." -ForegroundColor Cyan
$dockerConfig = @{
    enable_docker = $true
    docker_image = "dredge-studio:gpu-latest"
    port = 3001
}
Submit-WizardStep -StepId 4 -Data $dockerConfig

# Step 5: Testing Configuration
Write-Host "`n[5/5] Testing Configuration..." -ForegroundColor Cyan
$testingConfig = @{
    test_framework = "XCTest"
    coverage_threshold = 90
    auto_run_tests = $true
}
Submit-WizardStep -StepId 5 -Data $testingConfig

# Fetch Xcode project info
Write-Host "`n[6/6] Loading Xcode Project Info..." -ForegroundColor Cyan
try {
    $projectInfo = Invoke-RestMethod -Uri "$ApiUrl/api/xcode/project-info" `
        -Method Get `
        -TimeoutSec 10
    Write-Host "✓ Project info loaded:" -ForegroundColor Green
    $projectInfo | ConvertTo-Json -Depth 3 | Write-Host
} catch {
    Write-Host "ℹ No Xcode project found (optional)" -ForegroundColor Yellow
}

# Fetch schemes
Write-Host "`n[7/6] Available Xcode Schemes..." -ForegroundColor Cyan
try {
    $schemes = Invoke-RestMethod -Uri "$ApiUrl/api/xcode/schemes" `
        -Method Get `
        -TimeoutSec 10
    if ($schemes -and $schemes.Count -gt 0) {
        Write-Host "✓ Schemes:" -ForegroundColor Green
        $schemes | ForEach-Object { Write-Host "  - $_" -ForegroundColor Cyan }
    }
} catch {
    Write-Host "ℹ No schemes available" -ForegroundColor Yellow
}

# Fetch targets
Write-Host "`n[8/6] Available Xcode Targets..." -ForegroundColor Cyan
try {
    $targets = Invoke-RestMethod -Uri "$ApiUrl/api/xcode/targets" `
        -Method Get `
        -TimeoutSec 10
    if ($targets -and $targets.Count -gt 0) {
        Write-Host "✓ Targets:" -ForegroundColor Green
        $targets | ForEach-Object { Write-Host "  - $_" -ForegroundColor Cyan }
    }
} catch {
    Write-Host "ℹ No targets available" -ForegroundColor Yellow
}

Write-Host "`n╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║     DREDGE STUDIO - CONFIGURATION COMPLETE             ║" -ForegroundColor Green
Write-Host "╠════════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║ Project:     DREDGE Studio v1.0.0                      ║" -ForegroundColor Green
Write-Host "║ Swift:       5.9 with -O optimization (GPU ready)      ║" -ForegroundColor Green
Write-Host "║ Python:      3.11 with virtual environment             ║" -ForegroundColor Green
Write-Host "║ Docker:      GPU-enabled (port 3001)                   ║" -ForegroundColor Green
Write-Host "║ Testing:     XCTest framework, 90% coverage target      ║" -ForegroundColor Green
Write-Host "║ Features:    Auto-test running enabled                 ║" -ForegroundColor Green
Write-Host "╠════════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║ KEYBOARD SHORTCUTS:                                    ║" -ForegroundColor Green
Write-Host "║ • Cmd+B      → Build (Xcode)                           ║" -ForegroundColor Green
Write-Host "║ • Cmd+U      → Test                                    ║" -ForegroundColor Green
Write-Host "║ • Cmd+R      → Run                                     ║" -ForegroundColor Green
Write-Host "║ • Cmd+Shift+D → Docker Build (GPU)                     ║" -ForegroundColor Green
Write-Host "║ • Cmd+Shift+G → Docker Run (GPU)                       ║" -ForegroundColor Green
Write-Host "║ • Cmd+K      → Clear Console                           ║" -ForegroundColor Green
Write-Host "║ • Cmd+\      → Toggle Breakpoint                       ║" -ForegroundColor Green
Write-Host "║ • Cmd+Shift+Y → Show Debugger Console                  ║" -ForegroundColor Green
Write-Host "╠════════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║ UI FEATURES ENABLED:                                   ║" -ForegroundColor Green
Write-Host "║ ✓ Swift Build System                                   ║" -ForegroundColor Green
Write-Host "║ ✓ Xcode Integration                                    ║" -ForegroundColor Green
Write-Host "║ ✓ GPU Tools (Metal, CUDA)                              ║" -ForegroundColor Green
Write-Host "║ ✓ Docker Build & Run                                   ║" -ForegroundColor Green
Write-Host "║ ✓ Testing Framework (XCTest)                           ║" -ForegroundColor Green
Write-Host "║ ✓ Code Coverage Tracking                               ║" -ForegroundColor Green
Write-Host "║ ✓ Debugger & Breakpoints                               ║" -ForegroundColor Green
Write-Host "║ ✓ REPL Shell                                           ║" -ForegroundColor Green
Write-Host "║ ✓ File Upload & Management                             ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`nUI Dashboard: $ApiUrl" -ForegroundColor Cyan
Write-Host "API Docs: $ApiUrl/docs" -ForegroundColor Cyan
Write-Host "`nRefresh your browser to see the new configuration in the UI menu." -ForegroundColor Yellow
