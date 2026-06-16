# DREDGE Studio - Quick Reference Guide

## Overview
DREDGE Studio is an interactive development environment running on `http://127.0.0.1:8000` with GPU support, Swift/Python compilation, testing, and debugging capabilities.

---

## 📋 Menu Features Guide

### 1. REPL Console
**Location:** Sidebar → Development → REPL Console (default)

**What it does:**
- Execute Swift or Python commands interactively
- Test code snippets without building
- Inspect output and debug expressions

**How to use:**
1. Type a Swift or Python command in the input box at the bottom
2. Press `Enter` or click the **Run** button
3. Output appears above

**Example Commands:**
```swift
print("Hello from DREDGE Studio")
let x = 42
print(x * 2)
```

```python
print("Hello from Python")
x = 42
print(x * 2)
```

**Shortcuts:**
- `Enter` — Execute command
- `Cmd+K` — Clear console

---

### 2. Debugger
**Location:** Sidebar → Development → Debugger

**What it does:**
- Set and manage breakpoints in your code
- Inspect variable values during execution
- Step through code line-by-line

**How to use:**
1. Click **Breakpoints** panel (left side)
2. Add breakpoints by specifying file and line number
3. Click **Variables** panel (right side) to inspect values during debugging
4. Enable/disable individual breakpoints

**Keyboard Shortcuts:**
- `Cmd+\` — Toggle breakpoint
- `Cmd+Shift+Y` — Show/hide debugger

**Example:**
```
File: main.swift
Line: 15
Condition: (optional) x > 10
```

---

### 3. Tests
**Location:** Sidebar → Development → Tests

**What it does:**
- Discover all tests in your project
- Run tests individually or all at once
- View test results with pass/fail status

**How to use:**
1. Click **Run All Tests** button
2. Tests execute and results appear below
3. Green ✓ = passed, Red ✗ = failed, Yellow ⊘ = skipped
4. Click on individual test to see output

**Keyboard Shortcuts:**
- `Cmd+U` — Run tests
- `Cmd+Shift+U` — Run Xcode tests

**Test Status Colors:**
- 🟢 **Passed** — Test succeeded
- 🔴 **Failed** — Test failed (view output for details)
- 🟡 **Skipped** — Test was skipped

---

### 4. Setup Wizard
**Location:** Sidebar → Configuration → Setup Wizard

**What it does:**
- Step-by-step project configuration
- 5 stages: Project → Swift → Python → Docker → Testing

**How to use:**
1. Fill in fields for current step
2. Click **Next** to proceed
3. Click **Previous** to go back
4. Complete all 5 steps

**The 5 Steps:**

#### Step 1: Project Setup
- **project_name** — Your project name (required)
- **description** — What your project does
- **version** — Version number (e.g., 0.1.0)

#### Step 2: Swift Configuration
- **swift_version** — Choose 5.8, 5.9, or 5.10
- **optimization_level** — Choose `-O` (fast), `-Osize` (small), or `-Onone` (debug)
- **enable_testing** — Checkmark to enable XCTest

#### Step 3: Python Configuration
- **python_version** — Choose 3.8, 3.9, 3.10, 3.11, or 3.12
- **virtual_env** — Checkmark to use virtual environment

#### Step 4: Docker Configuration
- **enable_docker** — Checkmark to enable Docker support
- **docker_image** — Docker image name (e.g., `dredge-studio:gpu-latest`)
- **port** — Container port (default 3001)

#### Step 5: Testing Configuration
- **test_framework** — Choose XCTest or pytest
- **coverage_threshold** — Code coverage target (e.g., 90)
- **auto_run_tests** — Checkmark to auto-run on save

---

### 5. Settings
**Location:** Sidebar → Configuration → Settings

**What it does:**
- Edit server configuration (host, port, debug mode)
- Configure MCP (Model Context Protocol) settings
- Adjust logging level and format

**Configuration Options:**

**Server:**
- `host` — Listen address (0.0.0.0 for all interfaces)
- `port` — Server port (default 3000)
- `debug` — Enable debug mode (true/false)
- `threads` — Number of worker threads

**MCP (GPU/Model Support):**
- `host` — MCP host address
- `port` — MCP port (default 3002)
- `device` — GPU device selection (auto/cuda/metal/cpu)
- `threads` — MCP worker threads

**Logging:**
- `level` — Log level (DEBUG, INFO, WARNING, ERROR)
- `format` — Log format (json or text)

---

### 6. Swift Dependencies
**Location:** Sidebar → Build → Swift Dependencies

**What it does:**
- Resolve Swift package dependencies
- Build local DREDGE package
- View package dependency graph

**Actions Available:**

#### Resolve Packages
- Resolves all dependencies from `Package.swift`
- Updates package lock file
- Button: **Resolve**

#### Build Local DREDGE
- Builds the local DREDGE package
- Located in `swift/DREDGE` directory
- Button: **Build Dependency**

#### Build Swift CLI
- Compiles the Swift command-line tool
- Located in `swift/` directory
- Button: **Build Swift**

#### Package Graph
- Shows dependency tree and information
- Button: **Describe**

**Status Indicators:**
- 🟢 **Green dot** — Operation succeeded
- 🔴 **Red dot** — Operation failed
- ⚪ **Gray dot** — Pending/Ready

---

### 7. Build Swift
**Location:** Sidebar → Build → Build Swift
**Keyboard Shortcut:** `Cmd+B`

**What it does:**
- Compiles Swift code with optimizations enabled
- Outputs compiled binary to `build/` directory
- Shows build status and errors

**Build Output Includes:**
- Compilation time
- File count
- Any warnings or errors
- Output binary path

**Configuration (from wizard):**
- Optimization level: `-O` (fast)
- Swift version: 5.9
- Testing enabled

---

### 8. Build Python
**Location:** Sidebar → Build → Build Python

**What it does:**
- Compiles Python code
- Creates virtual environment (if enabled)
- Packages dependencies

**Configuration (from wizard):**
- Python version: 3.11
- Virtual environment: enabled

---

## 🎮 Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| `Cmd+B` | Build (Xcode/Swift) |
| `Cmd+U` | Run Tests |
| `Cmd+R` | Run Swift Package |
| `Cmd+K` | Clear Console |
| `Cmd+\` | Toggle Breakpoint |
| `Cmd+Shift+Y` | Show Debugger Console |
| `Cmd+Shift+D` | Docker Build (GPU) |
| `Cmd+Shift+G` | Docker Run (GPU) |
| `Cmd+Shift+I` | Archive (Xcode) |
| `Cmd+Shift+B` | Code Analysis |
| `Cmd+0` | Toggle Navigator |
| `Cmd+Option+0` | Toggle Inspector |

---

## 🚀 Common Workflows

### Workflow 1: Create & Test a Swift Project
1. Go to **Setup Wizard**
2. Fill in Project Setup (Step 1)
3. Select Swift 5.9, optimization `-O` (Step 2)
4. Skip Python (Step 3)
5. Enable Docker for port 3001 (Step 4)
6. Enable XCTest with 90% coverage (Step 5)
7. Click **REPL Console** to test code
8. Click **Tests** to run test suite
9. Click **Build Swift** to compile

### Workflow 2: Debug Swift Code
1. Go to **Debugger**
2. Add breakpoint at desired file/line
3. Run code from **REPL Console**
4. Execution pauses at breakpoint
5. Inspect variables in the **Variables** panel
6. Step through code using debugger controls

### Workflow 3: Build & Run with GPU
1. Complete **Setup Wizard** (enable Docker, port 3001)
2. Click **Build Swift** to compile
3. Press `Cmd+Shift+D` to build Docker image with GPU support
4. Press `Cmd+Shift+G` to run container with GPU enabled
5. View output in **REPL Console**

### Workflow 4: Manage Dependencies
1. Go to **Swift Dependencies**
2. Click **Resolve** to update package lock
3. Click **Describe** to view dependency graph
4. Click **Build Dependency** to compile DREDGE package
5. Click **Build Swift** to recompile CLI with new deps

---

## 📊 Status Bar

Bottom of the screen shows:
- **Status Indicator** — 🟢 Ready (green) or 🔴 Error (red)
- **Current Time** — Updates every second

---

## 🔧 Settings You Just Applied

The PowerShell configuration script applied:

| Setting | Value |
|---------|-------|
| Swift Version | 5.9 |
| Optimization | `-O` (whole-module) |
| Python Version | 3.11 |
| Docker Image | `dredge-studio:gpu-latest` |
| Docker Port | 3001 |
| GPU Support | Metal + CUDA enabled |
| Test Framework | XCTest |
| Coverage Target | 90% |
| Auto-run Tests | Yes |
| Theme | Xcode Dark |

---

## 💡 Tips & Tricks

1. **Clear History:** Click the **Clear** button in the header to reset REPL output
2. **Export Session:** Click **Export** to download REPL history
3. **Multi-line Code:** In REPL, use semicolons to separate Swift statements: `let x = 5; print(x)`
4. **Fast Feedback:** Use REPL Console first to test code before building
5. **GPU Builds:** Always use `Cmd+Shift+D` (Docker Build) for GPU-accelerated compilation
6. **Test Coverage:** After running tests, coverage % appears in the test results panel

---

## 🆘 Troubleshooting

**Problem:** "Failed to create REPL session"
- **Solution:** Refresh browser, check Uvicorn server is running on port 8000

**Problem:** Tests show "No tests discovered"
- **Solution:** Ensure test files are in standard location (`Tests/` or `*_test.swift`)

**Problem:** Build fails with "Swift not found"
- **Solution:** Verify Swift installation in terminal: `swift --version`

**Problem:** Docker build fails
- **Solution:** Ensure Docker Desktop is running and GPU driver is installed

**Problem:** GPU not detected
- **Solution:** Check device selection in Settings → MCP → device (set to "auto" or specific GPU)

---

## 📞 Quick Links

- **UI Dashboard:** http://127.0.0.1:8000
- **API Documentation:** http://127.0.0.1:8000/docs
- **Health Check:** http://127.0.0.1:8000/health

---

**Enjoy building with DREDGE Studio!** 🚀
