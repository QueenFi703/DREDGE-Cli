# DREDGE-Cli Troubleshooting Guide

This guide helps diagnose and resolve common issues when using DREDGE-Cli with Swift Package Manager, Python, and MCP Server.

## Table of Contents

- [Swift Package Issues](#swift-package-issues)
- [MCP Server Issues](#mcp-server-issues)
- [Python Environment Issues](#python-environment-issues)
- [Integration Issues](#integration-issues)
- [Common Error Messages](#common-error-messages)
- [Platform-Specific Issues](#platform-specific-issues)

---

## Swift Package Issues

### Issue: Cannot resolve Swift package dependencies

**Symptoms:**
```
error: package at '...' is using Swift tools version 5.9.0 but the installed version is 5.x.x
```

**Solution:**
1. Check your Swift version:
   ```bash
   swift --version
   ```
2. Ensure you have Swift 5.9 or later installed
3. Update Swift toolchain: https://swift.org/download/
4. Clean and rebuild:
   ```bash
   swift package clean
   swift build
   ```

### Issue: Module 'DREDGECli' not found

**Symptoms:**
```
error: no such module 'DREDGECli'
```

**Solution:**
1. Verify you're using the correct module name (no hyphen):
   ```swift
   import DREDGECli  // ✓ Correct
   import DREDGE-Cli  // ✗ Wrong
   ```

2. Ensure dependencies are resolved:
   ```bash
   swift package resolve
   swift build
   ```

### Issue: Tests fail to compile on Linux

**Symptoms:**
```
Note: MCP Client networking requires macOS 12.0+ or iOS 15.0+
```

**Expected Behavior:**
- Network-dependent tests will skip on Linux
- Core API tests should still pass
- This is intentional - networking features are Apple platform-specific

**Verification:**
```bash
swift test
# Should show: Executed X tests, with Y tests skipped and 0 failures
```

---

## MCP Server Issues

### Issue: MCP Server connection refused

**Symptoms:**
```
Error: Connection refused to http://localhost:3002
```

**Solution:**
1. Check if MCP server is running:
   ```bash
   lsof -i :3002
   # or
   curl http://localhost:3002/
   ```

2. Start the MCP server:
   ```bash
   # From repository root
   python -m dredge mcp
   # or
   dredge-cli mcp --port 3002
   ```

3. Verify server is responding:
   ```bash
   curl http://localhost:3002/
   # Should return JSON with server info
   ```

**Expected Output:**
```json
{
  "name": "DREDGE Quasimoto String Theory MCP Server",
  "version": "0.1.4",
  "protocol": "Model Context Protocol v1.0",
  "capabilities": {...}
}
```

### Issue: MCP Server returns 404

**Symptoms:**
```
Error: 404 Not Found
```

**Common Causes:**
1. Using wrong endpoint path
   - ✓ Correct: `http://localhost:3002/mcp`
   - ✗ Wrong: `http://localhost:3002/api/mcp`

2. Wrong HTTP method
   - GET `/` - Server info
   - POST `/mcp` - MCP operations

**Verification:**
```bash
# List capabilities
curl http://localhost:3002/

# Load model
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "load_model", "params": {"model_type": "quasimoto_1d"}}'
```

### Issue: MCP Server operation fails

**Symptoms:**
```json
{
  "success": false,
  "error": "Invalid model type"
}
```

**Solution:**
1. Check available model types:
   ```bash
   curl http://localhost:3002/ | jq '.capabilities.models'
   ```

2. Valid model types:
   - `quasimoto_1d` - 1D wave function
   - `quasimoto_4d` - 4D spatiotemporal  
   - `quasimoto_6d` - 6D high-dimensional
   - `quasimoto_ensemble` - Ensemble model
   - `string_theory` - String theory model

3. Example valid request:
   ```bash
   curl -X POST http://localhost:3002/mcp \
     -H "Content-Type: application/json" \
     -d '{
       "operation": "load_model",
       "params": {"model_type": "quasimoto_1d"}
     }'
   ```

---

## Python Environment Issues

### Issue: Module 'dredge' not found

**Symptoms:**
```
ModuleNotFoundError: No module named 'dredge'
```

**Solution:**
1. Verify package is installed:
   ```bash
   pip list | grep dredge
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. Check Python version compatibility:
   ```bash
   python --version  # Should be 3.9-3.12
   ```

### Issue: Dependencies not installed

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
1. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify installation:
   ```bash
   pip list | grep -E "torch|flask|numpy"
   ```

3. Expected versions:
   - flask >= 3.0.0
   - torch >= 2.0.0
   - numpy >= 1.24.0
   - matplotlib >= 3.5.0

### Issue: Port already in use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
1. Find process using the port:
   ```bash
   lsof -i :3002
   # or
   lsof -i :3001
   ```

2. Kill the process:
   ```bash
   kill -9 <PID>
   ```

3. Use different port:
   ```bash
   dredge-cli mcp --port 3003
   # or
   dredge-cli serve --port 3004
   ```

---

## Integration Issues

### Issue: Swift cannot connect to Python MCP server

**Symptoms:**
- Tests skip with "MCP Server not available"
- Connection timeout errors

**Debugging Steps:**

1. **Verify server is running:**
   ```bash
   curl http://localhost:3002/
   ```

2. **Check firewall/network:**
   ```bash
   # Test connectivity
   nc -zv localhost 3002
   ```

3. **Enable debug logging in Swift:**
   ```swift
   let client = MCPClient(serverURL: "http://localhost:3002")
   print("Attempting connection to: \(client.serverURL)")
   ```

4. **Run end-to-end tests:**
   ```bash
   # Start server in one terminal
   python -m dredge mcp
   
   # In another terminal, run tests
   swift test --filter EndToEndTests
   ```

**Expected Test Output:**
```
Test Suite 'EndToEndTests' started
✓ Connected to: DREDGE Quasimoto String Theory MCP Server
✓ Server version: 0.1.4
✓ Loaded model: quasimoto_1d
Test Suite 'EndToEndTests' passed
```

### Issue: String Theory calculations differ between Python and Swift

**Symptoms:**
- Results don't match between implementations

**Verification:**

1. **Test Python implementation:**
   ```python
   from dredge.string_theory import StringVibration
   sv = StringVibration(dimensions=10)
   spectrum = sv.compute_spectrum(max_modes=5)
   print(spectrum)
   ```

2. **Test Swift implementation:**
   ```swift
   let st = StringTheory(dimensions: 10)
   let spectrum = st.modeSpectrum(maxModes: 5)
   print(spectrum)
   ```

3. **Compare results:**
   - Energy levels should increase monotonically
   - Vibrational modes at boundaries (x=0, x=1) should be 0
   - Maximum amplitude at x=0.5 for n=1 should be 1.0

---

## Common Error Messages

### `URLError: Connection refused`

**Meaning:** Cannot connect to MCP server

**Fix:** Start the MCP server (see [MCP Server Issues](#mcp-server-issues))

### `JSONDecodeError: Expecting value`

**Meaning:** Server returned non-JSON response

**Possible Causes:**
- Server crashed or returned error page
- Wrong endpoint URL
- Server not fully started

**Fix:**
```bash
# Check server logs
python -m dredge mcp --debug

# Verify endpoint
curl -v http://localhost:3002/
```

### `ModuleNotFoundError: No module named 'quasimoto_extended_benchmark'`

**Meaning:** Benchmark module not in Python path

**Fix:**
```bash
# Ensure benchmarks directory is accessible
export PYTHONPATH="${PYTHONPATH}:$(pwd)/benchmarks"

# Or install in development mode
pip install -e .
```

### `Test skipped - Network tests only available on Apple platforms`

**Meaning:** Running network tests on Linux

**Expected:** This is normal behavior
- Core API tests run on all platforms
- Network tests require macOS 12+/iOS 15+
- Tests gracefully skip when platform doesn't support networking

---

## Platform-Specific Issues

### macOS

**Issue: Xcode Command Line Tools not installed**

**Fix:**
```bash
xcode-select --install
```

**Issue: Permission denied errors**

**Fix:**
```bash
# Use --user flag for pip
pip install --user -e .
```

### Linux

**Issue: Missing system libraries for PyTorch**

**Fix:**
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel
```

**Issue: Swift not installed**

**Fix:**
```bash
# Follow official Swift installation guide
# https://swift.org/download/#linux
```

### GitHub Codespaces

**Issue: Ports not forwarded**

**Fix:**
1. Check `.devcontainer/devcontainer.json` includes:
   ```json
   "forwardPorts": [3001, 3002]
   ```

2. Manually forward ports in VS Code:
   - Ports tab → Forward Port → Enter 3001, 3002

**Issue: Out of memory**

**Expected Behavior:**
- MCP server with PyTorch may use significant memory
- Codespaces has 8GB RAM limit

**Fix:**
```bash
# Use smaller models
# Reduce batch sizes in benchmarks
# Close unused applications
```

---

## Getting Help

If you continue to experience issues:

1. **Check existing issues:** https://github.com/QueenFi703/DREDGE-Cli/issues
2. **Create a new issue** with:
   - Platform (macOS/Linux/iOS)
   - Swift version (`swift --version`)
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce

3. **Include diagnostic information:**
   ```bash
   # System info
   uname -a
   
   # Swift version
   swift --version
   
   # Python packages
   pip list
   
   # Running processes
   ps aux | grep -E "dredge|python"
   
   # Port status
   lsof -i :3001 -i :3002
   ```

---

## Quick Reference

### Essential Commands

```bash
# Python: Install
pip install -e .

# Python: Start MCP server
python -m dredge mcp

# Python: Start Dolly server  
python -m dredge serve

# Python: Run tests
pytest tests/ -v

# Swift: Build
swift build

# Swift: Run
swift run dredge-cli

# Swift: Test
swift test

# Swift: Test specific suite
swift test --filter IntegrationTests

# Check server status
curl http://localhost:3002/
curl http://localhost:3001/health
```

### Default Ports

- **3001** - DREDGE x Dolly Server
- **3002** - MCP Server (Quasimoto + String Theory)

### Default Values

- **String Theory Dimensions:** 10
- **String Theory Length:** 1.0
- **MCP Server URL:** http://localhost:3002

---

*Last updated: 2026-01-17*
