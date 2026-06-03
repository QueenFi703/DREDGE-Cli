# Interactive DREDGE Studio - User Guide

## Overview

Interactive DREDGE Studio provides a complete web-based development environment for Swift, Python, and Xcode projects. It includes:

- **REPL Console** - Execute Swift and Python commands interactively
- **Debugger** - Set breakpoints and inspect variables
- **Test Runner** - Discover and execute tests
- **Configuration Wizard** - Step-by-step project setup
- **Build Tools** - Compile Swift and Python packages
- **Shell Access** - Execute arbitrary commands

## Getting Started

### Prerequisites

- Docker and docker-compose installed
- Python 3.8+
- Swift 5.8+ (for Swift development)
- Git

### Installation & Setup

#### Windows (PowerShell)

```powershell
# Make script executable and run
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\start-interactive.ps1
```

#### macOS/Linux (Bash)

```bash
# Make script executable and run
chmod +x start-interactive.sh
./start-interactive.sh
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements-interactive.txt

# Build Docker image (if needed)
docker build --target dev -t dredge-dev:latest .

# Start services
docker-compose -f docker-compose.interactive.yml up -d

# Access the UI
# Open http://localhost:8000 in your browser
```

## Features

### 1. REPL Console

Execute commands directly in Swift or Python:

```swift
// Swift example
let greeting = "Hello, DREDGE!"
print(greeting)

// Query Swift package info
import Foundation
let date = Date()
print(date.description)
```

```python
# Python example
name = "DREDGE"
print(f"Hello, {name}!")

import json
data = {"key": "value"}
print(json.dumps(data, indent=2))
```

**Features:**
- Syntax highlighting
- Command history (use arrow keys)
- Session persistence
- Export session to file
- Real-time output streaming via WebSocket

### 2. Configuration Wizard

The setup wizard guides you through:

1. **Project Setup** - Name, description, version
2. **Swift Configuration** - Version, optimization, testing
3. **Python Configuration** - Version, virtual environment
4. **Docker Configuration** - Enable Docker, image, ports
5. **Testing Configuration** - Framework, coverage, automation

Navigate using Next/Previous buttons or click steps directly.

### 3. Test Runner

Automatically discover and run tests:

```
Tests discovered:
✓ TestFile1.swift
✓ TestFile2.swift
  - test_swift_feature1 ✓ (0.234s)
  - test_swift_feature2 ✓ (0.156s)
```

**Features:**
- Auto-discovery of test files
- Individual test execution
- Run all tests
- Test results with timing
- Pass/fail/skip indicators

### 4. Debugger

Set breakpoints and inspect execution:

1. Click **Debugger** in sidebar
2. Set breakpoints by clicking line numbers
3. Run code in REPL or execute build
4. View call stack and variables
5. Step through execution

**Available Commands:**
- Continue - Resume execution
- Step Over - Execute current line
- Step Into - Enter function
- Step Out - Exit function
- Get Variables - Inspect current scope
- Get Stack - View call stack

### 5. Build Tools

Build your project directly from the UI:

**Build Swift**
```
Status: Building...
Output: Building for debugging...
Compilation complete (2.341s)
Executable: /path/to/.build/debug/DREDGE
```

**Build Python**
```
Status: Building...
Output: Compiling sources...
Build complete (0.891s)
```

### 6. Shell Access

Execute arbitrary commands:

```bash
# List files
ls -la

# Git operations
git status
git add .
git commit -m "Update code"

# Package management
pip list
swift package describe
```

## API Endpoints

All features are accessible via REST API:

### REPL

```bash
# Create session
POST /api/repl/sessions?language=swift

# Execute command
POST /api/repl/execute
{
  "session_id": "xxx",
  "command": "print(\"Hello\")",
  "language": "swift"
}

# Get session info
GET /api/repl/sessions/{session_id}

# Delete session
DELETE /api/repl/sessions/{session_id}
```

### Configuration

```bash
# Get wizard steps
GET /api/wizard/steps

# Update configuration
POST /api/config/update
{
  "category": "project",
  "key": "name",
  "value": "MyProject"
}

# Get current config
GET /api/config/current

# Get config schema
GET /api/config/schema
```

### Tests

```bash
# Discover tests
GET /api/tests/discover?directory=swift/Tests

# Run specific test
POST /api/tests/run
{
  "name": "TestFile",
  "test_file": "path/to/TestFile.swift",
  "language": "swift"
}

# Run all tests
POST /api/tests/run-all
```

### Build

```bash
# Build Swift
POST /api/build/swift?target=debug

# Build Python
POST /api/build/python
```

### Debug

```bash
# Set breakpoint
POST /api/debug/breakpoint
{
  "file": "DREDGE.swift",
  "line": 42,
  "condition": "x > 10"
}

# List breakpoints
GET /api/debug/breakpoints

# Delete breakpoint
DELETE /api/debug/breakpoint/{file}/{line}
```

### Shell

```bash
# Execute shell command
POST /api/shell/execute
{
  "cmd": "swift package describe"
}
```

## WebSocket Connections

Real-time streaming via WebSocket:

### REPL Stream

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/repl/session-id');

ws.onopen = () => {
  ws.send(JSON.stringify({ command: 'print("Hello")' }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.output);
};
```

### Debug Stream

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/debug/session-id');

ws.send(JSON.stringify({ type: 'get_stack' }));
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.data);
};
```

## Advanced Usage

### Docker Execution

By default, Swift commands run locally. To execute inside Docker:

```bash
# Modify environment variables
export DREDGE_SWIFT_DOCKER=true
docker-compose -f docker-compose.interactive.yml restart dredge-api
```

### Custom Configuration

Create `.dredge/config.yaml`:

```yaml
project:
  name: MyProject
  description: My awesome project
  version: 0.1.0

swift:
  version: "5.9"
  optimization: "-O"
  enable_testing: true

python:
  version: "3.11"
  virtual_env: true

docker:
  enabled: true
  image: "dredge-dev:latest"
  port: 8000

testing:
  framework: XCTest
  coverage_threshold: 80
  auto_run_tests: false
```

### Session Export

Export your REPL session:

1. Click **Export** button
2. Session saved as `session_<timestamp>.json`
3. Contains all commands and outputs

```json
{
  "session_id": "xxx",
  "language": "swift",
  "created_at": "2024-01-01T12:00:00",
  "commands": [
	"let x = 42",
	"print(x)"
  ],
  "outputs": [
	"42"
  ]
}
```

## Troubleshooting

### Port Already in Use

If port 8000 is already in use:

```powershell
# Windows: Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
docker-compose -f docker-compose.interactive.yml down
# Edit docker-compose.interactive.yml and change port
docker-compose -f docker-compose.interactive.yml up -d
```

### Docker Build Fails

```bash
# Check Docker resources
docker system df

# Clean up
docker system prune -a

# Rebuild image
docker build --target dev --no-cache -t dredge-dev:latest .
```

### Swift Execution Fails

```bash
# Check Swift installation
swift --version

# Run tests to verify
swift test

# If using Docker, ensure image is built
docker inspect dredge-dev:latest
```

### WebSocket Connection Issues

```javascript
// Check browser console for errors
// Ensure WebSocket URLs use ws:// not http://
// Check firewall settings
// Verify server is running: curl http://localhost:8000/health
```

## Performance Tips

1. **Use Docker** - Isolates execution and prevents system pollution
2. **Export sessions** - Save progress for long-running work
3. **Clear REPL** - Use Clear button to reset output buffer
4. **Optimize builds** - Use `-Osize` for smaller binaries
5. **Cache packages** - Let Docker cache Swift packages

## Security Considerations

⚠️ **Important**: This tool exposes command execution capabilities. Use only in:

- Local development environments
- Trusted networks
- Behind authentication
- In Docker containers

**Never expose this publicly without authentication!**

To add basic auth:

```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

@app.get("/api/repl/sessions")
async def create_session(credentials: HTTPBasicCredentials = Depends(security)):
	# Verify credentials
	if credentials.username != "admin" or credentials.password != "secret":
		raise HTTPException(status_code=401)
	# ... rest of endpoint
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Enter | Execute REPL command |
| ↑/↓ | Command history (REPL) |
| Ctrl+L | Clear output |
| Tab | Auto-complete (planned) |
| Ctrl+S | Export session (planned) |

## Support

For issues or feature requests:

1. Check troubleshooting section
2. View logs: `docker-compose -f docker-compose.interactive.yml logs -f`
3. Open issue on GitHub with:
   - Full error message
   - Steps to reproduce
   - Environment info (OS, Docker version, Swift version)

## Next Steps

- Explore the REPL with Swift and Python commands
- Run the test suite to validate your setup
- Use the configuration wizard to customize your project
- Set breakpoints and debug your code
- Export sessions for future reference
