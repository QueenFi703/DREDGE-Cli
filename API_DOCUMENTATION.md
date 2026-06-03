# Interactive DREDGE API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API has no authentication. For production, add:

```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.get("/api/...")
async def protected_endpoint(credentials = Depends(security)):
	# Validate token
	pass
```

## Content Types

- Request: `application/json`
- Response: `application/json`
- WebSocket: `application/octet-stream` (JSON over WS)

---

## Endpoints

### Health Check

**GET** `/health`

Check server status and session count.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00",
  "repl_sessions": 2
}
```

---

### REPL Console

#### Create Session

**POST** `/api/repl/sessions`

Create a new REPL session for command execution.

**Query Parameters:**
- `language` (string, default: "swift") - Language for session: `swift`, `python`

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "language": "swift",
  "created_at": "2024-01-15T12:00:00"
}
```

---

#### Execute Command

**POST** `/api/repl/execute`

Execute a command in an existing session.

**Request Body:**
```json
{
  "command": "let x = 42\nprint(x)",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "language": "swift"
}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "command": "let x = 42\nprint(x)",
  "output": "42\n",
  "error": null,
  "execution_time": 0.234,
  "timestamp": "2024-01-15T12:00:00"
}
```

**Status Codes:**
- `200` - Command executed successfully
- `500` - Execution error

---

#### Get Session Info

**GET** `/api/repl/sessions/{session_id}`

Retrieve information about a specific session.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "language": "swift",
  "created_at": "2024-01-15T12:00:00",
  "command_count": 5,
  "last_command": "print(\"done\")"
}
```

---

#### Delete Session

**DELETE** `/api/repl/sessions/{session_id}`

Clean up and delete a REPL session.

**Response:**
```json
{
  "status": "deleted",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

#### WebSocket REPL Stream

**WS** `/ws/repl/{session_id}`

Real-time REPL execution with streaming output.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/repl/session-id');

ws.onopen = () => {
  ws.send(JSON.stringify({
	command: 'print("hello")'
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Output:', result.output);
  console.log('Error:', result.error);
  console.log('Execution Time:', result.execution_time);
};
```

**Message Format:**

Sent:
```json
{
  "command": "print(\"hello\")"
}
```

Received:
```json
{
  "command": "print(\"hello\")",
  "output": "hello\n",
  "error": null,
  "execution_time": 0.123
}
```

---

### Configuration Wizard

#### Get Wizard Steps

**GET** `/api/wizard/steps`

Retrieve all configuration wizard steps.

**Response:**
```json
[
  {
	"step_id": 1,
	"title": "Project Setup",
	"description": "Configure basic project settings",
	"fields": [
	  {
		"name": "project_name",
		"type": "text",
		"required": true
	  }
	],
	"validation_rules": {
	  "project_name": {
		"min_length": 1,
		"max_length": 50,
		"pattern": "^[a-zA-Z0-9_-]+$"
	  }
	}
  }
]
```

---

#### Update Configuration

**POST** `/api/config/update`

Update a configuration value.

**Request Body:**
```json
{
  "category": "project",
  "key": "name",
  "value": "MyAwesomeProject",
  "validate": true
}
```

**Response:**
```json
{
  "status": "updated",
  "result": {
	"category": "project",
	"key": "name",
	"old_value": "OldProject",
	"new_value": "MyAwesomeProject"
  }
}
```

---

#### Get Current Configuration

**GET** `/api/config/current`

Retrieve current configuration values.

**Response:**
```json
{
  "project": {
	"name": "MyProject",
	"description": "My awesome project",
	"version": "0.1.0"
  },
  "swift": {
	"version": "5.9",
	"optimization_level": "-O",
	"enable_testing": true
  },
  "python": {
	"version": "3.11",
	"virtual_env": true
  }
}
```

---

#### Get Configuration Schema

**GET** `/api/config/schema`

Get the schema defining allowed configuration.

**Response:**
```json
{
  "project": {
	"name": { "type": "string", "required": true },
	"description": { "type": "string" },
	"version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" }
  }
}
```

---

### Testing

#### Discover Tests

**GET** `/api/tests/discover`

Find all test files in a directory.

**Query Parameters:**
- `directory` (string, default: "swift/Tests") - Directory to search

**Response:**
```json
{
  "tests": [
	{
	  "name": "TestFile1",
	  "path": "/path/to/TestFile1Tests.swift",
	  "language": "swift"
	},
	{
	  "name": "test_module",
	  "path": "/path/to/test_module.py",
	  "language": "python"
	}
  ]
}
```

---

#### Run Test

**POST** `/api/tests/run`

Execute a specific test.

**Request Body:**
```json
{
  "name": "TestFile1",
  "description": "Test suite 1",
  "test_file": "/path/to/TestFile1Tests.swift",
  "language": "swift",
  "tags": ["unit", "core"]
}
```

**Response:**
```json
{
  "test_name": "TestFile1",
  "status": "passed",
  "duration": 1.234,
  "output": "Test results...",
  "error": null,
  "timestamp": "2024-01-15T12:00:00"
}
```

---

#### Run All Tests

**POST** `/api/tests/run-all`

Execute all tests in a directory.

**Query Parameters:**
- `directory` (string, default: "swift/Tests") - Directory containing tests

**Response:**
```json
{
  "results": [
	{
	  "test_name": "TestFile1",
	  "status": "passed",
	  "duration": 0.234,
	  "output": "✓ All assertions passed",
	  "error": null
	},
	{
	  "test_name": "TestFile2",
	  "status": "failed",
	  "duration": 0.567,
	  "output": "Test output",
	  "error": "Assertion failed"
	}
  ]
}
```

---

### Debugging

#### Set Breakpoint

**POST** `/api/debug/breakpoint`

Create a breakpoint at a specific location.

**Request Body:**
```json
{
  "file": "DREDGE.swift",
  "line": 42,
  "condition": "x > 10",
  "enabled": true
}
```

**Response:**
```json
{
  "status": "set",
  "breakpoint": {
	"file": "DREDGE.swift",
	"line": 42,
	"condition": "x > 10",
	"enabled": true
  }
}
```

---

#### List Breakpoints

**GET** `/api/debug/breakpoints`

Get all active breakpoints.

**Response:**
```json
{
  "breakpoints": [
	{
	  "file": "DREDGE.swift",
	  "line": 42,
	  "condition": "x > 10",
	  "enabled": true
	},
	{
	  "file": "main.swift",
	  "line": 15,
	  "condition": null,
	  "enabled": false
	}
  ]
}
```

---

#### Delete Breakpoint

**DELETE** `/api/debug/breakpoint/{file}/{line}`

Remove a breakpoint.

**Path Parameters:**
- `file` - File name
- `line` - Line number

**Response:**
```json
{
  "status": "deleted"
}
```

---

#### WebSocket Debug Stream

**WS** `/ws/debug/{session_id}`

Real-time debugging information.

**Message Types:**

Get Stack:
```json
{
  "type": "get_stack"
}
```

Response:
```json
{
  "type": "get_stack",
  "data": {
	"stack": [
	  {
		"function": "main",
		"file": "main.swift",
		"line": 15
	  }
	]
  }
}
```

Get Variables:
```json
{
  "type": "get_variables"
}
```

Response:
```json
{
  "type": "get_variables",
  "data": {
	"variables": {
	  "x": { "type": "Int", "value": "42" },
	  "name": { "type": "String", "value": "\"hello\"" }
	}
  }
}
```

---

### Build & Compilation

#### Build Swift

**POST** `/api/build/swift`

Build the Swift package.

**Query Parameters:**
- `target` (string, default: "debug") - Build configuration: `debug`, `release`

**Response:**
```json
{
  "status": "success",
  "output": "Building for debugging...\nBuild complete!",
  "errors": []
}
```

**Status Values:** `success`, `failed`, `in_progress`

---

#### Build Python

**POST** `/api/build/python`

Build the Python package.

**Response:**
```json
{
  "status": "success",
  "output": "Compiling...\nBuild complete!"
}
```

---

### Shell Execution

#### Execute Command

**POST** `/api/shell/execute`

Run a shell command with output capture.

**Request Body:**
```json
{
  "cmd": "swift package describe"
}
```

**Response:**
```json
{
  "command": "swift package describe",
  "stdout": "DREDGE 0.1.0: CLI tool for Swift development",
  "stderr": "",
  "return_code": 0
}
```

---

### File Management

#### Upload File

**POST** `/api/files/upload`

Upload a file to the server.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (binary)

**Response:**
```json
{
  "filename": "myfile.swift",
  "size": 1234,
  "path": "uploads/myfile.swift"
}
```

---

## Error Handling

All errors return appropriate HTTP status codes with error details:

```json
{
  "detail": "Session not found"
}
```

**Common Status Codes:**
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `404` - Not Found
- `500` - Server Error
- `503` - Service Unavailable

---

## Rate Limiting

Currently unlimited. Add with:

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/repl/execute")
@limiter.limit("10/minute")
async def execute_repl_command(request: Request, ...):
	pass
```

---

## Pagination

Large result sets support pagination:

```
GET /api/tests/discover?skip=0&limit=10
```

---

## Testing with curl

```bash
# Create session
curl -X POST http://localhost:8000/api/repl/sessions?language=swift

# Execute command
curl -X POST http://localhost:8000/api/repl/execute \
  -H "Content-Type: application/json" \
  -d '{
	"command": "print(\"hello\")",
	"session_id": "xxx",
	"language": "swift"
  }'

# Run tests
curl -X POST http://localhost:8000/api/tests/run-all

# Build Swift
curl -X POST http://localhost:8000/api/build/swift?target=debug

# Get configuration
curl http://localhost:8000/api/config/current
```

---

## SDK Availability

JavaScript/TypeScript SDK coming soon!

```javascript
import { DREDGEClient } from '@dredge/sdk';

const client = new DREDGEClient('http://localhost:8000');
const session = await client.repl.createSession('swift');
const result = await client.repl.execute(session.id, 'print("hello")');
```

---

## Changelog

### v1.0.0
- Initial release with REPL, Wizard, Tests, Debug, Build endpoints
- WebSocket support for streaming
- Basic authentication placeholder
- Docker support
