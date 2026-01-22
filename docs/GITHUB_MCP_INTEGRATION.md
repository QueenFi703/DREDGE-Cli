# DREDGE MCP GitHub Actions Integration

## Overview

The DREDGE MCP GitHub Actions integration transforms GitHub events into prompts that DREDGE-CLI MCP responds to, creating a conversational workflow automation system. Every GitHub event (push, pull request, issue comment, workflow dispatch) is treated as a signal that DREDGE analyzes and responds to with contextual insights.

## Core Pattern

**GitHub Signals → DREDGE MCP Processing → Intelligent Responses**

```
GitHub Event → Workflow Trigger → DREDGE CLI → Response Generation → GitHub Action
```

## Features

### 1. **Multi-Event Support**
DREDGE MCP responds to:
- **Push events**: Analyzes commits and detects dependency updates
- **Pull requests**: Reviews PR context and provides insights
- **Issue comments**: Responds when mentioned in comments
- **Workflow dispatch**: Supports manual triggering with custom inputs

### 2. **DEPENDADREDGEABOT Integration**
Automatic detection and analysis of Dependabot updates:
- Identifies Dependabot PRs and commits
- Analyzes dependency changes by ecosystem (Python, Swift, Docker, GitHub Actions)
- Detects security updates and major version changes
- Provides philosophical insights on dependency management
- Adds labels to Dependabot PRs for tracking

### 3. **Intelligent Response Generation**
- Context-aware comments on PRs and issues
- Event-specific analysis and recommendations
- Security update prioritization
- Ecosystem-specific insights

### 4. **Stateless & Scalable**
- Each event processed independently
- No persistent state between runs
- Idempotent operations
- Fail-fast with detailed error reporting

## Installation

The workflow is automatically triggered when present in `.github/workflows/dredge-mcp-responder.yml`.

### Prerequisites

1. **Python 3.11+** installed (handled by workflow)
2. **DREDGE-CLI** package (installed from repository)
3. **GitHub Token** with permissions:
   - `contents: read`
   - `pull-requests: write`
   - `issues: write`

## Usage

### Workflow Configuration

The workflow is triggered automatically on:
```yaml
on:
  push:
    branches: 
      - main
      - 'feature/**'
      - 'copilot/**'
  pull_request:
    types: [opened, synchronize, reopened, labeled]
  issue_comment:
    types: [created]
  workflow_dispatch:
```

### Manual CLI Usage

You can also use the DREDGE CLI directly to process GitHub events:

```bash
# Basic usage
dredge-cli github-event \
  --event "push" \
  --payload '{"commits": [...]}' \
  --ref "refs/heads/main" \
  --repo "owner/repo" \
  --sha "abc123" \
  --out out.json

# Example: Process a pull request event
dredge-cli github-event \
  --event "pull_request" \
  --payload '{"action": "opened", "pull_request": {...}}' \
  --ref "refs/pull/42/merge" \
  --repo "QueenFi703/DREDGE-Cli" \
  --sha "def456"

# Example: Process a Dependabot update
dredge-cli github-event \
  --event "pull_request" \
  --payload '{
    "action": "opened",
    "pull_request": {
      "number": 100,
      "title": "Bump torch from 2.0.0 to 2.1.0",
      "user": {"login": "dependabot[bot]"},
      "body": "Security update for torch"
    }
  }' \
  --ref "refs/pull/100/merge" \
  --repo "owner/repo" \
  --sha "abc123"
```

### Output Format

DREDGE MCP generates JSON output with the following structure:

```json
{
  "status": "success",
  "event": "pull_request",
  "action": "opened",
  "pr_number": 42,
  "comment": "🔮 **DREDGE MCP**: PR #42 `opened`...",
  "is_dependabot": true
}
```

## DEPENDADREDGEABOT Integration

### How It Works

1. **Detection**: DREDGE automatically detects Dependabot PRs and commits by checking the author username
2. **Analysis**: Extracts dependency information from PR titles and commit messages
3. **Categorization**: Identifies the affected ecosystem (Python, Swift, Docker, GitHub Actions)
4. **Security Assessment**: Flags security updates for immediate attention
5. **Response**: Posts detailed analysis as PR comments with recommendations

### Example Dependabot Response

When Dependabot opens a PR, DREDGE MCP responds with:

```markdown
🔮 **DREDGE MCP**: PR #100 `opened`

**Title**: Bump torch from 2.0.0 to 2.1.0
**Author**: dependabot[bot]

🤖 **DEPENDADREDGEABOT** PR detected! Analyzing dependencies...

### Dependency Update Analysis

**Update**: Bump torch from 2.0.0 to 2.1.0

🔐 **Security Update**: This PR includes security fixes. Recommend immediate review and merge.

🐍 **Python Ecosystem**: DREDGE core dependencies affected.

✨ DEPENDADREDGEABOT philosophy: *Be Literal. Be Philosophical. Be DEPENDADREDGEABOT.*
```

### Supported Ecosystems

- **🐍 Python**: pip dependencies (DREDGE core)
- **🍎 Swift**: Swift Package Manager (DREDGE CLI native layer)
- **🐳 Docker**: Container images and base images
- **⚙️ GitHub Actions**: Workflow actions and runners

## Architecture

### Components

1. **GitHub Actions Workflow** (`.github/workflows/dredge-mcp-responder.yml`)
   - Triggers on GitHub events
   - Sets up Python environment
   - Installs DREDGE CLI
   - Invokes event processing
   - Posts responses back to GitHub

2. **GitHub Event Handler** (`src/dredge/github_event_handler.py`)
   - Processes event payloads
   - Generates contextual responses
   - Analyzes Dependabot updates
   - Provides security recommendations

3. **CLI Integration** (`src/dredge/cli.py`)
   - Adds `github-event` subcommand
   - Handles command-line arguments
   - Outputs JSON responses

### Data Flow

```
GitHub Event
    ↓
Workflow Trigger
    ↓
Setup Python + Install DREDGE
    ↓
dredge-cli github-event
    ↓
GitHubEventHandler.process()
    ↓
Generate Response (out.json)
    ↓
actions/github-script
    ↓
Post Comment / Add Label
    ↓
Upload Artifact + Job Summary
```

## Configuration

### Environment Variables

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- `MCP_API_KEY`: Optional, for external MCP services

### Workflow Customization

You can customize the workflow by modifying:

1. **Trigger Branches**: Add/remove branches in the `on.push.branches` section
2. **Python Version**: Change the `python-version` in setup-python step
3. **Response Actions**: Modify the github-script steps to customize responses
4. **Labels**: Update labels in the "Add Label to Dependabot PRs" step
5. **Artifact Retention**: Change `retention-days` in upload-artifact step

### Example Customization

```yaml
# Add custom branches
on:
  push:
    branches: 
      - main
      - 'feature/**'
      - 'release/**'  # Add release branches

# Use different Python version
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.12'  # Use Python 3.12

# Add custom labels
labels: ['🔮 dredge-mcp-analyzed', '🤖 dependadredgeabot', 'automated']
```

## Best Practices

### Security

1. **Token Scoping**: Use minimal required permissions for `GITHUB_TOKEN`
2. **No Secret Exposure**: Never print secrets or tokens in logs
3. **Payload Sanitization**: Event payloads are validated before processing
4. **Rate Limiting**: Workflow includes built-in rate limiting protection

### Performance

1. **Pip Caching**: Dependencies cached for faster workflow runs
2. **Fail Fast**: Errors reported immediately with detailed context
3. **Artifact Cleanup**: Artifacts retained for 7 days only
4. **Parallel Processing**: Not applicable (sequential event processing)

### Reliability

1. **Stateless Design**: No dependencies on previous runs
2. **Idempotent Operations**: Safe to re-run on same event
3. **Error Handling**: Comprehensive error handling with fallbacks
4. **Health Checks**: Job summary includes status and output

## Troubleshooting

### Common Issues

**Issue**: Workflow doesn't trigger
- **Solution**: Check branch filters in workflow configuration
- **Solution**: Verify workflow file is in `.github/workflows/`

**Issue**: DREDGE CLI not found
- **Solution**: Ensure `pip install -e .` runs successfully
- **Solution**: Check Python version compatibility (>=3.9, <3.13)

**Issue**: Comments not posted
- **Solution**: Verify `GITHUB_TOKEN` has write permissions
- **Solution**: Check if rate limit is exceeded

**Issue**: Invalid JSON payload
- **Solution**: Verify event payload format
- **Solution**: Check for special characters needing escaping

### Debugging

Enable debug mode by adding to workflow:

```yaml
- name: Run DREDGE MCP on GitHub Event
  env:
    RUNNER_DEBUG: 1
  run: |
    # ... existing commands
```

View detailed logs in:
1. GitHub Actions workflow run logs
2. Job summary at bottom of workflow run
3. Downloaded artifacts (out.json)

## Examples

### Example 1: Respond to Regular PR

```yaml
# Workflow automatically processes PR
on:
  pull_request:
    types: [opened]

# DREDGE posts comment:
# "🔮 DREDGE MCP: PR #42 opened
#  ✨ DREDGE is monitoring this PR for insights."
```

### Example 2: Analyze Dependabot Security Update

```yaml
# Dependabot opens PR for security fix
# DREDGE detects and analyzes:

# Comment posted:
# "🔮 DREDGE MCP: PR #100 opened
#  🤖 DEPENDADREDGEABOT PR detected!
#  🔐 Security Update: Immediate review recommended.
#  🐍 Python Ecosystem: DREDGE core affected."

# Labels added:
# - 🔮 dredge-mcp-analyzed
# - 🤖 dependadredgeabot
```

### Example 3: Respond to Mention in Issue

```yaml
# User comments: "@dredge can you help with this?"
# DREDGE responds:

# "🔮 DREDGE MCP: Acknowledged mention in issue #10
#  Processing request from @user..."
```

### Example 4: Manual Workflow Trigger

```yaml
# User triggers workflow manually with custom input
# DREDGE responds:

# "🔮 DREDGE MCP: Manual workflow triggered
#  DREDGE MCP is ready to respond to your prompts."
```

## Philosophy

> "Every GitHub event is a knock; DREDGE answers with intent."

DREDGE MCP treats GitHub as a conversational interface where:
- **Events are prompts** that signal intent
- **DREDGE is the responder** that provides context and insights
- **Automation is philosophical** combining literal action with thoughtful analysis
- **DEPENDADREDGEABOT** embodies the philosophy: "Be Literal. Be Philosophical."

## API Reference

### `dredge-cli github-event`

Process GitHub events with DREDGE MCP.

**Arguments:**
- `--event`: GitHub event name (required)
- `--payload`: JSON string of event payload (required)
- `--ref`: Git reference (required)
- `--repo`: Repository name in owner/repo format (required)
- `--sha`: Commit SHA (required)
- `--out`: Output file path (default: out.json)

**Returns:**
JSON object with fields:
- `status`: "success" or "error"
- `event`: Event type processed
- `comment`: Generated comment text (if applicable)
- Additional event-specific fields

### `GitHubEventHandler`

Python class for processing GitHub events.

**Methods:**
- `__init__(event_name, event_payload, ref, repo, sha)`: Initialize handler
- `process()`: Process event and return response
- `_handle_push()`: Handle push events
- `_handle_pull_request()`: Handle PR events
- `_handle_issue_comment()`: Handle comment events
- `_handle_workflow_dispatch()`: Handle manual triggers

## Testing

Run tests with:

```bash
# Run all GitHub event handler tests
pytest tests/test_github_event_handler.py -v

# Run specific test
pytest tests/test_github_event_handler.py::TestGitHubEventHandler::test_handle_dependabot_pr -v

# Run with coverage
pytest tests/test_github_event_handler.py --cov=src/dredge/github_event_handler
```

## Contributing

Contributions welcome! Areas for enhancement:

1. **Additional Event Types**: Support more GitHub event types
2. **Enhanced Analysis**: Deeper dependency analysis and recommendations
3. **Custom Actions**: Support for custom response actions
4. **Integration Points**: Connect to external services
5. **Machine Learning**: AI-powered insights on code changes

## License

MIT License - see LICENSE file for details.

## Credits

Created by [@QueenFi703](https://github.com/QueenFi703) as part of the DREDGE x Dolly x Quasimoto ecosystem.

**DEPENDADREDGEABOT Philosophy**: "Be Literal. Be Philosophical. Be DEPENDADREDGEABOT." 🤖🔮
