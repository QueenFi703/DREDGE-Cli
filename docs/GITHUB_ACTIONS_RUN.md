# GitHub Actions Run Helper

This document describes how to use the `scripts/gh-actions-run.sh` helper and the
underlying `gh` (GitHub CLI) commands to inspect, rerun, and trigger GitHub Actions
workflows — without needing a Personal Access Token.

---

## Prerequisites

Install and authenticate the GitHub CLI once:

```bash
# Install (see https://cli.github.com for platform-specific instructions)
# macOS:
brew install gh

# Windows:
winget install --id GitHub.cli

# Linux (Debian/Ubuntu):
sudo apt install gh

# Authenticate (opens a browser; no PAT required)
gh auth login

# Verify
gh auth status
```

---

## scripts/gh-actions-run.sh

A small wrapper that accepts a **run id** or a **full run URL**, plus an optional
`-R owner/repo` flag, and can print the run summary, full logs, or structured JSON.

```
Usage: ./scripts/gh-actions-run.sh [OPTIONS] <run-id-or-url>

OPTIONS
  -R, --repo <owner/repo>   Target repository
  -l, --log                 Print full run logs
  -j, --json                Print run details as JSON
  -h, --help                Show help
```

### Examples — run id 23652704571 in QueenFi703/amazon-iap-kotlin

```bash
# 1. Summary (status, conclusion, jobs)
./scripts/gh-actions-run.sh 23652704571 -R QueenFi703/amazon-iap-kotlin

# 2. Full logs
./scripts/gh-actions-run.sh --log 23652704571 -R QueenFi703/amazon-iap-kotlin

# 3. JSON output (for scripting / CI gates)
./scripts/gh-actions-run.sh --json 23652704571 -R QueenFi703/amazon-iap-kotlin

# 4. Using the full run URL (repo is inferred automatically)
./scripts/gh-actions-run.sh \
  https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571
```

---

## Common gh Commands

### List recent runs

```bash
# Last 10 runs on main
gh run list -R QueenFi703/amazon-iap-kotlin --branch main --limit 10

# Filter by workflow name
gh run list -R QueenFi703/amazon-iap-kotlin \
  --workflow "AWS Gradle Pipeline (S3 + CodeArtifact + Device Farm)" \
  --limit 5
```

### View a specific run

```bash
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin
```

### Fetch logs

```bash
# Full logs (all jobs)
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin --log

# Logs for failed steps only
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin --log-failed
```

### JSON output for scripting

```bash
# Top-level run metadata
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin \
  --json status,conclusion,event,headBranch,headSha,createdAt,updatedAt,url,workflowName

# Job-level status
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin \
  --json jobs -q '.jobs[] | {name, status, conclusion, url}'
```

### Re-run failed jobs

```bash
# Re-run only the failed jobs in the run
gh run rerun 23652704571 -R QueenFi703/amazon-iap-kotlin --failed

# Re-run the entire run
gh run rerun 23652704571 -R QueenFi703/amazon-iap-kotlin
```

### Trigger a workflow_dispatch run

```bash
gh workflow run "AWS Gradle Pipeline (S3 + CodeArtifact + Device Farm)" \
  -R QueenFi703/amazon-iap-kotlin --ref main
```

---

## Related documentation

- [docs/GITHUB_ACTIONS_CONTAINERS.md](GITHUB_ACTIONS_CONTAINERS.md) — container workflow details
- [GitHub CLI manual](https://cli.github.com/manual/) — full `gh run` reference
