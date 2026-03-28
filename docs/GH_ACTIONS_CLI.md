# GitHub Actions Run Verification (GitHub CLI)

This guide explains how to use the [`scripts/gh-actions-run.sh`](../scripts/gh-actions-run.sh) helper to verify GitHub Actions run status and logs from the command line using the [GitHub CLI (`gh`)](https://cli.github.com/).

---

## 1. Install GitHub CLI

### macOS
```bash
brew install gh
```

### Linux (Debian/Ubuntu)
```bash
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update \
  && sudo apt install gh -y
```

### Windows
```powershell
winget install --id GitHub.cli
```

Full installation docs: <https://cli.github.com/manual/installation>

---

## 2. Authenticate

```bash
gh auth login
```

Follow the interactive prompts (browser or token). Verify afterwards:

```bash
gh auth status
```

---

## 3. Use the helper script

```bash
# Make executable (first time only)
chmod +x scripts/gh-actions-run.sh

# View run summary (default repo: QueenFi703/amazon-iap-kotlin)
scripts/gh-actions-run.sh 23652704571

# Specify a different repository
scripts/gh-actions-run.sh -R QueenFi703/amazon-iap-kotlin 23652704571

# Print full logs
scripts/gh-actions-run.sh --log 23652704571

# Print JSON status
scripts/gh-actions-run.sh --json 23652704571

# Combine flags
scripts/gh-actions-run.sh -R QueenFi703/amazon-iap-kotlin --json --log 23652704571
```

### Script flags

| Flag | Description |
|------|-------------|
| `-R owner/repo` / `--repo owner/repo` | Target repository (default: `QueenFi703/amazon-iap-kotlin`) |
| `--log` | Print the full run logs |
| `--json` | Print run status, conclusion, branch, SHA, and URL as JSON |
| `-h` / `--help` | Show help |

---

## 4. Equivalent raw `gh` commands

The script wraps the following standard `gh` invocations — you can also run them directly:

```bash
# Run summary
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin

# Full logs
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin --log

# JSON status
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin \
  --json status,conclusion,event,headBranch,headSha,createdAt,updatedAt,url,name

# Per-job JSON summary
gh run view 23652704571 -R QueenFi703/amazon-iap-kotlin \
  --json jobs -q '.jobs[] | {name, status, conclusion, url}'
```

---

## 5. List recent runs

```bash
# Last 10 runs on main
gh run list -R QueenFi703/amazon-iap-kotlin --branch main --limit 10

# Filter by workflow name
gh run list -R QueenFi703/amazon-iap-kotlin \
  --workflow "AWS Gradle Pipeline (S3 + CodeArtifact + Device Farm)" \
  --branch main --limit 5
```

---

## 6. Re-run failed jobs

```bash
# Re-run only the failed jobs of a specific run
gh run rerun 23652704571 -R QueenFi703/amazon-iap-kotlin --failed

# Re-run the entire run
gh run rerun 23652704571 -R QueenFi703/amazon-iap-kotlin
```

---

## 7. Trigger a new run (workflow_dispatch)

```bash
gh workflow run "AWS Gradle Pipeline (S3 + CodeArtifact + Device Farm)" \
  -R QueenFi703/amazon-iap-kotlin \
  --ref main
```

> **Note:** The workflow must have `on: workflow_dispatch` enabled for this to work.

---

## 8. Watch a run in real time

```bash
gh run watch 23652704571 -R QueenFi703/amazon-iap-kotlin
```
