# GitHub App – Actions Run Inspector

A serverless, GitHub-hosted tool that authenticates as a GitHub App installation
and queries GitHub Actions workflow run status and jobs.

---

## Table of Contents

1. [Overview](#overview)
2. [Create the GitHub App](#create-the-github-app)
3. [Generate and store the private key](#generate-and-store-the-private-key)
4. [Install the app and get the installation ID](#install-the-app-and-get-the-installation-id)
5. [Configure repository secrets](#configure-repository-secrets)
6. [Run locally](#run-locally)
7. [Run via workflow\_dispatch](#run-via-workflow_dispatch)
8. [Output format](#output-format)
9. [Replicate into QueenFi703/amazon-iap-kotlin](#replicate-into-queenfi703amazon-iap-kotlin)

---

## Overview

The inspector is a small Node.js 24 / TypeScript package in `github-app/`.  
It authenticates using three GitHub App credentials and calls:

- `GET /repos/{owner}/{repo}/actions/runs/{run_id}` – run status
- `GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs` – jobs list

Results are written as JSON to stdout and (when running via Actions) uploaded
as a workflow artifact.

---

## Create the GitHub App

1. Go to **Settings → Developer settings → GitHub Apps** for your account:  
   `https://github.com/settings/apps/new`

2. Fill in:
   | Field | Value |
   |---|---|
   | **GitHub App name** | `DREDGE Actions Inspector` (or any name) |
   | **Homepage URL** | `https://github.com/QueenFi703/DREDGE-Cli` |
   | **Webhook → Active** | **uncheck** (no webhook needed) |

3. Under **Repository permissions**, set:
   | Permission | Level |
   |---|---|
   | **Actions** | Read-only |
   | **Metadata** | Read-only (mandatory) |

4. Under **Where can this GitHub App be installed**, choose:
   - **Only on this account** (for personal use), or
   - **Any account** (if you want to inspect runs in other orgs).

5. Click **Create GitHub App**.  
   Note the **App ID** shown on the settings page (a small integer, e.g. `12345`).

---

## Generate and store the private key

1. On the App settings page, scroll to **Private keys** and click
   **Generate a private key**.
2. A `.pem` file is downloaded automatically.
3. Open the file and copy its entire contents, including the
   `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----` lines.

> **Security** – Never commit the `.pem` file. Store it only in GitHub Actions secrets.

---

## Install the app and get the installation ID

1. On the App settings page, click **Install App** in the left sidebar.
2. Choose the account/organization and select the repositories you want to
   inspect (e.g. `DREDGE-Cli` and `amazon-iap-kotlin`).
3. After installing, the browser URL changes to:
   ```
   https://github.com/settings/installations/<installation_id>
   ```
   Copy that numeric **installation ID**.

Alternatively, retrieve it via the API after authenticating as the app:
```bash
curl -H "Authorization: Bearer <app_jwt>" \
     https://api.github.com/app/installations
```

---

## Configure repository secrets

In `QueenFi703/DREDGE-Cli` (and any other repo where you want the workflow to
run), go to **Settings → Secrets and variables → Actions** and add:

| Secret name | Value |
|---|---|
| `GITHUB_APP_ID` | Numeric App ID from the App settings page |
| `GITHUB_APP_PRIVATE_KEY` | Full PEM contents of the downloaded `.pem` file |
| `GITHUB_APP_INSTALLATION_ID` | Numeric installation ID from the install URL |

> Multi-line PEM values are stored verbatim by GitHub. The code automatically
> handles both `\n`-escaped and literal-newline formats.

---

## Run locally

### Prerequisites

- Node.js ≥ 24 (`node --version`)
- The three secrets exported as environment variables

```bash
export GITHUB_APP_ID=12345
export GITHUB_APP_PRIVATE_KEY="$(cat /path/to/private-key.pem)"
export GITHUB_APP_INSTALLATION_ID=67890
```

### Install dependencies

```bash
cd github-app
npm install
```

### Inspect a run by URL

```bash
npm run inspect -- --run-url https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571
```

### Inspect a run by owner/repo/run_id

```bash
npm run inspect -- --owner QueenFi703 --repo amazon-iap-kotlin --run-id 23652704571
```

### Use the CLI bin directly

```bash
node bin/actions-run-inspect --run-url https://github.com/QueenFi703/DREDGE-Cli/actions/runs/12345
```

### Output

The command writes a JSON object to **stdout**. Pipe it to `jq` for pretty output:

```bash
npm run inspect -- --run-url <url> | jq .
```

---

## Run via workflow_dispatch

1. Go to **Actions → Actions Run Inspector** in `QueenFi703/DREDGE-Cli`.
2. Click **Run workflow** and fill in:
   - **run_url** – full URL of the run to inspect (recommended), **or**
   - **owner** + **repo** + **run_id** – explicit coordinates
   - **include_logs** – `true` to include step-level detail
3. Click **Run workflow**.
4. After the job finishes, download the `run-inspection-<run_id>` artifact for
   the full JSON output.
5. The **Summary** tab shows a human-readable table with status, conclusion, and
   job counts.

---

## Output format

```json
{
  "run": {
    "id": 23652704571,
    "name": "CI",
    "status": "completed",
    "conclusion": "success",
    "html_url": "https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571",
    "head_branch": "main",
    "head_sha": "abc123...",
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:05:00Z",
    "run_started_at": "2025-01-01T00:00:01Z",
    "run_attempt": 1,
    "workflow_id": 999
  },
  "jobs": [
    {
      "id": 111,
      "name": "build",
      "status": "completed",
      "conclusion": "success",
      "started_at": "2025-01-01T00:00:05Z",
      "completed_at": "2025-01-01T00:04:55Z",
      "html_url": "https://github.com/...",
      "steps": [
        {
          "name": "Checkout",
          "status": "completed",
          "conclusion": "success",
          "number": 1
        }
      ]
    }
  ],
  "meta": {
    "owner": "QueenFi703",
    "repo": "amazon-iap-kotlin",
    "runId": 23652704571,
    "fetchedAt": "2025-01-01T00:06:00Z"
  }
}
```

---

## Replicate into QueenFi703/amazon-iap-kotlin

To run the same inspector from `amazon-iap-kotlin`:

1. **Copy the package folder** into that repo:
   ```bash
   cp -r github-app /path/to/amazon-iap-kotlin/
   ```

2. **Copy the workflow**:
   ```bash
   cp .github/workflows/actions-run-inspector.yml \
      /path/to/amazon-iap-kotlin/.github/workflows/
   ```

3. **Add the same three secrets** (`GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`,
   `GITHUB_APP_INSTALLATION_ID`) to `QueenFi703/amazon-iap-kotlin` via
   Settings → Secrets and variables → Actions.  
   Use the **same App** if it was installed on both repos; the installation ID
   remains the same.

4. **Update `.gitignore`** (if not already present):
   ```
   github-app/node_modules/
   github-app/dist/
   ```

5. Commit and push. The **Actions Run Inspector** workflow will now be available
   under the **Actions** tab of `amazon-iap-kotlin`.
