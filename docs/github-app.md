# GitHub Actions Inspector — GitHub App

A minimal Node.js + TypeScript HTTP service that uses a **GitHub App** installation token to query GitHub Actions workflow run status and job conclusions for any repository the app is installed on.

---

## Table of Contents

1. [Overview](#overview)
2. [Creating a GitHub App](#creating-a-github-app)
3. [Required Permissions](#required-permissions)
4. [Installing the App on a Repository](#installing-the-app-on-a-repository)
5. [Environment Variables](#environment-variables)
6. [Running Locally](#running-locally)
7. [API Reference](#api-reference)
8. [Calling the Endpoint with curl](#calling-the-endpoint-with-curl)
9. [Demo Script](#demo-script)
10. [Deployment Notes](#deployment-notes)

---

## Overview

The **GitHub Actions Inspector** exposes two HTTP endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/actions/run` | Fetch run summary + jobs list |

Authentication is handled entirely through a GitHub App installation token — **no Personal Access Token (PAT) is used**.

Core auth flow: **Private key → JWT → Installation access token → GitHub API**.

---

### Token Flow (ChatGPT/Copilot server integration)

If you are integrating from a ChatGPT/Copilot-hosted service, the high-level OAuth flow is:

`ChatGPT/Copilot server → GitHub API → refresh token`

- The ChatGPT/Copilot server stores the refresh token securely.
- It exchanges the refresh token at GitHub endpoints to obtain a short-lived access token.
- The access token is then used for GitHub API calls; when it expires, the server repeats the refresh-token exchange.

> Note: This OAuth refresh-token flow is separate from GitHub App installation-token auth used by this inspector.
GitHub App auth uses a signed JWT + installation token exchange instead of OAuth refresh tokens.

---

## Creating a GitHub App

1. Go to **GitHub → Settings → Developer settings → GitHub Apps → New GitHub App**
   (or navigate to `https://github.com/settings/apps/new`).

2. Fill in the required fields:

   | Field | Value |
   |-------|-------|
   | **GitHub App name** | e.g. `my-actions-inspector` |
   | **Homepage URL** | `https://github.com/QueenFi703` |
   | **Webhook** | Uncheck *Active* (not needed for this app) |

3. Under **Repository permissions**, set:

   | Permission | Access |
   |------------|--------|
   | **Actions** | Read-only |
   | **Metadata** | Read-only *(required by GitHub automatically)* |

   > `Contents` is optional — only needed if you also want to read workflow YAML files.

4. Click **Create GitHub App**.

5. Note the **App ID** shown at the top of the app settings page.

6. Scroll down to **Private keys** and click **Generate a private key**.  
   A `.pem` file will be downloaded — keep it secure.

### Recommended manifest values (Dredge)

When you manifest the app for Dredge, use:

| Manifest field | Value |
|---|---|
| **Homepage URL** | `https://github.com/QueenFi703` |
| **Webhook** | Disabled unless you explicitly need events |
| **Repository access** | Only selected repositories |

This keeps the integration aligned with the GitHub App model: app private key → short-lived JWT (max 10 minutes) → short-lived installation token (about 1 hour) → GitHub API calls.

---

## Required Permissions

| Permission | Level | Reason |
|------------|-------|--------|
| Actions | Read | List/get workflow runs and jobs |
| Metadata | Read | Required by GitHub for all apps |

---

## Installing the App on a Repository

1. In the GitHub App settings page, click **Install App**.
2. Choose the account/organisation.
3. Select **Only select repositories** and pick the target repo(s).
4. Click **Install**.
5. After installation, note the **Installation ID** from the URL:  
   `https://github.com/settings/installations/<INSTALLATION_ID>`

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_APP_ID` | ✅ | The numeric App ID from the GitHub App settings page |
| `GITHUB_APP_INSTALLATION_ID` | ✅ | The numeric Installation ID from the install URL |
| `GITHUB_APP_PRIVATE_KEY` | ✅ | Full PEM content of the private key. Supports `\n` escape sequences or literal newlines |
| `PORT` | ❌ | HTTP port (default: `3003`) |

> **Security**: Never commit `.env` or the private key file. Both are excluded by `.gitignore`.

### Multi-line private key in an `.env` file

```dotenv
GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA...
...
-----END RSA PRIVATE KEY-----"
```

Or as an inline value with `\n` escapes (useful in CI secrets):

```
GITHUB_APP_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----\nMIIEo...
```

---

## Running Locally

```bash
# 1. Enter the app directory
cd github-app

# 2. Install dependencies
npm install

# 3. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your real App ID, Installation ID, and private key

# 4. Start the development server (ts-node, no build step required)
npm run dev

# 5. Or build first, then start
npm run build
npm start
```

The server starts on `http://localhost:3003` by default.

---

## API Reference

### `GET /health`

Returns the server liveness status.

**Response:**

```json
{
  "status": "ok",
  "app_configured": true,
  "timestamp": "2026-03-28T14:00:00.000Z"
}
```

`app_configured` is `false` when the required environment variables are missing.

---

### `GET /actions/run`

Returns a workflow run summary and jobs list.

**Query parameters (option A — explicit):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `owner` | string | Repository owner (user or org) |
| `repo` | string | Repository name |
| `run_id` | number | Workflow run ID |

**Query parameters (option B — run URL):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_url` | string | Full GitHub Actions run URL |

**Response (200):**

```json
{
  "run": {
    "id": 23652704571,
    "name": "AWS Gradle Pipeline",
    "status": "completed",
    "conclusion": "failure",
    "html_url": "https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571",
    "head_branch": "main",
    "head_sha": "abc123...",
    "event": "push",
    "created_at": "2026-03-28T10:00:00Z",
    "updated_at": "2026-03-28T10:05:00Z"
  },
  "jobs": [
    {
      "id": 987654,
      "name": "build",
      "status": "completed",
      "conclusion": "failure",
      "started_at": "2026-03-28T10:01:00Z",
      "completed_at": "2026-03-28T10:04:00Z",
      "html_url": "https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571/jobs/987654"
    }
  ]
}
```

**Error responses:**

| Status | Reason |
|--------|--------|
| `400` | Missing or invalid parameters |
| `503` | GitHub App not configured (env vars missing) |
| `500` | Upstream GitHub API error |

---

## Calling the Endpoint with curl

### Using explicit owner/repo/run_id

```bash
curl -s "http://localhost:3003/actions/run?owner=QueenFi703&repo=amazon-iap-kotlin&run_id=23652704571" \
  | python3 -m json.tool
```

### Using a run URL

```bash
curl -s "http://localhost:3003/actions/run?run_url=https%3A%2F%2Fgithub.com%2FQueenFi703%2Famazon-iap-kotlin%2Factions%2Fruns%2F23652704571" \
  | python3 -m json.tool
```

### Health check

```bash
curl -s http://localhost:3003/health | python3 -m json.tool
```

---

## Demo Script

The included `scripts/check-run.sh` wraps the curl call for convenience:

```bash
# Make executable (first time only)
chmod +x scripts/check-run.sh

# Query by run URL
./scripts/check-run.sh \
  "https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571"

# Query by individual params
./scripts/check-run.sh \
  --owner QueenFi703 \
  --repo amazon-iap-kotlin \
  --run-id 23652704571

# Point at a remote inspector instance
INSPECTOR_URL=https://my-inspector.example.com \
  ./scripts/check-run.sh "https://github.com/..."
```

---

## Deployment Notes

- Set the three `GITHUB_APP_*` environment variables in your hosting platform's secrets manager.
- The private key value should be the raw PEM with `\n` replacing literal newlines (most CI/hosting environments handle this automatically).
- The service has no persistent state; it can be run as a single stateless container or serverless function.
- Default port is `3003` to avoid conflicts with the existing DREDGE servers on `3001`/`3002`.
