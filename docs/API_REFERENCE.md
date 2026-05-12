# DREDGE API Reference

## DREDGE API v1

Base URL: `https://api.dredge.fi/v1`

Auth: `Authorization: Bearer drg_live_xxxxx`

### Core loop

`scan → decide → fix → PR → bill`

## Endpoints

### Scan

- `POST /scan` — start repo analysis.
- `GET /scan/{scan_id}` — get scan status.

### Issues

- `GET /issues/{scan_id}` — list findings from a completed scan.

### Fix

- `POST /fix` — generate a patch for an issue.
- `GET /fix/{fix_id}` — fetch fix metadata.

### Pull requests

- `POST /pulls/create` — open a GitHub PR from a generated fix.

### Thresh Agent

- `POST /agent/evaluate` — decide priority, whether to fix, and auto-merge safety.
- `POST /agent/autopilot` — enable autonomous maintenance rules for a repo.

### Billing and usage

- `GET /usage` — usage and plan limits.

### API key management

- `POST /auth/key` — create an API key.

## Webhooks

- `scan.completed`
- `fix.generated`
- `pull.created`
- `risk.detected`

## Recommended MVP stack

- API: FastAPI
- Queue: Redis + Celery
- Database: Postgres
- Auth: JWT + API keys
- Hosting: Railway (or Fly.io)
- Containerization: Docker Compose
