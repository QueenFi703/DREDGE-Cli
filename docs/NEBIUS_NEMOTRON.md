# DREDGE + Nebius + NVIDIA Nemotron

This integration adds NVIDIA Nemotron reasoning to DREDGE through the Nebius Token Factory OpenAI-compatible API.

## Architecture

```text
DREDGE API -> Nebius Token Factory -> NVIDIA Nemotron 3 Super -> DREDGE reasoning
```

The DREDGE core remains provider-agnostic. This is an additional reasoning path and does not replace the existing provider chain.

## Environment

Set these variables only in the deployment environment; never commit the API key.

```bash
NEBIUS_API_KEY=<your-token-factory-api-key>
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
NEBIUS_MODEL=nvidia/nemotron-3-super-120b-a12b
```

## Smoke test

```bash
python -m pytest tests/test_nebius.py -v
curl http://127.0.0.1:8000/api/dredge/nebius/status
```

## Nebius deployment

Deploy DREDGE as a containerized application in Nebius AI Cloud or use Nebius Token Factory directly for inference. Configure the environment variables above in the deployment environment and keep the API key out of source control.

## Hackathon alignment

The DREDGE integration uses NVIDIA Nemotron as the reasoning model and Nebius Token Factory as the inference runtime. The application-specific layer is the DREDGE context/reasoning pipeline exposed through `/api/dredge/nebius/reason`.
