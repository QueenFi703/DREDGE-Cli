# DREDGE + Nebius + NVIDIA Nemotron

This integration adds NVIDIA Nemotron reasoning to DREDGE through the Nebius Token Factory OpenAI-compatible API.

## Hackathon architecture

```text
DREDGE API
   |
   +-- /api/dredge/nebius/reason
   |
   v
Nebius Token Factory
   |
   v
NVIDIA Nemotron 3 Super
```

The DREDGE core remains provider-agnostic. The Nebius adapter is an additional reasoning path and does not replace the existing provider chain.

## Environment

Set these variables in the deployment environment; never commit the API key.

```bash
NEBIUS_API_KEY=<your-token-factory-api-key>
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
NEBIUS_MODEL=nvidia/nemotron-3-super-120b-a12b
```

`NEBIUS_MODEL` can be changed to another NVIDIA open model available in Token Factory.

## Local smoke test

```bash
export NEBIUS_API_KEY="..."
export NEBIUS_MODEL="nvidia/nemotron-3-super-120b-a12b"
python -m pytest tests/test_nebius.py -v
```

With the Flask API running:

```bash
curl http://127.0.0.1:8000/api/dredge/nebius/status

curl -X POST http://127.0.0.1:8000/api/dredge/nebius/reason \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Distill this insight into the next best action.","context":"DREDGE memory"}'
```

## Nebius AI Cloud / console deployment

For a Nebius AI Cloud Serverless AI endpoint, use the existing repository Docker image and expose the application port used by the image. Configure the environment variables above in the endpoint's **Environment variables** section and enable token authentication for a production endpoint.

Nebius documents Serverless AI endpoints under **AI Services → Endpoints**. The service supports containerized workloads and public endpoints, and endpoint logs/metrics are available in the console.

For the hackathon, the project can alternatively make runtime inference calls to Nebius Token Factory; the hackathon rules explicitly accept a runtime call to the Token Factory inference API or deployment/run on Nebius AI Cloud.

## Hackathon compliance

The Nebius x NVIDIA Global AI Hackathon requires a working application running on Nebius Token Factory or Nebius AI Cloud and using at least one NVIDIA open source model. DREDGE's new reasoning endpoint makes that dependency explicit and testable.

The submission README should call out:

- NVIDIA Nemotron as the reasoning model.
- Nebius Token Factory as the inference runtime.
- `/api/dredge/nebius/reason` as the integration surface.
- The DREDGE memory/context pipeline as the application-specific layer.
