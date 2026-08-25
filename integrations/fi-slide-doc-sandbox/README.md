# Fi Slide & Document Sandbox

A GPT-native presentation/document studio integrated into DREDGE.

## Integration map

- **DREDGE:** optional advisory context through the MCP `unified_inference` operation. The adapter accepts either `DREDGE_MCP_URL` or the original `DREDGE_URL` environment variable.
- **OpenAI:** structured artifact generation through the Responses API.
- **PPTX/DOCX:** `pptxgenjs` and `jszip` are real npm dependencies in this integration; the original self-contained build vendored these runtimes.
- **Thresh:** GitHub Actions workflow analysis is run after the sandbox smoke test. The Thresh action is pinned to commit `5f047a931b266eda5648851b3aa0f507b0b048a3` and runs in report-only mode (`commit-fixes: false`).
- **GitHub Actions:** `actions/checkout@v4` and `actions/setup-node@v4` provide the Node CI path.
- **Hugging Face:** kept as a provider/model-extension boundary rather than a hard runtime dependency. This avoids coupling the artifact schema to a second inference API while leaving room for an optional HF inference provider later.

## Run

```bash
npm install
OPENAI_API_KEY=... DREDGE_MCP_URL=http://localhost:3002 npm start
```

Open `http://localhost:8787`.

## Health check

`GET /api/health` verifies that the Node server is alive without requiring an OpenAI key.

## DREDGE behavior

When `useDredge` is enabled, the server sends the user's brief to DREDGE through `/mcp` using `unified_inference`. DREDGE is advisory: if it is unavailable, ordinary generation still proceeds.

The original sandbox had a configuration mismatch (`DREDGE_MCP_URL` in `.env.example` but `DREDGE_URL` in code); this integration accepts both names.
