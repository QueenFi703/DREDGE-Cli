import "dotenv/config";
import express, { Request, Response, NextFunction } from "express";
import { App } from "@octokit/app";

// ─── Config ──────────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.PORT ?? "3003", 10);
const APP_ID = process.env.GITHUB_APP_ID ?? "";
const INSTALLATION_ID = parseInt(
  process.env.GITHUB_APP_INSTALLATION_ID ?? "0",
  10
);
// Support literal \n in the env var as well as real newlines
const PRIVATE_KEY = (process.env.GITHUB_APP_PRIVATE_KEY ?? "").replace(
  /\\n/g,
  "\n"
);

// ─── GitHub App client ───────────────────────────────────────────────────────

function buildApp(): App | null {
  if (!APP_ID || !PRIVATE_KEY || !INSTALLATION_ID) {
    return null;
  }
  return new App({ appId: APP_ID, privateKey: PRIVATE_KEY });
}

let githubApp: App | null = buildApp();

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Parse a GitHub Actions run URL into { owner, repo, run_id }. */
export function parseRunUrl(
  url: string
): { owner: string; repo: string; run_id: number } | null {
  const match = url.match(
    /github\.com\/([^/]+)\/([^/]+)\/actions\/runs\/(\d+)/
  );
  if (!match) return null;
  return { owner: match[1], repo: match[2], run_id: parseInt(match[3], 10) };
}

// ─── Express app ─────────────────────────────────────────────────────────────

const app = express();
app.use(express.json());

// GET /health
app.get("/health", (_req: Request, res: Response) => {
  res.json({
    status: "ok",
    app_configured: githubApp !== null,
    timestamp: new Date().toISOString(),
  });
});

// GET /actions/run?owner=&repo=&run_id= (or ?run_url=)
app.get("/actions/run", async (req: Request, res: Response) => {
  try {
    if (!githubApp) {
      res.status(503).json({
        error:
          "GitHub App not configured. Set GITHUB_APP_ID, GITHUB_APP_INSTALLATION_ID, and GITHUB_APP_PRIVATE_KEY.",
      });
      return;
    }

    let owner: string | undefined;
    let repo: string | undefined;
    let run_id: number | undefined;

    if (req.query.run_url) {
      const parsed = parseRunUrl(String(req.query.run_url));
      if (!parsed) {
        res.status(400).json({
          error:
            "Invalid run_url. Expected format: https://github.com/{owner}/{repo}/actions/runs/{run_id}",
        });
        return;
      }
      ({ owner, repo, run_id } = parsed);
    } else {
      owner = req.query.owner ? String(req.query.owner) : undefined;
      repo = req.query.repo ? String(req.query.repo) : undefined;
      run_id = req.query.run_id ? parseInt(String(req.query.run_id), 10) : undefined;
    }

    if (!owner || !repo || !run_id || isNaN(run_id)) {
      res.status(400).json({
        error:
          "Missing required parameters. Provide owner, repo, and run_id — or run_url.",
      });
      return;
    }

    const octokit = await githubApp.getInstallationOctokit(INSTALLATION_ID);

    // Fetch run summary
    const { data: run } = await octokit.request(
      "GET /repos/{owner}/{repo}/actions/runs/{run_id}",
      { owner, repo, run_id }
    );

    // Fetch jobs list
    const { data: jobsData } = await octokit.request(
      "GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
      { owner, repo, run_id }
    );

    const jobs = jobsData.jobs.map((job) => ({
      id: job.id,
      name: job.name,
      status: job.status,
      conclusion: job.conclusion,
      started_at: job.started_at,
      completed_at: job.completed_at,
      html_url: job.html_url,
    }));

    res.json({
      run: {
        id: run.id,
        name: run.name,
        status: run.status,
        conclusion: run.conclusion,
        html_url: run.html_url,
        head_branch: run.head_branch,
        head_sha: run.head_sha,
        event: run.event,
        created_at: run.created_at,
        updated_at: run.updated_at,
      },
      jobs,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ error: message });
  }
});

// Generic error handler
// eslint-disable-next-line @typescript-eslint/no-unused-vars
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  res.status(500).json({ error: err.message });
});

// ─── Start ───────────────────────────────────────────────────────────────────

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`GitHub Actions Inspector listening on http://localhost:${PORT}`);
    console.log(`  GET /health`);
    console.log(`  GET /actions/run?owner=<owner>&repo=<repo>&run_id=<id>`);
    console.log(`  GET /actions/run?run_url=<github_actions_run_url>`);
  });
}

export default app;
