/**
 * index.ts – GitHub App Actions Run Inspector
 *
 * Authenticates as a GitHub App installation and fetches workflow run details
 * and job list for a given run. Outputs a JSON summary to stdout.
 *
 * Required environment variables:
 *   GITHUB_APP_ID              – numeric App ID
 *   GITHUB_APP_PRIVATE_KEY     – PEM-encoded private key (with or without newlines escaped)
 *   GITHUB_APP_INSTALLATION_ID – numeric installation ID
 *
 * Usage:
 *   npx tsx src/index.ts --run-url https://github.com/owner/repo/actions/runs/12345
 *   npx tsx src/index.ts --owner owner --repo repo --run-id 12345
 *   npx tsx src/index.ts --run-url ... --include-logs
 */

import { createAppAuth } from "@octokit/auth-app";
import { Octokit } from "@octokit/rest";
import { resolveCoordinates } from "./parse-run-url.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RunSummary {
  run: {
    id: number;
    name: string | null | undefined;
    status: string | null;
    conclusion: string | null;
    html_url: string;
    head_branch: string | null;
    head_sha: string;
    created_at: string | null;
    updated_at: string | null;
    run_started_at?: string | null;
    run_attempt?: number | null;
    workflow_id: number;
    workflow_url?: string;
  };
  jobs: Array<{
    id: number;
    name: string;
    status: string;
    conclusion: string | null;
    started_at: string | null;
    completed_at: string | null;
    html_url: string | null;
    steps?: Array<{
      name: string;
      status: string;
      conclusion: string | null;
      number: number;
    }>;
  }>;
  meta: {
    owner: string;
    repo: string;
    runId: number;
    fetchedAt: string;
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseArgs(argv: string[]): {
  runUrl?: string;
  owner?: string;
  repo?: string;
  runId?: string;
  includeLogs: boolean;
} {
  const args = argv.slice(2);
  const result: {
    runUrl?: string;
    owner?: string;
    repo?: string;
    runId?: string;
    includeLogs: boolean;
  } = { includeLogs: false };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case "--run-url":
        result.runUrl = args[++i];
        break;
      case "--owner":
        result.owner = args[++i];
        break;
      case "--repo":
        result.repo = args[++i];
        break;
      case "--run-id":
        result.runId = args[++i];
        break;
      case "--include-logs":
        result.includeLogs = true;
        break;
      default:
        // Support --flag=value syntax
        if (arg.startsWith("--run-url=")) {
          result.runUrl = arg.slice("--run-url=".length);
        } else if (arg.startsWith("--owner=")) {
          result.owner = arg.slice("--owner=".length);
        } else if (arg.startsWith("--repo=")) {
          result.repo = arg.slice("--repo=".length);
        } else if (arg.startsWith("--run-id=")) {
          result.runId = arg.slice("--run-id=".length);
        }
    }
  }
  return result;
}

function normalizePrivateKey(raw: string): string {
  // GitHub Actions secrets collapse newlines; restore them if needed.
  if (raw.includes("\\n")) {
    return raw.replace(/\\n/g, "\n");
  }
  return raw;
}

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(
      `Missing required environment variable: ${name}. ` +
        "Set GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY, and GITHUB_APP_INSTALLATION_ID."
    );
  }
  return value;
}

// ---------------------------------------------------------------------------
// Core inspection logic (exported for programmatic use)
// ---------------------------------------------------------------------------

export async function inspectRun(opts: {
  owner: string;
  repo: string;
  runId: number;
  appId: number;
  privateKey: string;
  installationId: number;
  includeSteps?: boolean;
}): Promise<RunSummary> {
  const octokit = new Octokit({
    authStrategy: createAppAuth,
    auth: {
      appId: opts.appId,
      privateKey: opts.privateKey,
      installationId: opts.installationId,
    },
  });

  // Fetch the workflow run
  const { data: run } = await octokit.rest.actions.getWorkflowRun({
    owner: opts.owner,
    repo: opts.repo,
    run_id: opts.runId,
  });

  // Fetch jobs (paginated – collect all pages)
  const jobs: RunSummary["jobs"] = [];
  for await (const page of octokit.paginate.iterator(
    octokit.rest.actions.listJobsForWorkflowRun,
    {
      owner: opts.owner,
      repo: opts.repo,
      run_id: opts.runId,
      per_page: 100,
    }
  )) {
    for (const job of page.data) {
      jobs.push({
        id: job.id,
        name: job.name,
        status: job.status,
        conclusion: job.conclusion ?? null,
        started_at: job.started_at ?? null,
        completed_at: job.completed_at ?? null,
        html_url: job.html_url ?? null,
        ...(opts.includeSteps && job.steps
          ? {
              steps: job.steps.map((s) => ({
                name: s.name,
                status: s.status,
                conclusion: s.conclusion ?? null,
                number: s.number,
              })),
            }
          : {}),
      });
    }
  }

  return {
    run: {
      id: run.id,
      name: run.name,
      status: run.status,
      conclusion: run.conclusion ?? null,
      html_url: run.html_url,
      head_branch: run.head_branch ?? null,
      head_sha: run.head_sha,
      created_at: run.created_at ?? null,
      updated_at: run.updated_at ?? null,
      run_started_at: run.run_started_at ?? null,
      run_attempt: run.run_attempt ?? null,
      workflow_id: run.workflow_id,
      workflow_url: run.workflow_url,
    },
    jobs,
    meta: {
      owner: opts.owner,
      repo: opts.repo,
      runId: opts.runId,
      fetchedAt: new Date().toISOString(),
    },
  };
}

// ---------------------------------------------------------------------------
// CLI entrypoint
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const cliArgs = parseArgs(process.argv);

  let coords: { owner: string; repo: string; runId: number };
  try {
    coords = resolveCoordinates({
      runUrl: cliArgs.runUrl,
      owner: cliArgs.owner,
      repo: cliArgs.repo,
      runId: cliArgs.runId,
    });
  } catch (err) {
    process.stderr.write(
      `Error: ${err instanceof Error ? err.message : String(err)}\n`
    );
    process.stderr.write(
      "Usage: actions-run-inspect --run-url <url>\n" +
        "   or: actions-run-inspect --owner <owner> --repo <repo> --run-id <id>\n"
    );
    process.exit(1);
  }

  let appId: number;
  let privateKey: string;
  let installationId: number;
  try {
    appId = parseInt(requireEnv("GITHUB_APP_ID"), 10);
    privateKey = normalizePrivateKey(requireEnv("GITHUB_APP_PRIVATE_KEY"));
    installationId = parseInt(requireEnv("GITHUB_APP_INSTALLATION_ID"), 10);
  } catch (err) {
    process.stderr.write(
      `Configuration error: ${err instanceof Error ? err.message : String(err)}\n`
    );
    process.exit(1);
  }

  try {
    const summary = await inspectRun({
      ...coords,
      appId,
      privateKey,
      installationId,
      includeSteps: true,
    });
    process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
  } catch (err) {
    process.stderr.write(
      `Inspection failed: ${err instanceof Error ? err.message : String(err)}\n`
    );
    process.exit(1);
  }
}

// Run when executed directly (ESM-compatible check)
if (
  process.argv[1] &&
  (process.argv[1].endsWith("/index.ts") ||
    process.argv[1].endsWith("/index.js") ||
    process.argv[1].endsWith("actions-run-inspect"))
) {
  main().catch((err) => {
    process.stderr.write(String(err) + "\n");
    process.exit(1);
  });
}
