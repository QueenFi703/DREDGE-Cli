import fs from "fs";
import { Octokit } from "@octokit/rest";
import { getOctokit } from "./githubAppAuth.js";
import { parseRunUrl } from "./parseRunUrl.js";
import { summarize } from "./summarize.js";

async function getLatestRunId(
  octokit: Octokit,
  owner: string,
  repo: string,
  workflowId?: string
): Promise<number> {
  if (workflowId) {
    const { data } = await octokit.actions.listWorkflowRuns({
      owner,
      repo,
      workflow_id: workflowId,
      per_page: 1
    });
    if (data.workflow_runs.length === 0) {
      throw new Error(
        `No workflow runs found for ${owner}/${repo} workflow "${workflowId}"`
      );
    }
    return data.workflow_runs[0].id;
  }

  const { data } = await octokit.actions.listWorkflowRunsForRepo({
    owner,
    repo,
    per_page: 1
  });
  if (data.workflow_runs.length === 0) {
    throw new Error(`No workflow runs found for ${owner}/${repo}`);
  }
  return data.workflow_runs[0].id;
}

async function main() {
  const {
    owner,
    repo,
    run_id,
    run_url,
    workflow_id,
    post_comment,
    comment_url
  } = process.env;

  const octokit = await getOctokit();

  let resolved: { owner: string; repo: string; run_id: number };

  if (run_url) {
    resolved = parseRunUrl(run_url);
  } else if (run_id && !isNaN(Number(run_id))) {
    if (!owner || !repo) {
      throw new Error(
        "owner and repo environment variables are required when using run_id"
      );
    }
    resolved = { owner, repo, run_id: Number(run_id) };
  } else if (owner && repo) {
    // Latest-run mode: fetch the most recent run for the given repo
    const latestRunId = await getLatestRunId(
      octokit,
      owner,
      repo,
      workflow_id
    );
    resolved = { owner, repo, run_id: latestRunId };
  } else {
    throw new Error(
      "Provide run_url, run_id (with owner+repo), or owner+repo to fetch the latest run"
    );
  }

  const run = await octokit.actions.getWorkflowRun({
    owner: resolved.owner,
    repo: resolved.repo,
    run_id: resolved.run_id
  });

  const jobs = await octokit.actions.listJobsForWorkflowRun({
    owner: resolved.owner,
    repo: resolved.repo,
    run_id: resolved.run_id
  });

  const result = summarize(run.data, jobs.data.jobs);

  const json = JSON.stringify(result, null, 2);

  // Save artifact
  fs.writeFileSync("result.json", json);

  // Step summary (only available inside a GitHub Actions runner)
  const stepSummaryPath = process.env.GITHUB_STEP_SUMMARY;
  if (stepSummaryPath) {
    fs.appendFileSync(
      stepSummaryPath,
      `## Actions Run Inspector\n\n**${resolved.owner}/${resolved.repo}** · run \`${resolved.run_id}\`\n\n\`\`\`json\n${json}\n\`\`\`\n`
    );
  }

  // Optional comment
  if (post_comment === "true" && comment_url) {
    await octokit.request(`POST ${comment_url}`, {
      body: `### 🤖 Actions Run Inspector\n\n**${resolved.owner}/${resolved.repo}** · run \`${resolved.run_id}\`\n\n\`\`\`json\n${json}\n\`\`\``
    });
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
