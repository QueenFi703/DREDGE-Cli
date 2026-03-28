import fs from "fs";
import { getOctokit } from "./githubAppAuth.js";
import { parseRunUrl } from "./parseRunUrl.js";
import { summarize } from "./summarize.js";

async function main() {
  const {
    owner,
    repo,
    run_id,
    run_url,
    post_comment,
    comment_url
  } = process.env;

  let resolved = { owner, repo, run_id: Number(run_id) };

  if (!run_url && isNaN(Number(run_id))) {
    throw new Error("run_id must be a valid number when run_url is not provided");
  }

  if (run_url) {
    resolved = parseRunUrl(run_url);
  }

  const octokit = await getOctokit();

  const run = await octokit.actions.getWorkflowRun({
    owner: resolved.owner!,
    repo: resolved.repo!,
    run_id: resolved.run_id!
  });

  const jobs = await octokit.actions.listJobsForWorkflowRun({
    owner: resolved.owner!,
    repo: resolved.repo!,
    run_id: resolved.run_id!
  });

  const result = summarize(run.data, jobs.data.jobs);

  const json = JSON.stringify(result, null, 2);

  // Save artifact
  fs.writeFileSync("result.json", json);

  // Step summary
  fs.appendFileSync(
    process.env.GITHUB_STEP_SUMMARY!,
    `## Actions Run Inspector\n\n\`\`\`json\n${json}\n\`\`\`\n`
  );

  // Optional comment
  if (post_comment === "true" && comment_url) {
    await octokit.request(`POST ${comment_url}`, {
      body: `### 🤖 Actions Run Inspector\n\n\`\`\`json\n${json}\n\`\`\``
    });
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
