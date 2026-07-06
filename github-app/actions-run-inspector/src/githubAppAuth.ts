import { createAppAuth } from "@octokit/auth-app";
import { Octokit } from "@octokit/rest";

export async function getOctokit(): Promise<Octokit> {
  const appId = process.env.GITHUB_APP_ID;
  const privateKey = process.env.GITHUB_APP_PRIVATE_KEY;
  const installationId = process.env.GITHUB_APP_INSTALLATION_ID;

  // When all three GitHub App credentials are present, authenticate as the
  // installed app (required for private repos and write operations).
  if (appId && privateKey && installationId) {
    const auth = createAppAuth({
      appId,
      privateKey,
      installationId
    });

    const installationAuth = await auth({ type: "installation" });

    return new Octokit({
      auth: installationAuth.token
    });
  }

  // Fall back to GITHUB_TOKEN for public-repo read access (e.g. the built-in
  // Actions token is sufficient to inspect QueenFi703/amazon-iap-kotlin).
  const githubToken = process.env.GITHUB_TOKEN;
  if (githubToken) {
    return new Octokit({ auth: githubToken });
  }

  throw new Error(
    "No authentication configured. Set GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY + " +
    "GITHUB_APP_INSTALLATION_ID for GitHub App auth, or set GITHUB_TOKEN for " +
    "public-repo access."
  );
}
