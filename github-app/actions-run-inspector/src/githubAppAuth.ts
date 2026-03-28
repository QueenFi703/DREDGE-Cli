import { createAppAuth } from "@octokit/auth-app";
import { Octokit } from "@octokit/rest";

export async function getOctokit() {
  const appId = process.env.GITHUB_APP_ID;
  const privateKey = process.env.GITHUB_APP_PRIVATE_KEY;
  const installationId = process.env.GITHUB_APP_INSTALLATION_ID;

  if (!appId || !privateKey || !installationId) {
    throw new Error(
      "Missing required environment variables: GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY, GITHUB_APP_INSTALLATION_ID"
    );
  }

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
