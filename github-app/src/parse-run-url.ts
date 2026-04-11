/**
 * parse-run-url.ts
 *
 * Utilities for parsing a GitHub Actions run URL or extracting owner/repo/run_id
 * from various input forms.
 */

export interface RunCoordinates {
  owner: string;
  repo: string;
  runId: number;
}

/**
 * Parse a GitHub Actions run URL of the form:
 *   https://github.com/{owner}/{repo}/actions/runs/{run_id}
 *
 * Returns null when the URL does not match the expected pattern.
 */
export function parseRunUrl(url: string): RunCoordinates | null {
  const pattern =
    /^https:\/\/github\.com\/([^/]+)\/([^/]+)\/actions\/runs\/(\d+)(?:[/?#]|$)/;
  const match = pattern.exec(url);
  if (!match) {
    return null;
  }
  return {
    owner: match[1],
    repo: match[2],
    runId: parseInt(match[3], 10),
  };
}

/**
 * Resolve run coordinates from the various ways a caller can specify a run:
 *  - `run_url`  – full GitHub Actions run URL (takes precedence)
 *  - `owner` + `repo` + `run_id`  – explicit fields
 *
 * Throws if neither form provides enough information.
 */
export function resolveCoordinates(opts: {
  runUrl?: string;
  owner?: string;
  repo?: string;
  runId?: string | number;
}): RunCoordinates {
  if (opts.runUrl) {
    const parsed = parseRunUrl(opts.runUrl);
    if (!parsed) {
      throw new Error(
        `Invalid run URL: "${opts.runUrl}". ` +
          "Expected format: https://github.com/{owner}/{repo}/actions/runs/{run_id}"
      );
    }
    return parsed;
  }

  if (!opts.owner || !opts.repo || opts.runId === undefined) {
    throw new Error(
      "Must provide either --run-url or all of --owner, --repo, --run-id"
    );
  }

  const runId =
    typeof opts.runId === "number"
      ? opts.runId
      : parseInt(opts.runId as string, 10);

  if (isNaN(runId) || runId <= 0) {
    throw new Error(`Invalid run_id: "${opts.runId}"`);
  }

  return { owner: opts.owner, repo: opts.repo, runId };
}
