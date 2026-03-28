export function parseRunUrl(runUrl: string) {
  const match = runUrl.match(
    /github\.com\/([^/]+)\/([^/]+)\/actions\/runs\/(\d+)/
  );

  if (!match) {
    throw new Error("Invalid run_url format");
  }

  return {
    owner: match[1],
    repo: match[2],
    run_id: Number(match[3])
  };
}
