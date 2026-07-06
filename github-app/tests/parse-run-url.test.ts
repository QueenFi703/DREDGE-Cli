/**
 * Unit tests for run URL parsing utilities.
 */

import { parseRunUrl, resolveCoordinates } from "../src/parse-run-url.js";

describe("parseRunUrl", () => {
  test("parses a valid GitHub Actions run URL", () => {
    const url =
      "https://github.com/QueenFi703/DREDGE-Cli/actions/runs/12345678";
    const result = parseRunUrl(url);
    expect(result).not.toBeNull();
    expect(result!.owner).toBe("QueenFi703");
    expect(result!.repo).toBe("DREDGE-Cli");
    expect(result!.runId).toBe(12345678);
  });

  test("parses a URL that includes trailing path segments", () => {
    const url =
      "https://github.com/owner/my-repo/actions/runs/99999/jobs/111222";
    const result = parseRunUrl(url);
    expect(result).not.toBeNull();
    expect(result!.owner).toBe("owner");
    expect(result!.repo).toBe("my-repo");
    expect(result!.runId).toBe(99999);
  });

  test("returns null for an unrelated URL", () => {
    expect(parseRunUrl("https://github.com/owner/repo")).toBeNull();
  });

  test("returns null for a non-GitHub URL", () => {
    expect(
      parseRunUrl("https://example.com/owner/repo/actions/runs/1")
    ).toBeNull();
  });

  test("returns null for an empty string", () => {
    expect(parseRunUrl("")).toBeNull();
  });

  test("parses a URL with a real run id from amazon-iap-kotlin", () => {
    const url =
      "https://github.com/QueenFi703/amazon-iap-kotlin/actions/runs/23652704571";
    const result = parseRunUrl(url);
    expect(result).not.toBeNull();
    expect(result!.owner).toBe("QueenFi703");
    expect(result!.repo).toBe("amazon-iap-kotlin");
    expect(result!.runId).toBe(23652704571);
  });
});

describe("resolveCoordinates", () => {
  test("resolves from run_url", () => {
    const coords = resolveCoordinates({
      runUrl:
        "https://github.com/QueenFi703/DREDGE-Cli/actions/runs/12345678",
    });
    expect(coords).toEqual({
      owner: "QueenFi703",
      repo: "DREDGE-Cli",
      runId: 12345678,
    });
  });

  test("resolves from explicit owner/repo/run_id (string)", () => {
    const coords = resolveCoordinates({
      owner: "QueenFi703",
      repo: "DREDGE-Cli",
      runId: "12345678",
    });
    expect(coords).toEqual({
      owner: "QueenFi703",
      repo: "DREDGE-Cli",
      runId: 12345678,
    });
  });

  test("resolves from explicit owner/repo/run_id (number)", () => {
    const coords = resolveCoordinates({
      owner: "QueenFi703",
      repo: "DREDGE-Cli",
      runId: 12345678,
    });
    expect(coords).toEqual({
      owner: "QueenFi703",
      repo: "DREDGE-Cli",
      runId: 12345678,
    });
  });

  test("run_url takes precedence over explicit fields", () => {
    const coords = resolveCoordinates({
      runUrl:
        "https://github.com/QueenFi703/DREDGE-Cli/actions/runs/12345678",
      owner: "other-owner",
      repo: "other-repo",
      runId: "999",
    });
    expect(coords.owner).toBe("QueenFi703");
    expect(coords.repo).toBe("DREDGE-Cli");
    expect(coords.runId).toBe(12345678);
  });

  test("throws for an invalid run_url", () => {
    expect(() =>
      resolveCoordinates({ runUrl: "https://github.com/owner/repo" })
    ).toThrow(/Invalid run URL/);
  });

  test("throws when owner/repo/run_id are missing and no run_url", () => {
    expect(() =>
      resolveCoordinates({ owner: "owner", repo: "repo" })
    ).toThrow(/Must provide either/);
  });

  test("throws for a non-numeric run_id", () => {
    expect(() =>
      resolveCoordinates({ owner: "owner", repo: "repo", runId: "notanumber" })
    ).toThrow(/Invalid run_id/);
  });
});
