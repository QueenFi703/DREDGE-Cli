"""
DREDGE Async GitHub Client
Provides async HTTP access to the GitHub API with connection pooling
and batch request support for improved throughput.
"""
import asyncio
from typing import Any, Dict, List, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    AIOHTTP_AVAILABLE = False

# Default concurrency limit for connection pool
DEFAULT_CONCURRENCY = 10

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com"


class AsyncGitHubClient:
    """
    Async GitHub API client with connection pooling.

    Uses aiohttp for non-blocking HTTP requests and a semaphore to cap
    concurrent connections.  Falls back gracefully when aiohttp is not
    installed (raises ``RuntimeError`` at request time).
    """

    def __init__(
        self,
        token: Optional[str] = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        base_url: str = GITHUB_API_BASE,
    ):
        """
        Initialise the client.

        Args:
            token: GitHub personal-access token (Bearer auth).  If omitted
                   the client makes unauthenticated requests.
            concurrency: Maximum simultaneous requests (default: 10).
            base_url: GitHub API base URL (override for GHE).
        """
        self.token = token
        self.concurrency = concurrency
        self.base_url = base_url.rstrip("/")
        self._session: Optional[Any] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncGitHubClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is required for AsyncGitHubClient. "
                "Install it with: pip install aiohttp"
            )
        if self._session is None or self._session.closed:
            headers: Dict[str, str] = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            connector = aiohttp.TCPConnector(limit=self.concurrency)
            self._session = aiohttp.ClientSession(
                headers=headers, connector=connector
            )
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Core request method
    # ------------------------------------------------------------------

    async def get(self, path: str, **params: Any) -> Dict[str, Any]:
        """
        Perform a single GET request to the GitHub API.

        Args:
            path: API path, e.g. ``/repos/owner/repo/pulls``.
            **params: Query-string parameters.

        Returns:
            Parsed JSON response as a dictionary (or list wrapped in a dict).
        """
        await self._ensure_session()
        url = f"{self.base_url}{path}"
        async with self._semaphore:  # type: ignore[union-attr]
            async with self._session.get(url, params=params or None) as resp:  # type: ignore[union-attr]
                resp.raise_for_status()
                data = await resp.json()
                # Normalise list responses to a consistent dict shape
                if isinstance(data, list):
                    return {"items": data, "count": len(data)}
                return data

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------

    async def get_batch(
        self, paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple API paths concurrently.

        Requests are fanned out up to ``concurrency`` at a time.

        Args:
            paths: List of API path strings.

        Returns:
            List of responses in the same order as ``paths``.
        """
        await self._ensure_session()
        tasks = [self.get(path) for path in paths]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    # ------------------------------------------------------------------
    # Convenience methods for common GitHub resources
    # ------------------------------------------------------------------

    async def get_pull_request(
        self, owner: str, repo: str, pr_number: int
    ) -> Dict[str, Any]:
        """Fetch a single pull request."""
        return await self.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")

    async def get_pull_requests_batch(
        self, owner: str, repo: str, pr_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """Fetch multiple pull requests concurrently."""
        paths = [
            f"/repos/{owner}/{repo}/pulls/{n}" for n in pr_numbers
        ]
        return await self.get_batch(paths)

    async def get_dependabot_alerts(
        self, owner: str, repo: str
    ) -> Dict[str, Any]:
        """Fetch all Dependabot security alerts for a repository."""
        return await self.get(
            f"/repos/{owner}/{repo}/dependabot/alerts",
            state="open",
        )


def fetch_pull_requests(
    owner: str,
    repo: str,
    pr_numbers: List[int],
    token: Optional[str] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """
    Synchronous convenience wrapper for batch PR fetching.

    Runs the async event loop internally so callers don't need to be async.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pr_numbers: List of PR numbers to fetch.
        token: Optional GitHub token.
        concurrency: Maximum simultaneous connections.

    Returns:
        List of PR response dicts.
    """

    async def _run() -> List[Dict[str, Any]]:
        async with AsyncGitHubClient(
            token=token, concurrency=concurrency
        ) as client:
            return await client.get_pull_requests_batch(owner, repo, pr_numbers)

    return asyncio.run(_run())
