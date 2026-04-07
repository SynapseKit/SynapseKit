from __future__ import annotations

import asyncio
import base64
from typing import Literal

from .base import Document

ContentType = Literal["files", "issues", "prs", "readme"]


class GitHubLoader:
    """Load files, issues, pull requests, or README from a GitHub repository."""

    def __init__(
        self,
        repo: str,
        content_type: ContentType = "readme",
        token: str | None = None,
        path: str | None = None,
        limit: int | None = None,
    ) -> None:
        """
        Initialize GitHubLoader.

        Args:
            repo: Repository in "owner/repo" format
            content_type: Type of content to load ("files", "issues", "prs", "readme")
            token: Optional GitHub token for higher rate limits
            path: Optional path filter for files (e.g., "src/")
            limit: Optional limit on number of items to load
        """
        self._repo = repo
        self._content_type = content_type
        self._token = token
        self._path = path
        self._limit = limit

    def _get_headers(self) -> dict[str, str]:
        """Build HTTP headers for GitHub API."""
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _request_with_retry(
        self, client, url: str, max_retries: int = 3
    ) -> dict | list:
        """Make HTTP request with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                response = await client.get(url, headers=self._get_headers())
                
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                
                response.raise_for_status()
                return response.json()
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise RuntimeError("Max retries exceeded")

    async def _load_readme(self, client) -> list[Document]:
        """Load repository README."""
        url = f"https://api.github.com/repos/{self._repo}/readme"
        data = await self._request_with_retry(client, url)
        
        content = base64.b64decode(data["content"]).decode("utf-8")
        readme_url = f"https://github.com/{self._repo}#readme"
        
        return [
            Document(
                text=content,
                metadata={
                    "source": "github",
                    "repo": self._repo,
                    "type": "readme",
                    "url": readme_url,
                },
            )
        ]

    async def _load_issues(self, client) -> list[Document]:
        """Load repository issues (excluding pull requests)."""
        url = f"https://api.github.com/repos/{self._repo}/issues"
        params = []
        if self._limit:
            params.append(f"per_page={self._limit}")
        if params:
            url += "?" + "&".join(params)
        
        issues = await self._request_with_retry(client, url)
        
        docs = []
        for issue in issues:
            # Filter out pull requests
            if "pull_request" in issue:
                continue
            
            text = f"# {issue['title']}\n\n{issue.get('body', '')}"
            metadata = {
                "source": "github",
                "repo": self._repo,
                "type": "issue",
                "author": issue["user"]["login"],
                "date": issue["created_at"],
                "url": issue["html_url"],
                "number": issue["number"],
            }
            docs.append(Document(text=text, metadata=metadata))
            
            if self._limit and len(docs) >= self._limit:
                break
        
        return docs

    async def _load_prs(self, client) -> list[Document]:
        """Load repository pull requests."""
        url = f"https://api.github.com/repos/{self._repo}/pulls"
        params = []
        if self._limit:
            params.append(f"per_page={self._limit}")
        if params:
            url += "?" + "&".join(params)
        
        prs = await self._request_with_retry(client, url)
        
        docs = []
        for pr in prs:
            text = f"# {pr['title']}\n\n{pr.get('body', '')}"
            metadata = {
                "source": "github",
                "repo": self._repo,
                "type": "pr",
                "author": pr["user"]["login"],
                "date": pr["created_at"],
                "url": pr["html_url"],
                "number": pr["number"],
            }
            docs.append(Document(text=text, metadata=metadata))
            
            if self._limit and len(docs) >= self._limit:
                break
        
        return docs

    async def _load_files(self, client) -> list[Document]:
        """Load repository files using Git Trees API."""
        # Get default branch
        repo_url = f"https://api.github.com/repos/{self._repo}"
        repo_data = await self._request_with_retry(client, repo_url)
        default_branch = repo_data.get("default_branch", "main")
        
        # Get full file tree
        tree_url = f"https://api.github.com/repos/{self._repo}/git/trees/{default_branch}?recursive=1"
        tree_data = await self._request_with_retry(client, tree_url)
        
        # Filter for blob (file) types
        files = [item for item in tree_data["tree"] if item["type"] == "blob"]
        
        # Apply path filter if specified
        if self._path:
            files = [f for f in files if f["path"].startswith(self._path)]
        
        # Apply limit
        if self._limit:
            files = files[: self._limit]
        
        docs = []
        for file_item in files:
            file_path = file_item["path"]
            
            # Fetch file content using raw URL
            raw_url = f"https://raw.githubusercontent.com/{self._repo}/{default_branch}/{file_path}"
            
            try:
                response = await client.get(raw_url, headers=self._get_headers())
                response.raise_for_status()
                content = response.text
            except Exception:
                # Skip files that can't be fetched (binary, too large, etc.)
                continue
            
            github_url = f"https://github.com/{self._repo}/blob/{default_branch}/{file_path}"
            
            docs.append(
                Document(
                    text=content,
                    metadata={
                        "source": "github",
                        "repo": self._repo,
                        "type": "file",
                        "path": file_path,
                        "url": github_url,
                    },
                )
            )
        
        return docs

    async def load(self) -> list[Document]:
        """Load content from GitHub repository."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        async with httpx.AsyncClient() as client:
            if self._content_type == "readme":
                return await self._load_readme(client)
            elif self._content_type == "issues":
                return await self._load_issues(client)
            elif self._content_type == "prs":
                return await self._load_prs(client)
            elif self._content_type == "files":
                return await self._load_files(client)
            else:
                raise ValueError(f"Unknown content_type: {self._content_type}")

    def load_sync(self) -> list[Document]:
        """Synchronous wrapper for load()."""
        return asyncio.run(self.load())
