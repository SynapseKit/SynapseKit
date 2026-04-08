"""Jira issue loader using JQL queries."""

import asyncio
from typing import Any

from .base import Document


class JiraLoader:
    """Load Jira issues using JQL queries.

    Args:
        url: Jira instance URL (e.g., 'https://your-domain.atlassian.net')
        username: Jira username/email
        api_token: Jira API token
        jql: JQL query string (e.g., 'project = PROJ AND status = Open')
        limit: Maximum number of issues to load (default: None for all)

    Example:
        >>> loader = JiraLoader(
        ...     url="https://your-domain.atlassian.net",
        ...     username="your-email@example.com",
        ...     api_token="your-api-token",
        ...     jql="project = MYPROJ AND status = Open"
        ... )
        >>> documents = loader.load()
    """

    def __init__(
        self,
        url: str,
        username: str,
        api_token: str,
        jql: str,
        limit: int | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._username = username
        self._api_token = api_token
        self._jql = jql
        self._limit = limit

    def _extract_text_from_adf(self, node: Any) -> str:
        """Extract text from Atlassian Document Format (ADF) recursively.

        Args:
            node: ADF node (dict or str)

        Returns:
            Plain text extracted from the ADF structure
        """
        if isinstance(node, str):
            return node

        if not isinstance(node, dict):
            return ""

        text_parts = []

        # Extract direct text
        if "text" in node:
            text_parts.append(node["text"])

        # Recursively process content array
        if "content" in node and isinstance(node["content"], list):
            for child in node["content"]:
                child_text = self._extract_text_from_adf(child)
                if child_text:
                    text_parts.append(child_text)

        return " ".join(text_parts)

    def _extract_description(self, description: Any) -> str:
        """Extract description text from either plain string or ADF format.

        Args:
            description: Description field from Jira (can be string, dict, or None)

        Returns:
            Plain text description
        """
        if not description:
            return ""

        if isinstance(description, str):
            return description

        # ADF format
        if isinstance(description, dict):
            return self._extract_text_from_adf(description)

        return ""

    def _extract_comments(self, comment_field: Any) -> list[str]:
        """Extract comment texts from comment field.

        Args:
            comment_field: Comment field from Jira (can be dict or list)

        Returns:
            List of comment texts
        """
        if not comment_field:
            return []

        comments_list = []

        # Handle nested structure: {"comments": [...]}
        if isinstance(comment_field, dict):
            comments_list = comment_field.get("comments", [])
        # Handle direct list
        elif isinstance(comment_field, list):
            comments_list = comment_field
        else:
            return []

        comment_texts = []
        for comment in comments_list:
            if not isinstance(comment, dict):
                continue

            body = comment.get("body")
            if not body:
                continue

            # Extract text from ADF or plain string
            if isinstance(body, str):
                comment_texts.append(body)
            elif isinstance(body, dict):
                text = self._extract_text_from_adf(body)
                if text:
                    comment_texts.append(text)

        return comment_texts

    def _issue_to_document(self, issue: dict[str, Any]) -> Document:
        """Convert Jira issue to Document.

        Args:
            issue: Jira issue object from API

        Returns:
            Document with combined text and metadata
        """
        issue_key = issue.get("key", "")
        fields = issue.get("fields", {})

        # Extract fields
        summary = fields.get("summary", "")
        description = self._extract_description(fields.get("description"))
        status = fields.get("status", {}).get("name", "")

        assignee = fields.get("assignee")
        assignee_name = assignee.get("displayName", "") if assignee else None

        priority = fields.get("priority")
        priority_name = priority.get("name", "") if priority else None

        comments = self._extract_comments(fields.get("comment"))

        # Combine text
        text_parts = []
        if summary:
            text_parts.append(f"Summary: {summary}")
        if description:
            text_parts.append(f"\nDescription: {description}")
        if comments:
            text_parts.append("\n\nComments:")
            for i, comment in enumerate(comments, 1):
                text_parts.append(f"\n{i}. {comment}")

        combined_text = "".join(text_parts)

        # Build metadata
        metadata = {
            "source": "jira",
            "key": issue_key,
            "status": status,
        }

        if assignee_name:
            metadata["assignee"] = assignee_name
        if priority_name:
            metadata["priority"] = priority_name

        return Document(text=combined_text, metadata=metadata)

    async def aload(self) -> list[Document]:
        """Load Jira issues asynchronously.

        Returns:
            List of Document objects containing issue data

        Raises:
            ImportError: If httpx is not installed
            RuntimeError: If API request fails
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx required for JiraLoader. Install with: pip install synapsekit[web]"
            ) from None

        endpoint = f"{self._url}/rest/api/3/search/jql"
        auth = (self._username, self._api_token)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        documents = []
        next_page_token = None
        max_retries = 3

        async with httpx.AsyncClient() as client:
            while True:
                payload = {
                    "jql": self._jql,
                    "maxResults": 50,
                    "fields": [
                        "summary",
                        "description",
                        "status",
                        "assignee",
                        "priority",
                        "comment",
                    ],
                }

                if next_page_token:
                    payload["nextPageToken"] = next_page_token

                # Retry logic for rate limits
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            endpoint,
                            json=payload,
                            auth=auth,
                            headers=headers,
                            timeout=30.0,
                        )

                        # Handle rate limits
                        if response.status_code == 429 and attempt < max_retries - 1:
                            retry_after = int(response.headers.get("Retry-After", 2))
                            await asyncio.sleep(retry_after)
                            continue

                        response.raise_for_status()
                        break

                    except httpx.HTTPStatusError as e:
                        if attempt == max_retries - 1:
                            raise RuntimeError(
                                f"Failed to fetch Jira issues: {e.response.status_code} {e.response.text}"
                            ) from e
                        await asyncio.sleep(2**attempt)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Failed to fetch Jira issues: {e}") from e
                        await asyncio.sleep(2**attempt)

                data = response.json()
                issues = data.get("issues", [])

                # Convert issues to documents
                for issue in issues:
                    doc = self._issue_to_document(issue)
                    documents.append(doc)

                    if self._limit and len(documents) >= self._limit:
                        return documents[: self._limit]

                # Check pagination
                is_last = data.get("isLast", True)
                if is_last:
                    break

                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break

        return documents

    def load(self) -> list[Document]:
        """Load Jira issues synchronously.

        Returns:
            List of Document objects containing issue data
        """
        return asyncio.run(self.aload())
