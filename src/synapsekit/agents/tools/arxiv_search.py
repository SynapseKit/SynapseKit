"""ArXiv Search Tool: search academic papers on arXiv."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class ArxivSearchTool(BaseTool):
    """Search arXiv for academic papers.

    Uses the arXiv API — no API key required, no extra dependencies (stdlib only).

    Usage::

        tool = ArxivSearchTool()
        result = await tool.run(query="attention is all you need")
    """

    name = "arxiv_search"
    description = (
        "Search arXiv for academic papers. "
        "Input: a search query. "
        "Returns: titles, authors, summaries, and links for matching papers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for arXiv",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of papers to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def run(self, query: str = "", max_results: int = 5, **kwargs: Any) -> ToolResult:
        search_query = query or kwargs.get("input", "")
        if not search_query:
            return ToolResult(output="", error="No search query provided.")

        try:
            import asyncio
            import urllib.request
            import xml.etree.ElementTree as ET
            from urllib.parse import quote_plus

            url = (
                f"http://export.arxiv.org/api/query"
                f"?search_query=all:{quote_plus(search_query)}"
                f"&start=0&max_results={max_results}"
            )

            loop = asyncio.get_event_loop()

            def _fetch():
                req = urllib.request.Request(url, headers={"User-Agent": "SynapseKit/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.read().decode()

            xml_data = await loop.run_in_executor(None, _fetch)
            root = ET.fromstring(xml_data)

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)

            if not entries:
                return ToolResult(output="No results found.")

            results = []
            for i, entry in enumerate(entries, 1):
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                authors = [
                    a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)
                ]
                link = ""
                for lnk in entry.findall("atom:link", ns):
                    if lnk.get("type") == "text/html":
                        link = lnk.get("href", "")
                        break
                if not link:
                    link = entry.findtext("atom:id", "", ns)

                author_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    author_str += f" (+{len(authors) - 5} more)"

                results.append(
                    f"{i}. **{title}**\n"
                    f"   Authors: {author_str}\n"
                    f"   Link: {link}\n"
                    f"   {summary[:500]}"
                )

            return ToolResult(output="\n\n".join(results))
        except Exception as e:
            return ToolResult(output="", error=f"arXiv search failed: {e}")
