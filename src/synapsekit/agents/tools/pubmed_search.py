"""PubMed Search Tool: search biomedical literature on PubMed."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class PubMedSearchTool(BaseTool):
    """Search PubMed for biomedical and life science research articles.

    Uses the NCBI E-utilities API — no API key required, no extra dependencies
    (stdlib only).

    Usage::

        tool = PubMedSearchTool()
        result = await tool.run(query="CRISPR gene editing")
    """

    name = "pubmed_search"
    description = (
        "Search PubMed for biomedical and life science research articles. "
        "Input: a search query. "
        "Returns: titles, authors, abstracts, and PubMed IDs for matching articles."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for PubMed",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of articles to return (default: 5)",
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

            loop = asyncio.get_event_loop()

            # Step 1: ESearch to get PMIDs
            esearch_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&term={quote_plus(search_query)}"
                f"&retmax={max_results}&retmode=xml"
            )

            def _esearch() -> str:
                req = urllib.request.Request(esearch_url, headers={"User-Agent": "SynapseKit/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.read().decode()

            esearch_xml = await loop.run_in_executor(None, _esearch)
            esearch_root = ET.fromstring(esearch_xml)

            id_list = esearch_root.findall(".//Id")
            if not id_list:
                return ToolResult(output="No results found.")

            pmids = [id_el.text for id_el in id_list if id_el.text]
            if not pmids:
                return ToolResult(output="No results found.")

            # Step 2: EFetch to get article details
            efetch_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=pubmed&id={','.join(pmids)}&retmode=xml"
            )

            def _efetch() -> str:
                req = urllib.request.Request(efetch_url, headers={"User-Agent": "SynapseKit/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.read().decode()

            efetch_xml = await loop.run_in_executor(None, _efetch)
            efetch_root = ET.fromstring(efetch_xml)

            articles = efetch_root.findall(".//PubmedArticle")
            if not articles:
                return ToolResult(output="No results found.")

            results = []
            for i, article in enumerate(articles, 1):
                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None and title_el.text else "Untitled"

                # Authors
                author_els = article.findall(".//Author")
                authors = []
                for author in author_els:
                    last = author.findtext("LastName", "")
                    fore = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {fore}".strip())

                author_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    author_str += f" (+{len(authors) - 5} more)"

                # Abstract
                abstract_parts = article.findall(".//AbstractText")
                abstract = " ".join(part.text for part in abstract_parts if part.text)[:500]

                # PMID
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""

                results.append(
                    f"{i}. **{title}**\n   Authors: {author_str}\n   PMID: {pmid}\n   {abstract}"
                )

            return ToolResult(output="\n\n".join(results))
        except Exception as e:
            return ToolResult(output="", error=f"PubMed search failed: {e}")
