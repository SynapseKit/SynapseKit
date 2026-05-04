from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from typing import Any

from .base import Document

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_ESEARCH_URL = f"{_BASE_URL}esearch.fcgi"
_EFETCH_URL = f"{_BASE_URL}efetch.fcgi"


def _element_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


class PubMedLoader:
    """Load biomedical articles from PubMed using NCBI E-utilities."""

    def __init__(
        self,
        query: str | None = None,
        pmids: list[str] | None = None,
        limit: int | None = 20,
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        normalized_query = (query or "").strip()
        normalized_pmids = [str(pmid).strip() for pmid in (pmids or []) if str(pmid).strip()]

        if not normalized_query and not normalized_pmids:
            raise ValueError("At least one of query or pmids must be provided")

        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._query = normalized_query or None
        self._pmids = normalized_pmids
        self._limit = limit
        self._email = email
        self._api_key = api_key

    def _throttle(self) -> None:
        delay_seconds = 0.1 if self._api_key else 0.34
        time.sleep(delay_seconds)

    def _request(self, endpoint: str, params: dict[str, Any]):
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[web]") from None

        self._throttle()
        try:
            response = httpx.get(endpoint, params=params, timeout=30.0)
        except httpx.RequestError as e:
            raise RuntimeError(f"PubMed request failed due to network error: {e}") from e
        if response.status_code != 200:
            raise RuntimeError(
                f"PubMed request failed ({endpoint}) with status "
                f"{response.status_code}: {response.text[:200]}"
            )
        return response

    def _esearch_pmids(self) -> list[str]:
        if not self._query:
            return []

        params: dict[str, Any] = {
            "db": "pubmed",
            "term": self._query,
            "retmode": "json",
            "tool": "synapsekit",
        }
        if self._limit is not None:
            params["retmax"] = self._limit
        if self._api_key:
            params["api_key"] = self._api_key
        if self._email:
            params["email"] = self._email

        response = self._request(_ESEARCH_URL, params)
        try:
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse ESearch response: {e}") from e

        if not isinstance(data, dict):
            return []

        esearch_result = data.get("esearchresult", {})
        if not isinstance(esearch_result, dict):
            return []
        id_list = esearch_result.get("idlist", [])
        if not isinstance(id_list, list):
            return []

        pmids = [str(pmid).strip() for pmid in id_list if str(pmid).strip()]
        if self._limit is not None:
            return pmids[: self._limit]
        return pmids

    def _collect_pmids(self) -> list[str]:
        combined_pmids = [*self._pmids]
        if self._query:
            combined_pmids.extend(self._esearch_pmids())

        unique_pmids: list[str] = []
        seen: set[str] = set()
        for pmid in combined_pmids:
            if pmid in seen:
                continue
            seen.add(pmid)
            unique_pmids.append(pmid)

        if self._limit is not None:
            return unique_pmids[: self._limit]
        return unique_pmids

    def _parse_authors(self, article: ET.Element) -> list[str]:
        authors: list[str] = []
        for author in article.findall(".//AuthorList/Author"):
            fore_name = _element_text(author.find("ForeName"))
            last_name = _element_text(author.find("LastName"))
            full_name = f"{fore_name} {last_name}".strip()

            if full_name:
                authors.append(full_name)
                continue

            collective_name = _element_text(author.find("CollectiveName"))
            if collective_name:
                authors.append(collective_name)

        return authors

    def _parse_publication_date(self, article: ET.Element) -> str:
        pub_date = article.find(".//PubDate")
        if pub_date is None:
            return ""

        medline_date = _element_text(pub_date.find("MedlineDate"))
        if medline_date:
            return medline_date

        year = _element_text(pub_date.find("Year"))
        month = _element_text(pub_date.find("Month"))
        day = _element_text(pub_date.find("Day"))
        parts = [part for part in (year, month, day) if part]
        return "-".join(parts)

    def _parse_efetch_xml(self, xml_text: str) -> list[Document]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        documents: list[Document] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = _element_text(article.find(".//PMID"))
            title = _element_text(article.find(".//ArticleTitle"))

            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                content = _element_text(abstract_text)
                label = abstract_text.attrib.get("Label")
                if label and content:
                    content = f"{label}: {content}"
                if content:
                    abstract_parts.append(content)
            abstract = "\n".join(abstract_parts)

            text_parts = [part for part in (title, abstract) if part]
            text = "\n\n".join(text_parts)
            if not text.strip():
                continue

            metadata = {
                "source": "pubmed",
                "pmid": pmid,
                "title": title,
                "authors": self._parse_authors(article),
                "journal": _element_text(article.find(".//Journal/Title")),
                "publication_date": self._parse_publication_date(article),
            }
            metadata = {k: v for k, v in metadata.items() if v not in (None, "", [], {})}
            documents.append(Document(text=text, metadata=metadata))

        return documents

    def _efetch_documents(self, pmids: list[str]) -> list[Document]:
        if not pmids:
            return []

        batch_size = 200
        documents: list[Document] = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]

            params: dict[str, Any] = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "tool": "synapsekit",
            }

            if self._api_key:
                params["api_key"] = self._api_key
            if self._email:
                params["email"] = self._email

            response = self._request(_EFETCH_URL, params)
            documents.extend(self._parse_efetch_xml(response.text))

        return documents

    def load(self) -> list[Document]:
        pmids = self._collect_pmids()
        if not pmids:
            return []
        return self._efetch_documents(pmids)

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
