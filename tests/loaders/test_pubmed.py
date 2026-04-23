from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.pubmed import PubMedLoader


def _mock_response(status_code: int = 200, json_data=None, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=json_data)
    response.text = text
    return response


def _pubmed_xml() -> str:
    return """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Targeted Therapy in Oncology</ArticleTitle>
        <Abstract>
          <AbstractText>Background text.</AbstractText>
          <AbstractText>Methods text.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Journal of Clinical Testing</Title>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
              <Month>01</Month>
              <Day>15</Day>
            </PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_init_requires_query_or_pmids() -> None:
    with pytest.raises(ValueError, match="At least one of query or pmids"):
        PubMedLoader()


def test_query_flow_uses_esearch_then_efetch() -> None:
    esearch_response = _mock_response(json_data={"esearchresult": {"idlist": ["12345", "67890"]}})
    efetch_response = _mock_response(text=_pubmed_xml())

    with patch("httpx.get", side_effect=[esearch_response, efetch_response]) as mock_get:
        with patch("time.sleep") as mock_sleep:
            docs = PubMedLoader(
                query="oncology",
                limit=2,
                email="research@example.com",
                api_key="test-key",
            ).load()

    assert len(docs) == 1
    assert docs[0].metadata["pmid"] == "12345"
    assert mock_get.call_count == 2

    first_call = mock_get.call_args_list[0]
    assert first_call.args[0].endswith("esearch.fcgi")
    assert first_call.kwargs["params"]["db"] == "pubmed"
    assert first_call.kwargs["params"]["term"] == "oncology"
    assert first_call.kwargs["params"]["retmax"] == 2
    assert first_call.kwargs["params"]["retmode"] == "json"
    assert first_call.kwargs["params"]["api_key"] == "test-key"
    assert first_call.kwargs["params"]["email"] == "research@example.com"

    second_call = mock_get.call_args_list[1]
    assert second_call.args[0].endswith("efetch.fcgi")
    assert second_call.kwargs["params"]["db"] == "pubmed"
    assert second_call.kwargs["params"]["id"] == "12345,67890"
    assert second_call.kwargs["params"]["retmode"] == "xml"

    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(0.1)


def test_direct_pmid_flow_skips_esearch() -> None:
    efetch_response = _mock_response(text=_pubmed_xml())

    with patch("httpx.get", return_value=efetch_response) as mock_get:
        with patch("time.sleep"):
            docs = PubMedLoader(pmids=["12345"]).load()

    assert len(docs) == 1
    assert mock_get.call_count == 1
    assert mock_get.call_args.args[0].endswith("efetch.fcgi")
    assert mock_get.call_args.kwargs["params"]["id"] == "12345"


def test_empty_results_return_empty_list() -> None:
    esearch_response = _mock_response(json_data={"esearchresult": {"idlist": []}})

    with patch("httpx.get", return_value=esearch_response) as mock_get:
        with patch("time.sleep"):
            docs = PubMedLoader(query="no_results").load()

    assert docs == []
    assert mock_get.call_count == 1
    assert mock_get.call_args.args[0].endswith("esearch.fcgi")


def test_malformed_xml_returns_empty_list() -> None:
    efetch_response = _mock_response(text="<PubmedArticleSet><broken>")

    with patch("httpx.get", return_value=efetch_response):
        with patch("time.sleep"):
            docs = PubMedLoader(pmids=["12345"]).load()

    assert docs == []


def test_metadata_and_text_parsing() -> None:
    efetch_response = _mock_response(text=_pubmed_xml())

    with patch("httpx.get", return_value=efetch_response):
        with patch("time.sleep"):
            docs = PubMedLoader(pmids=["12345"]).load()

    assert len(docs) == 1
    doc = docs[0]

    assert "Targeted Therapy in Oncology" in doc.text
    assert "Background text." in doc.text
    assert "Methods text." in doc.text

    metadata = doc.metadata
    assert metadata["source"] == "pubmed"
    assert metadata["pmid"] == "12345"
    assert metadata["title"] == "Targeted Therapy in Oncology"
    assert metadata["authors"] == ["John Smith", "Jane Doe"]
    assert metadata["journal"] == "Journal of Clinical Testing"
    assert metadata["publication_date"] == "2024-01-15"


def test_http_error_raises_runtime_error() -> None:
    error_response = _mock_response(status_code=503, text="Service Unavailable")

    with patch("httpx.get", return_value=error_response):
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="PubMed request failed"):
                PubMedLoader(pmids=["12345"]).load()


def test_aload_delegates_to_load() -> None:
    loader = PubMedLoader(pmids=["12345"])
    with patch.object(loader, "load", return_value=[Document(text="x", metadata={})]) as mock_load:
        docs = asyncio.run(loader.aload())

    mock_load.assert_called_once()
    assert len(docs) == 1


def test_batching_multiple_requests() -> None:
    pmids = [str(i) for i in range(300)]

    efetch_response = _mock_response(text=_pubmed_xml())

    with patch("httpx.get", return_value=efetch_response) as mock_get:
        with patch("time.sleep"):
            PubMedLoader(pmids=pmids, limit=None).load()

    assert mock_get.call_count == 2


def test_metadata_excludes_empty_fields() -> None:
    xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123</PMID>
      <Article>
        <ArticleTitle></ArticleTitle>
        <Abstract>
          <AbstractText>Only abstract content.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

    efetch_response = _mock_response(text=xml)

    with patch("httpx.get", return_value=efetch_response):
        with patch("time.sleep"):
            docs = PubMedLoader(pmids=["123"]).load()

    assert len(docs) == 1
    metadata = docs[0].metadata

    assert "title" not in metadata
    assert "journal" not in metadata


def test_abstract_labels_are_included() -> None:
    xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123</PMID>
      <Article>
        <ArticleTitle>Test Title</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background text.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

    efetch_response = _mock_response(text=xml)

    with patch("httpx.get", return_value=efetch_response):
        with patch("time.sleep"):
            docs = PubMedLoader(pmids=["123"]).load()

    assert len(docs) == 1
    assert "BACKGROUND: Background text." in docs[0].text


def test_tool_parameter_in_requests() -> None:
    esearch_response = _mock_response(json_data={"esearchresult": {"idlist": ["123"]}})
    efetch_response = _mock_response(text=_pubmed_xml())

    with patch("httpx.get", side_effect=[esearch_response, efetch_response]) as mock_get:
        with patch("time.sleep"):
            PubMedLoader(query="test").load()

    esearch_call = mock_get.call_args_list[0]
    assert esearch_call.kwargs["params"]["tool"] == "synapsekit"

    efetch_call = mock_get.call_args_list[1]
    assert efetch_call.kwargs["params"]["tool"] == "synapsekit"
