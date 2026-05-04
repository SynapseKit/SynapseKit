"""Tests for new loaders: Trello, Firestore, Zendesk, Intercom, Freshdesk,
HackerNews, Reddit, Twitter, GoogleCalendar."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document

# ---------------------------------------------------------------------------
# TrelloLoader
# ---------------------------------------------------------------------------


class TestTrelloLoader:
    def test_missing_api_key(self) -> None:
        from synapsekit.loaders.trello import TrelloLoader

        with pytest.raises(ValueError, match="api_key must be provided"):
            TrelloLoader(api_key="", token="tok", board_id="bid")

    def test_missing_token(self) -> None:
        from synapsekit.loaders.trello import TrelloLoader

        with pytest.raises(ValueError, match="token must be provided"):
            TrelloLoader(api_key="key", token="", board_id="bid")

    def test_missing_board_id(self) -> None:
        from synapsekit.loaders.trello import TrelloLoader

        with pytest.raises(ValueError, match="board_id must be provided"):
            TrelloLoader(api_key="key", token="tok", board_id="")

    def test_import_error(self) -> None:
        from synapsekit.loaders.trello import TrelloLoader

        loader = TrelloLoader(api_key="k", token="t", board_id="b")
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="requests required"):
                loader.load()

    def test_load_returns_documents(self) -> None:
        from synapsekit.loaders.trello import TrelloLoader

        mock_requests = MagicMock()

        # lists endpoint
        lists_resp = MagicMock()
        lists_resp.json.return_value = [{"id": "list1", "name": "To Do"}]
        # cards endpoint
        cards_resp = MagicMock()
        cards_resp.json.return_value = [
            {
                "id": "card1",
                "name": "Fix bug",
                "desc": "Details here",
                "idList": "list1",
                "labels": [{"name": "bug"}],
                "shortUrl": "https://trello.com/c/card1",
            }
        ]
        mock_requests.get.side_effect = [lists_resp, cards_resp]

        loader = TrelloLoader(api_key="k", token="t", board_id="b")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Fix bug" in docs[0].text
        assert "Details here" in docs[0].text
        assert docs[0].metadata["source"] == "trello"
        assert docs[0].metadata["card_id"] == "card1"
        assert docs[0].metadata["list_name"] == "To Do"
        assert docs[0].metadata["labels"] == ["bug"]

    def test_filter_by_list_names(self) -> None:
        from synapsekit.loaders.trello import TrelloLoader

        mock_requests = MagicMock()

        lists_resp = MagicMock()
        lists_resp.json.return_value = [
            {"id": "list1", "name": "To Do"},
            {"id": "list2", "name": "Done"},
        ]
        cards_resp = MagicMock()
        cards_resp.json.return_value = [
            {
                "id": "card1",
                "name": "Task A",
                "desc": "",
                "idList": "list1",
                "labels": [],
                "shortUrl": "",
            },
            {
                "id": "card2",
                "name": "Task B",
                "desc": "",
                "idList": "list2",
                "labels": [],
                "shortUrl": "",
            },
        ]
        mock_requests.get.side_effect = [lists_resp, cards_resp]

        loader = TrelloLoader(api_key="k", token="t", board_id="b", list_names=["To Do"])
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["card_id"] == "card1"

    def test_aload(self) -> None:
        from synapsekit.loaders.trello import TrelloLoader

        mock_requests = MagicMock()
        lists_resp = MagicMock()
        lists_resp.json.return_value = [{"id": "l1", "name": "Backlog"}]
        cards_resp = MagicMock()
        cards_resp.json.return_value = [
            {
                "id": "c1",
                "name": "Async card",
                "desc": "",
                "idList": "l1",
                "labels": [],
                "shortUrl": "",
            }
        ]
        mock_requests.get.side_effect = [lists_resp, cards_resp]

        loader = TrelloLoader(api_key="k", token="t", board_id="b")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1
        assert docs[0].text == "Async card"


# ---------------------------------------------------------------------------
# FirestoreLoader
# ---------------------------------------------------------------------------


class TestFirestoreLoader:
    def test_missing_project_id(self) -> None:
        from synapsekit.loaders.firebase import FirestoreLoader

        with pytest.raises(ValueError, match="project_id must be provided"):
            FirestoreLoader(project_id="", collection="col")

    def test_missing_collection(self) -> None:
        from synapsekit.loaders.firebase import FirestoreLoader

        with pytest.raises(ValueError, match="collection must be provided"):
            FirestoreLoader(project_id="proj", collection="")

    def test_import_error(self) -> None:
        from synapsekit.loaders.firebase import FirestoreLoader

        loader = FirestoreLoader(project_id="p", collection="c")
        with patch.dict("sys.modules", {"google.cloud": None, "google.cloud.firestore": None}):
            with pytest.raises(ImportError, match="google-cloud-firestore required"):
                loader.load()

    def _make_firestore_mocks(self, docs_data: list[tuple[str, dict]]) -> MagicMock:
        """Build a mock firestore module with given (doc_id, data) pairs."""
        mock_firestore_mod = MagicMock()
        mock_client_inst = MagicMock()
        mock_firestore_mod.Client.return_value = mock_client_inst

        doc_refs = []
        for doc_id, data in docs_data:
            ref = MagicMock()
            ref.id = doc_id
            ref.to_dict.return_value = data
            doc_refs.append(ref)

        mock_query = MagicMock()
        mock_query.stream.return_value = doc_refs
        mock_client_inst.collection.return_value = mock_query
        return mock_firestore_mod

    def _make_sys_modules_patch(self, mock_firestore_mod: MagicMock) -> dict:
        """Build sys.modules patch dict wiring google.cloud.firestore correctly."""
        mock_google_cloud = MagicMock()
        mock_google_cloud.firestore = mock_firestore_mod
        mock_google = MagicMock()
        mock_google.cloud = mock_google_cloud
        mock_oauth2 = MagicMock()
        mock_sa = MagicMock()
        mock_oauth2.service_account = mock_sa
        mock_google.oauth2 = mock_oauth2
        return {
            "google": mock_google,
            "google.cloud": mock_google_cloud,
            "google.cloud.firestore": mock_firestore_mod,
            "google.oauth2": mock_oauth2,
            "google.oauth2.service_account": mock_sa,
        }

    def test_load_returns_documents(self) -> None:
        from synapsekit.loaders.firebase import FirestoreLoader

        mock_firestore_mod = self._make_firestore_mocks(
            [("doc123", {"title": "Hello", "body": "World"})]
        )
        loader = FirestoreLoader(project_id="proj", collection="articles")
        with patch.dict("sys.modules", self._make_sys_modules_patch(mock_firestore_mod)):
            docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].metadata["source"] == "firestore"
        assert docs[0].metadata["doc_id"] == "doc123"
        assert docs[0].metadata["collection"] == "articles"

    def test_metadata_fields_separation(self) -> None:
        from synapsekit.loaders.firebase import FirestoreLoader

        mock_firestore_mod = self._make_firestore_mocks(
            [("d1", {"title": "My Title", "author": "Alice", "body": "Content"})]
        )
        loader = FirestoreLoader(
            project_id="proj",
            collection="articles",
            metadata_fields=["author"],
        )
        with patch.dict("sys.modules", self._make_sys_modules_patch(mock_firestore_mod)):
            docs = loader.load()

        assert docs[0].metadata["author"] == "Alice"
        assert "author" not in docs[0].text
        assert "title: My Title" in docs[0].text

    def test_aload(self) -> None:
        from synapsekit.loaders.firebase import FirestoreLoader

        mock_firestore_mod = self._make_firestore_mocks([("adoc", {"k": "v"})])
        loader = FirestoreLoader(project_id="proj", collection="col")
        with patch.dict("sys.modules", self._make_sys_modules_patch(mock_firestore_mod)):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1


# ---------------------------------------------------------------------------
# ZendeskLoader
# ---------------------------------------------------------------------------


class TestZendeskLoader:
    def test_missing_subdomain(self) -> None:
        from synapsekit.loaders.zendesk import ZendeskLoader

        with pytest.raises(ValueError, match="subdomain must be provided"):
            ZendeskLoader(subdomain="", email="e", api_token="t")

    def test_missing_email(self) -> None:
        from synapsekit.loaders.zendesk import ZendeskLoader

        with pytest.raises(ValueError, match="email must be provided"):
            ZendeskLoader(subdomain="sub", email="", api_token="t")

    def test_missing_api_token(self) -> None:
        from synapsekit.loaders.zendesk import ZendeskLoader

        with pytest.raises(ValueError, match="api_token must be provided"):
            ZendeskLoader(subdomain="sub", email="e", api_token="")

    def test_invalid_status(self) -> None:
        from synapsekit.loaders.zendesk import ZendeskLoader

        with pytest.raises(ValueError, match="status must be one of"):
            ZendeskLoader(subdomain="s", email="e", api_token="t", status="invalid")

    def test_import_error(self) -> None:
        from synapsekit.loaders.zendesk import ZendeskLoader

        loader = ZendeskLoader(subdomain="s", email="e", api_token="t")
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="requests required"):
                loader.load()

    def test_load_returns_documents(self) -> None:
        from synapsekit.loaders.zendesk import ZendeskLoader

        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "tickets": [
                {
                    "id": 1,
                    "subject": "Login issue",
                    "description": "Cannot log in",
                    "status": "open",
                    "priority": "high",
                    "requester_id": 42,
                    "created_at": "2024-01-01T00:00:00Z",
                }
            ],
            "next_page": None,
        }
        mock_requests.get.return_value = mock_resp

        loader = ZendeskLoader(subdomain="myco", email="a@b.com", api_token="tok")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Login issue" in docs[0].text
        assert "Cannot log in" in docs[0].text
        assert docs[0].metadata["source"] == "zendesk"
        assert docs[0].metadata["ticket_id"] == 1
        assert docs[0].metadata["status"] == "open"

    def test_limit(self) -> None:
        from synapsekit.loaders.zendesk import ZendeskLoader

        mock_requests = MagicMock()
        mock_resp = MagicMock()
        tickets = [
            {
                "id": i,
                "subject": f"Ticket {i}",
                "description": "",
                "status": "open",
                "priority": None,
                "requester_id": i,
                "created_at": "",
            }
            for i in range(5)
        ]
        mock_resp.json.return_value = {"tickets": tickets, "next_page": None}
        mock_requests.get.return_value = mock_resp

        loader = ZendeskLoader(subdomain="s", email="e", api_token="t", limit=2)
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 2

    def test_aload(self) -> None:
        from synapsekit.loaders.zendesk import ZendeskLoader

        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "tickets": [
                {
                    "id": 99,
                    "subject": "Async",
                    "description": "",
                    "status": "open",
                    "priority": None,
                    "requester_id": 1,
                    "created_at": "",
                }
            ],
            "next_page": None,
        }
        mock_requests.get.return_value = mock_resp

        loader = ZendeskLoader(subdomain="s", email="e", api_token="t")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1


# ---------------------------------------------------------------------------
# IntercomLoader
# ---------------------------------------------------------------------------


class TestIntercomLoader:
    def test_missing_access_token(self) -> None:
        from synapsekit.loaders.intercom import IntercomLoader

        with pytest.raises(ValueError, match="access_token must be provided"):
            IntercomLoader(access_token="")

    def test_invalid_state(self) -> None:
        from synapsekit.loaders.intercom import IntercomLoader

        with pytest.raises(ValueError, match="state must be one of"):
            IntercomLoader(access_token="tok", state="unknown")

    def test_import_error(self) -> None:
        from synapsekit.loaders.intercom import IntercomLoader

        loader = IntercomLoader(access_token="tok")
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="requests required"):
                loader.load()

    def test_load_returns_documents(self) -> None:
        from synapsekit.loaders.intercom import IntercomLoader

        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "conversations": [
                {
                    "id": "conv1",
                    "title": "Help with billing",
                    "source": {"body": "I was charged twice"},
                    "state": "open",
                    "created_at": 1700000000,
                    "contacts": {"contacts": [{"id": "user1"}]},
                }
            ],
            "pages": {"next": None},
        }
        mock_requests.get.return_value = mock_resp

        loader = IntercomLoader(access_token="tok")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Help with billing" in docs[0].text
        assert docs[0].metadata["source"] == "intercom"
        assert docs[0].metadata["conversation_id"] == "conv1"
        assert docs[0].metadata["contact_id"] == "user1"

    def test_limit(self) -> None:
        from synapsekit.loaders.intercom import IntercomLoader

        mock_requests = MagicMock()
        mock_resp = MagicMock()
        conversations = [
            {
                "id": f"c{i}",
                "title": f"Conv {i}",
                "source": {"body": ""},
                "state": "open",
                "created_at": 0,
                "contacts": {"contacts": []},
            }
            for i in range(5)
        ]
        mock_resp.json.return_value = {
            "conversations": conversations,
            "pages": {"next": None},
        }
        mock_requests.get.return_value = mock_resp

        loader = IntercomLoader(access_token="tok", limit=3)
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 3

    def test_aload(self) -> None:
        from synapsekit.loaders.intercom import IntercomLoader

        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "conversations": [
                {
                    "id": "ac1",
                    "title": "Async conv",
                    "source": {"body": "body text"},
                    "state": "open",
                    "created_at": 0,
                    "contacts": {"contacts": []},
                }
            ],
            "pages": {"next": None},
        }
        mock_requests.get.return_value = mock_resp

        loader = IntercomLoader(access_token="tok")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1


# ---------------------------------------------------------------------------
# FreshdeskLoader
# ---------------------------------------------------------------------------


class TestFreshdeskLoader:
    def test_missing_domain(self) -> None:
        from synapsekit.loaders.freshdesk import FreshdeskLoader

        with pytest.raises(ValueError, match="domain must be provided"):
            FreshdeskLoader(domain="", api_key="k")

    def test_missing_api_key(self) -> None:
        from synapsekit.loaders.freshdesk import FreshdeskLoader

        with pytest.raises(ValueError, match="api_key must be provided"):
            FreshdeskLoader(domain="myco", api_key="")

    def test_import_error(self) -> None:
        from synapsekit.loaders.freshdesk import FreshdeskLoader

        loader = FreshdeskLoader(domain="myco", api_key="k")
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="requests required"):
                loader.load()

    def test_load_returns_documents(self) -> None:
        from synapsekit.loaders.freshdesk import FreshdeskLoader

        mock_requests = MagicMock()
        mock_resp_page1 = MagicMock()
        mock_resp_page1.json.return_value = [
            {
                "id": 1,
                "subject": "Broken login",
                "description_text": "Cannot sign in",
                "status": 2,
                "priority": 1,
                "requester_id": 10,
                "created_at": "2024-01-01",
            }
        ]
        mock_resp_page2 = MagicMock()
        mock_resp_page2.json.return_value = []  # stop pagination
        mock_requests.get.side_effect = [mock_resp_page1, mock_resp_page2]

        loader = FreshdeskLoader(domain="myco", api_key="k")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Broken login" in docs[0].text
        assert "Cannot sign in" in docs[0].text
        assert docs[0].metadata["source"] == "freshdesk"
        assert docs[0].metadata["ticket_id"] == 1

    def test_limit(self) -> None:
        from synapsekit.loaders.freshdesk import FreshdeskLoader

        mock_requests = MagicMock()
        tickets = [
            {
                "id": i,
                "subject": f"T{i}",
                "description_text": "",
                "status": 2,
                "priority": 1,
                "requester_id": i,
                "created_at": "",
            }
            for i in range(10)
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = tickets
        mock_requests.get.return_value = mock_resp

        loader = FreshdeskLoader(domain="myco", api_key="k", limit=3)
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 3

    def test_aload(self) -> None:
        from synapsekit.loaders.freshdesk import FreshdeskLoader

        mock_requests = MagicMock()
        mock_resp_p1 = MagicMock()
        mock_resp_p1.json.return_value = [
            {
                "id": 1,
                "subject": "Async ticket",
                "description_text": "",
                "status": 2,
                "priority": 1,
                "requester_id": 1,
                "created_at": "",
            }
        ]
        mock_resp_p2 = MagicMock()
        mock_resp_p2.json.return_value = []
        mock_requests.get.side_effect = [mock_resp_p1, mock_resp_p2]

        loader = FreshdeskLoader(domain="myco", api_key="k")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1


# ---------------------------------------------------------------------------
# HackerNewsLoader
# ---------------------------------------------------------------------------


class TestHackerNewsLoader:
    def test_invalid_story_type(self) -> None:
        from synapsekit.loaders.hackernews import HackerNewsLoader

        with pytest.raises(ValueError, match="story_type must be one of"):
            HackerNewsLoader(story_type="random")

    def test_invalid_limit(self) -> None:
        from synapsekit.loaders.hackernews import HackerNewsLoader

        with pytest.raises(ValueError, match="limit must be greater than 0"):
            HackerNewsLoader(limit=0)

    def test_load_returns_documents(self) -> None:
        from synapsekit.loaders.hackernews import HackerNewsLoader

        loader = HackerNewsLoader(story_type="top", limit=2)
        story_ids = [1001, 1002]
        items = {
            1001: {
                "id": 1001,
                "title": "Story One",
                "url": "https://example.com/1",
                "score": 100,
                "by": "alice",
                "time": 1700000000,
            },
            1002: {
                "id": 1002,
                "title": "Story Two",
                "text": "Body text",
                "score": 50,
                "by": "bob",
                "time": 1700000001,
            },
        }

        def mock_fetch_ids(self_inner: HackerNewsLoader) -> list[int]:
            return story_ids

        def mock_fetch_item(self_inner: HackerNewsLoader, item_id: int) -> dict:
            return items[item_id]

        with patch.object(HackerNewsLoader, "_fetch_story_ids", mock_fetch_ids):
            with patch.object(HackerNewsLoader, "_fetch_item", mock_fetch_item):
                docs = loader.load()

        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert "Story One" in docs[0].text
        assert docs[0].metadata["source"] == "hackernews"
        assert docs[0].metadata["story_type"] == "top"
        assert docs[0].metadata["by"] == "alice"

    def test_limit_applied(self) -> None:
        from synapsekit.loaders.hackernews import HackerNewsLoader

        loader = HackerNewsLoader(limit=1)
        story_ids = [1, 2, 3]
        item = {"id": 1, "title": "One", "url": "http://x.com", "score": 10, "by": "u", "time": 0}

        with patch.object(HackerNewsLoader, "_fetch_story_ids", lambda s: story_ids):
            with patch.object(HackerNewsLoader, "_fetch_item", lambda s, i: item):
                docs = loader.load()

        assert len(docs) == 1

    def test_aload(self) -> None:
        from synapsekit.loaders.hackernews import HackerNewsLoader

        loader = HackerNewsLoader(limit=1)
        item = {
            "id": 42,
            "title": "Async HN",
            "url": "http://x.com",
            "score": 5,
            "by": "u",
            "time": 0,
        }

        with patch.object(HackerNewsLoader, "_fetch_story_ids", lambda s: [42]):
            with patch.object(HackerNewsLoader, "_fetch_item", lambda s, i: item):
                docs = asyncio.run(loader.aload())

        assert len(docs) == 1
        assert "Async HN" in docs[0].text


# ---------------------------------------------------------------------------
# RedditLoader
# ---------------------------------------------------------------------------


class TestRedditLoader:
    def test_missing_subreddit(self) -> None:
        from synapsekit.loaders.reddit import RedditLoader

        with pytest.raises(ValueError, match="subreddit must be provided"):
            RedditLoader(subreddit="")

    def test_invalid_sort(self) -> None:
        from synapsekit.loaders.reddit import RedditLoader

        with pytest.raises(ValueError, match="sort must be one of"):
            RedditLoader(subreddit="python", sort="latest")

    def test_invalid_limit(self) -> None:
        from synapsekit.loaders.reddit import RedditLoader

        with pytest.raises(ValueError, match="limit must be greater than 0"):
            RedditLoader(subreddit="python", limit=0)

    def test_load_public_api(self) -> None:
        from synapsekit.loaders.reddit import RedditLoader

        loader = RedditLoader(subreddit="python", limit=2)
        fake_response_data = json.dumps(
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "p1",
                                "title": "Post One",
                                "selftext": "Body",
                                "subreddit": "python",
                                "score": 100,
                                "url": "https://redd.it/p1",
                                "author": "user1",
                                "created_utc": 0,
                            }
                        },
                        {
                            "data": {
                                "id": "p2",
                                "title": "Post Two",
                                "selftext": "",
                                "subreddit": "python",
                                "score": 50,
                                "url": "https://redd.it/p2",
                                "author": "user2",
                                "created_utc": 0,
                            }
                        },
                    ]
                }
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_response_data

        with patch("synapsekit.loaders.reddit.urllib.request.urlopen", return_value=mock_resp):
            docs = loader.load()

        assert len(docs) == 2
        assert "Post One" in docs[0].text
        assert "Body" in docs[0].text
        assert docs[0].metadata["source"] == "reddit"
        assert docs[0].metadata["subreddit"] == "python"

    def test_aload(self) -> None:
        from synapsekit.loaders.reddit import RedditLoader

        loader = RedditLoader(subreddit="python", limit=1)
        fake_data = json.dumps(
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "a1",
                                "title": "Async Reddit",
                                "selftext": "",
                                "subreddit": "python",
                                "score": 10,
                                "url": "https://x.com",
                                "author": "u",
                                "created_utc": 0,
                            }
                        }
                    ]
                }
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = fake_data

        with patch("synapsekit.loaders.reddit.urllib.request.urlopen", return_value=mock_resp):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1


# ---------------------------------------------------------------------------
# TwitterLoader
# ---------------------------------------------------------------------------


class TestTwitterLoader:
    def test_missing_bearer_token(self) -> None:
        from synapsekit.loaders.twitter import TwitterLoader

        with pytest.raises(ValueError, match="bearer_token must be provided"):
            TwitterLoader(bearer_token="", query="python")

    def test_neither_query_nor_username(self) -> None:
        from synapsekit.loaders.twitter import TwitterLoader

        with pytest.raises(ValueError, match="Either query or username must be provided"):
            TwitterLoader(bearer_token="tok")

    def test_invalid_max_results(self) -> None:
        from synapsekit.loaders.twitter import TwitterLoader

        with pytest.raises(ValueError, match="max_results must be between"):
            TwitterLoader(bearer_token="tok", query="python", max_results=5)

    def test_import_error(self) -> None:
        from synapsekit.loaders.twitter import TwitterLoader

        loader = TwitterLoader(bearer_token="tok", query="python")
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(ImportError, match="requests required"):
                loader.load()

    def test_load_by_query(self) -> None:
        from synapsekit.loaders.twitter import TwitterLoader

        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "tweet1",
                    "text": "Hello #python",
                    "author_id": "user1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "public_metrics": {"retweet_count": 5},
                }
            ]
        }
        mock_requests.get.return_value = mock_resp

        loader = TwitterLoader(bearer_token="tok", query="python", max_results=10)
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].text == "Hello #python"
        assert docs[0].metadata["source"] == "twitter"
        assert docs[0].metadata["tweet_id"] == "tweet1"

    def test_load_by_username(self) -> None:
        from synapsekit.loaders.twitter import TwitterLoader

        mock_requests = MagicMock()
        user_resp = MagicMock()
        user_resp.json.return_value = {"data": {"id": "uid123"}}
        tweets_resp = MagicMock()
        tweets_resp.json.return_value = {
            "data": [
                {
                    "id": "t1",
                    "text": "Tweet from user",
                    "author_id": "uid123",
                    "created_at": "2024-01-01T00:00:00Z",
                    "public_metrics": {},
                }
            ]
        }
        mock_requests.get.side_effect = [user_resp, tweets_resp]

        loader = TwitterLoader(bearer_token="tok", username="testuser", max_results=10)
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Tweet from user"

    def test_aload(self) -> None:
        from synapsekit.loaders.twitter import TwitterLoader

        mock_requests = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "at1",
                    "text": "Async tweet",
                    "author_id": "u1",
                    "created_at": "",
                    "public_metrics": {},
                }
            ]
        }
        mock_requests.get.return_value = mock_resp

        loader = TwitterLoader(bearer_token="tok", query="python", max_results=10)
        with patch.dict("sys.modules", {"requests": mock_requests}):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1


# ---------------------------------------------------------------------------
# GoogleCalendarLoader
# ---------------------------------------------------------------------------


class TestGoogleCalendarLoader:
    def test_missing_credentials_path(self) -> None:
        from synapsekit.loaders.google_calendar import GoogleCalendarLoader

        with pytest.raises(ValueError, match="credentials_path must be provided"):
            GoogleCalendarLoader(credentials_path="")

    def _make_gcal_mocks(self, events_result: dict) -> tuple:
        """Return (mock_sa_mod, mock_googleapiclient_mod) with events_result wired up."""
        mock_service = MagicMock()
        mock_service.events.return_value.list.return_value.execute.return_value = events_result

        mock_build = MagicMock(return_value=mock_service)

        mock_sa_mod = MagicMock()
        mock_sa_mod.Credentials.from_service_account_file.return_value = MagicMock()

        mock_discovery_mod = MagicMock()
        mock_discovery_mod.build = mock_build

        mock_googleapiclient_discovery = mock_discovery_mod

        return mock_sa_mod, mock_googleapiclient_discovery

    def test_import_error(self) -> None:
        from synapsekit.loaders.google_calendar import GoogleCalendarLoader

        loader = GoogleCalendarLoader(credentials_path="/fake/path.json")
        with patch.dict(
            "sys.modules",
            {
                "google": None,
                "google.oauth2": None,
                "google.oauth2.service_account": None,
                "googleapiclient": None,
                "googleapiclient.discovery": None,
            },
        ):
            with pytest.raises(ImportError, match="google-api-python-client"):
                loader.load()

    def test_load_returns_documents(self) -> None:
        from synapsekit.loaders.google_calendar import GoogleCalendarLoader

        events_result = {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Team Meeting",
                    "description": "Weekly sync",
                    "start": {"dateTime": "2024-01-01T10:00:00Z"},
                    "end": {"dateTime": "2024-01-01T11:00:00Z"},
                    "location": "Office",
                    "htmlLink": "https://calendar.google.com/event/evt1",
                    "organizer": {"email": "boss@co.com"},
                }
            ]
        }
        mock_sa_mod, mock_discovery = self._make_gcal_mocks(events_result)

        loader = GoogleCalendarLoader(credentials_path="/fake/creds.json")

        google_pkg = MagicMock()
        google_oauth2 = MagicMock()
        google_oauth2.service_account = mock_sa_mod
        google_pkg.oauth2 = google_oauth2

        with patch.dict(
            "sys.modules",
            {
                "google": google_pkg,
                "google.oauth2": google_oauth2,
                "google.oauth2.service_account": mock_sa_mod,
                "googleapiclient": MagicMock(),
                "googleapiclient.discovery": mock_discovery,
            },
        ):
            docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Team Meeting" in docs[0].text
        assert "Weekly sync" in docs[0].text
        assert docs[0].metadata["source"] == "google_calendar"
        assert docs[0].metadata["event_id"] == "evt1"
        assert docs[0].metadata["organizer"] == "boss@co.com"

    def test_no_description(self) -> None:
        from synapsekit.loaders.google_calendar import GoogleCalendarLoader

        events_result = {
            "items": [
                {
                    "id": "evt2",
                    "summary": "No Desc Event",
                    "start": {"date": "2024-01-02"},
                    "end": {"date": "2024-01-02"},
                    "organizer": {},
                }
            ]
        }
        mock_sa_mod, mock_discovery = self._make_gcal_mocks(events_result)
        loader = GoogleCalendarLoader(credentials_path="/fake/creds.json")

        google_oauth2 = MagicMock()
        google_oauth2.service_account = mock_sa_mod

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.oauth2": google_oauth2,
                "google.oauth2.service_account": mock_sa_mod,
                "googleapiclient": MagicMock(),
                "googleapiclient.discovery": mock_discovery,
            },
        ):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "No Desc Event"

    def test_aload(self) -> None:
        from synapsekit.loaders.google_calendar import GoogleCalendarLoader

        events_result = {
            "items": [
                {
                    "id": "ae1",
                    "summary": "Async Event",
                    "start": {"dateTime": "2024-01-01T09:00:00Z"},
                    "end": {"dateTime": "2024-01-01T10:00:00Z"},
                    "organizer": {"email": "org@co.com"},
                }
            ]
        }
        mock_sa_mod, mock_discovery = self._make_gcal_mocks(events_result)
        loader = GoogleCalendarLoader(credentials_path="/fake/creds.json")

        google_oauth2 = MagicMock()
        google_oauth2.service_account = mock_sa_mod

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.oauth2": google_oauth2,
                "google.oauth2.service_account": mock_sa_mod,
                "googleapiclient": MagicMock(),
                "googleapiclient.discovery": mock_discovery,
            },
        ):
            docs = asyncio.run(loader.aload())

        assert len(docs) == 1
