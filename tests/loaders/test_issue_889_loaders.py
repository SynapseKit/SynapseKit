from __future__ import annotations

import asyncio
from typing import Any

import pytest

from synapsekit.loaders import Document


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200, headers: dict[str, str] | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.content = payload if isinstance(payload, bytes) else b""

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHTTPClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = iter(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return next(self.responses)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return next(self.responses)


class FakeGmailService:
    def __init__(self) -> None:
        self.message = {
            "id": "msg-1",
            "threadId": "thread-1",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Hello"},
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 00:00:00 +0000"},
                ],
                "body": {"data": "SGVsbG8gZnJvbSBHbWFpbA=="},
            },
        }

    def users(self) -> Any:
        service = self

        class Users:
            def messages(self) -> Any:
                class Messages:
                    def list(self, **kwargs: Any) -> Any:
                        class Request:
                            def execute(self) -> dict[str, Any]:
                                return {"messages": [{"id": "msg-1", "threadId": "thread-1"}]}

                        return Request()

                    def get(self, **kwargs: Any) -> Any:
                        class Request:
                            def execute(self) -> dict[str, Any]:
                                return service.message

                        return Request()

                return Messages()

            def threads(self) -> Any:
                class Threads:
                    def list(self, **kwargs: Any) -> Any:
                        class Request:
                            def execute(self) -> dict[str, Any]:
                                if kwargs.get("pageToken"):
                                    return {"threads": [{"id": "thread-2"}]}
                                return {
                                    "threads": [{"id": "thread-1"}],
                                    "nextPageToken": "next",
                                }

                        return Request()

                    def get(self, **kwargs: Any) -> Any:
                        class Request:
                            def execute(self) -> dict[str, Any]:
                                return {
                                    "id": kwargs["id"],
                                    "messages": [service.message],
                                }

                        return Request()

                return Threads()

        return Users()


def test_gmail_loader_converts_messages_and_metadata() -> None:
    from synapsekit.loaders.gmail import GmailLoader

    docs = GmailLoader(service=FakeGmailService(), max_results=1).load()

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "Hello\nHello from Gmail"
    assert docs[0].metadata["source"] == "gmail"
    assert docs[0].metadata["message_id"] == "msg-1"
    assert docs[0].metadata["subject"] == "Hello"


def test_gmail_loader_requires_credentials_or_service() -> None:
    from synapsekit.loaders.gmail import GmailLoader

    with pytest.raises(ValueError, match="credentials"):
        GmailLoader()


def test_gmail_loader_paginates_threads() -> None:
    from synapsekit.loaders.gmail import GmailLoader

    docs = GmailLoader(service=FakeGmailService(), mode="threads", max_results=2).load()

    assert [doc.metadata["thread_id"] for doc in docs] == ["thread-1", "thread-2"]


def test_linear_loader_reads_issue_nodes() -> None:
    from synapsekit.loaders.linear import LinearLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "data": {
                        "issues": {
                            "nodes": [
                                {
                                    "id": "i1",
                                    "identifier": "SK-1",
                                    "title": "Fix loader",
                                    "state": {"name": "Todo"},
                                }
                            ]
                        }
                    }
                }
            )
        ]
    )
    docs = LinearLoader(api_key="token", resource="issues", client=client).load()

    assert docs[0].text == 'id: i1\nidentifier: SK-1\ntitle: Fix loader\nstate: {"name": "Todo"}'
    assert docs[0].metadata["source"] == "linear"
    assert docs[0].metadata["id"] == "i1"
    assert client.calls[0][0] == "POST"


def test_asana_loader_follows_next_page_and_async_loads() -> None:
    from synapsekit.loaders.asana import AsanaLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "data": [{"gid": "1", "name": "First", "notes": "A"}],
                    "next_page": {"uri": "/tasks?page=2"},
                }
            ),
            FakeResponse(
                {"data": [{"gid": "2", "name": "Second", "notes": "B"}], "next_page": None}
            ),
        ]
    )
    loader = AsanaLoader(access_token="token", project_id="project", client=client)

    docs = asyncio.run(loader.aload())

    assert [doc.metadata["id"] for doc in docs] == ["1", "2"]
    assert "First" in docs[0].text
    assert client.calls[1][1].endswith("/tasks?page=2")


def test_asana_loader_requests_configured_fields() -> None:
    from synapsekit.loaders.asana import AsanaLoader

    client = FakeHTTPClient([FakeResponse({"data": [{"gid": "1", "name": "Task"}]})])
    AsanaLoader(
        access_token="token",
        project_id="project",
        text_fields=["name", "assignee.name"],
        metadata_fields=["completed"],
        client=client,
    ).load()

    opt_fields = client.calls[0][2]["params"]["opt_fields"].split(",")
    assert opt_fields[:3] == ["gid", "name", "notes"]
    assert {"assignee.name", "completed"} <= set(opt_fields)


def test_asana_loader_rejects_cross_origin_pagination() -> None:
    from synapsekit.loaders.asana import AsanaLoader

    client = FakeHTTPClient(
        [FakeResponse({"data": [], "next_page": {"uri": "https://attacker.test/tasks?page=2"}})]
    )

    with pytest.raises(RuntimeError, match="cross-origin"):
        AsanaLoader(access_token="token", project_id="project", client=client).load()


def test_clickup_loader_returns_task_documents() -> None:
    from synapsekit.loaders.clickup import ClickUpLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "tasks": [{"id": "t1", "name": "Ship it", "description": "Now"}],
                    "last_page": True,
                }
            )
        ]
    )
    docs = ClickUpLoader(api_token="token", list_id="list-1", client=client).load()

    assert docs[0].metadata["task_id"] == "t1"
    assert "Ship it" in docs[0].text
    assert client.calls[0][1].endswith("/list/list-1/task")


def test_clickup_loader_quotes_path_identifiers() -> None:
    from synapsekit.loaders.clickup import ClickUpLoader

    client = FakeHTTPClient([FakeResponse({"tasks": [], "last_page": True})])
    ClickUpLoader(api_token="token", list_id="../private", client=client).load()

    assert "/list/..%2Fprivate/task" in client.calls[0][1]


def test_other_loaders_quote_path_identifiers() -> None:
    from synapsekit.loaders.box import BoxLoader
    from synapsekit.loaders.figma import FigmaLoader
    from synapsekit.loaders.zoom import ZoomLoader

    box_client = FakeHTTPClient([FakeResponse({"entries": []})])
    BoxLoader(access_token="token", folder_id="../private", client=box_client).load()
    assert "/folders/..%2Fprivate/items" in box_client.calls[0][1]

    figma_client = FakeHTTPClient([FakeResponse({"document": {}})])
    FigmaLoader(access_token="token", file_key="../design", client=figma_client).load()
    assert "/files/..%2Fdesign" in figma_client.calls[0][1]

    zoom_client = FakeHTTPClient([FakeResponse({"meetings": [], "next_page_token": ""})])
    ZoomLoader(access_token="token", user_id="../me", client=zoom_client).load()
    assert "/users/..%2Fme/recordings" in zoom_client.calls[0][1]


def test_monday_loader_uses_custom_endpoint() -> None:
    from synapsekit.loaders.monday import MondayLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {"data": {"boards": [{"id": "b1", "items": [{"id": "i1", "name": "Roadmap"}]}]}}
            )
        ]
    )
    docs = MondayLoader(
        api_key="token", board_id="b1", endpoint_url="http://monday.test/graphql", client=client
    ).load()

    assert docs[0].metadata["source"] == "monday"
    assert docs[0].metadata["id"] == "i1"
    assert client.calls[0][1] == "http://monday.test/graphql"


def test_monday_loader_follows_items_page_cursor() -> None:
    from synapsekit.loaders.monday import MondayLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "data": {
                        "boards": [
                            {
                                "items_page": {
                                    "cursor": "cursor-1",
                                    "items": [{"id": "i1", "name": "One"}],
                                }
                            }
                        ]
                    }
                }
            ),
            FakeResponse(
                {
                    "data": {
                        "boards": [
                            {
                                "items_page": {
                                    "cursor": None,
                                    "items": [{"id": "i2", "name": "Two"}],
                                }
                            }
                        ]
                    }
                }
            ),
        ]
    )

    docs = MondayLoader(api_key="token", board_id="b1", limit=2, client=client).load()

    assert [doc.metadata["id"] for doc in docs] == ["i1", "i2"]
    assert client.calls[1][2]["json"]["variables"]["cursor"] == "cursor-1"


def test_monday_loader_returns_no_document_for_empty_board() -> None:
    from synapsekit.loaders.monday import MondayLoader

    client = FakeHTTPClient([FakeResponse({"data": {"boards": [{"id": "b1", "name": "Empty"}]}})])

    assert MondayLoader(api_key="token", board_id="b1", client=client).load() == []


def test_monday_loader_caps_default_page_size() -> None:
    from synapsekit.loaders.monday import MondayLoader

    client = FakeHTTPClient([FakeResponse({"data": {"boards": []}})])
    MondayLoader(api_key="token", board_id="b1", limit=1000, client=client).load()

    assert client.calls[0][2]["json"]["variables"]["limit"] == 500


def test_box_loader_loads_file_content() -> None:
    from synapsekit.loaders.box import BoxLoader

    client = FakeHTTPClient(
        [
            FakeResponse({"entries": [{"id": "f1", "name": "notes.txt", "type": "file"}]}),
            FakeResponse(b"Box notes"),
        ]
    )
    docs = BoxLoader(access_token="token", folder_id="0", client=client).load()

    assert docs[0].text == "Box notes"
    assert docs[0].metadata["file_id"] == "f1"
    assert docs[0].metadata["name"] == "notes.txt"


def test_box_loader_does_not_mangle_binary_content() -> None:
    """Regression test for #1030: binary files must not be decoded as UTF-8
    with errors="replace" (mojibake); they should get a placeholder + the
    raw bytes kept in metadata, matching S3Loader's pattern."""
    from synapsekit.loaders.box import BoxLoader

    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x01\x02\xff\xfe"
    client = FakeHTTPClient(
        [
            FakeResponse({"entries": [{"id": "f1", "name": "photo.png", "type": "file"}]}),
            FakeResponse(png_bytes, headers={"Content-Type": "image/png"}),
        ]
    )
    docs = BoxLoader(access_token="token", folder_id="0", client=client).load()

    assert docs[0].text == "[Binary file: image/png]"
    assert docs[0].metadata["raw_content"] == png_bytes
    assert docs[0].metadata["content_type"] == "image/png"


def test_box_loader_follows_folder_pagination() -> None:
    from synapsekit.loaders.box import BoxLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "entries": [{"id": "f1", "name": "one.txt", "type": "file"}],
                    "total_count": 2,
                }
            ),
            FakeResponse(
                {
                    "entries": [{"id": "f2", "name": "two.txt", "type": "file"}],
                    "total_count": 2,
                }
            ),
            FakeResponse(b"one"),
            FakeResponse(b"two"),
        ]
    )

    docs = BoxLoader(access_token="token", folder_id="0", client=client).load()

    assert [doc.metadata["file_id"] for doc in docs] == ["f1", "f2"]
    assert client.calls[1][2]["params"]["offset"] > 0


def test_figma_loader_extracts_file_context() -> None:
    from synapsekit.loaders.figma import FigmaLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "name": "Design",
                    "lastModified": "today",
                    "document": {"name": "Canvas", "children": []},
                }
            )
        ]
    )
    docs = FigmaLoader(access_token="token", file_key="file-1", client=client).load()

    assert len(docs) == 1
    assert "Design" in docs[0].text
    assert docs[0].metadata["file_key"] == "file-1"


def test_figma_loader_lifts_selected_node_metadata() -> None:
    from synapsekit.loaders.figma import FigmaLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "nodes": {
                        "node-1": {"document": {"id": "node-1", "name": "Hero", "type": "FRAME"}}
                    }
                }
            )
        ]
    )

    docs = FigmaLoader(
        access_token="token", file_key="file-1", node_ids=["node-1"], client=client
    ).load()

    assert docs[0].metadata["name"] == "Hero"
    assert docs[0].metadata["type"] == "FRAME"


def test_zoom_loader_converts_recordings() -> None:
    from synapsekit.loaders.zoom import ZoomLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "meetings": [{"uuid": "z1", "topic": "Planning", "start_time": "today"}],
                    "next_page_token": "",
                }
            )
        ]
    )
    docs = ZoomLoader(access_token="token", user_id="me", client=client).load()

    assert docs[0].metadata["meeting_id"] == "z1"
    assert "Planning" in docs[0].text


def test_zoom_loader_downloads_transcript_files() -> None:
    from synapsekit.loaders.zoom import ZoomLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "meetings": [
                        {
                            "uuid": "z1",
                            "topic": "Planning",
                            "recording_files": [
                                {
                                    "file_type": "TRANSCRIPT",
                                    "download_url": "http://zoom.test/transcript.vtt",
                                }
                            ],
                        }
                    ],
                    "next_page_token": "",
                }
            ),
            FakeResponse(b"WEBVTT\\n\\nPlanning notes"),
        ]
    )

    docs = ZoomLoader(
        access_token="token", user_id="me", endpoint_url="http://zoom.test/v2", client=client
    ).load()

    assert "Planning notes" in docs[0].text
    assert docs[0].metadata["transcript"] == "WEBVTT\\n\\nPlanning notes"
    assert client.calls[1][1] == "http://zoom.test/transcript.vtt"


def test_zoom_loader_does_not_fetch_untrusted_transcript_urls() -> None:
    from synapsekit.loaders.zoom import ZoomLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "meetings": [
                        {
                            "uuid": "z1",
                            "topic": "Planning",
                            "recording_files": [
                                {
                                    "file_type": "TRANSCRIPT",
                                    "download_url": "https://attacker.test/transcript.vtt",
                                }
                            ],
                        }
                    ],
                    "next_page_token": "",
                }
            )
        ]
    )

    ZoomLoader(access_token="token", user_id="me", client=client).load()

    assert len(client.calls) == 1


def test_zoom_loader_enforces_limit_before_transcript_downloads() -> None:
    from synapsekit.loaders.zoom import ZoomLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "meetings": [
                        {
                            "uuid": "z1",
                            "topic": "First",
                            "recording_files": [
                                {"file_type": "TRANSCRIPT", "download_url": "transcript.vtt"}
                            ],
                        },
                        {
                            "uuid": "z2",
                            "topic": "Second",
                            "recording_files": [
                                {"file_type": "TRANSCRIPT", "download_url": "second.vtt"}
                            ],
                        },
                    ],
                    "next_page_token": "",
                }
            ),
            FakeResponse(b"first transcript"),
        ]
    )

    docs = ZoomLoader(access_token="token", user_id="me", limit=1, client=client).load()

    assert len(docs) == 1
    assert len(client.calls) == 2


def test_gmail_thread_loader_preserves_message_metadata() -> None:
    from synapsekit.loaders.gmail import GmailLoader

    docs = GmailLoader(service=FakeGmailService(), mode="threads", max_results=1).load()

    assert docs[0].metadata["messages"][0]["from"] == "alice@example.com"
    assert docs[0].metadata["messages"][0]["date"] == "Mon, 1 Jan 2024 00:00:00 +0000"
    assert docs[0].metadata["label_ids"] == ["INBOX"]


def test_gmail_loader_combines_multipart_text_parts() -> None:
    from synapsekit.loaders.gmail import GmailLoader

    loader = GmailLoader(service=FakeGmailService(), max_results=1)

    assert (
        loader._body({"parts": [{"body": {"data": "Zmlyc3Q="}}, {"body": {"data": "c2Vjb25k"}}]})
        == "first\nsecond"
    )


def test_shopify_loader_converts_products() -> None:
    from synapsekit.loaders.shopify import ShopifyLoader

    client = FakeHTTPClient(
        [FakeResponse({"products": [{"id": 1, "title": "Widget", "body_html": "Useful"}]})]
    )
    docs = ShopifyLoader(
        shop_url="https://shop.test", access_token="token", resource="products", client=client
    ).load()

    assert docs[0].metadata["id"] == 1
    assert "Widget" in docs[0].text
    assert "/admin/api/" in client.calls[0][1]


def test_shopify_loader_normalizes_host_override() -> None:
    from synapsekit.loaders.shopify import ShopifyLoader

    client = FakeHTTPClient([FakeResponse({"products": [{"id": 1, "title": "Widget"}]})])
    ShopifyLoader(host="shop.test", access_token="token", client=client).load()

    assert client.calls[0][1] == "https://shop.test/admin/api/2024-10/products.json"


def test_shopify_loader_reads_single_resource_response() -> None:
    from synapsekit.loaders.shopify import ShopifyLoader

    client = FakeHTTPClient([FakeResponse({"product": {"id": 1, "title": "Widget"}})])
    docs = ShopifyLoader(
        shop_url="https://shop.test",
        access_token="token",
        resource="products/1",
        client=client,
    ).load()

    assert docs[0].metadata["id"] == 1
    assert docs[0].text.startswith("id: 1")


def test_shopify_loader_reads_nested_resource_response() -> None:
    from synapsekit.loaders.shopify import ShopifyLoader

    client = FakeHTTPClient([FakeResponse({"products": [{"id": 1, "title": "Widget"}]})])
    docs = ShopifyLoader(
        shop_url="https://shop.test",
        access_token="token",
        resource="collections/10/products",
        client=client,
    ).load()

    assert docs[0].metadata["id"] == 1


def test_databricks_loader_reads_rows() -> None:
    from synapsekit.loaders.databricks import DatabricksLoader

    class Cursor:
        description = [("id",), ("text",)]

        def execute(self, query: str) -> None:
            self.query = query

        def fetchall(self) -> list[tuple[int, str]]:
            return [(1, "lakehouse")]

        def close(self) -> None:
            pass

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            pass

    docs = DatabricksLoader(
        host="workspace.test",
        http_path="/sql",
        access_token="token",
        query="SELECT * FROM docs",
        client=Connection(),
    ).load()

    assert docs[0].text == "id: 1\ntext: lakehouse"
    assert docs[0].metadata["source"] == "databricks"


def test_databricks_loader_preserves_endpoint_port() -> None:
    from synapsekit.loaders.databricks import DatabricksLoader

    loader = DatabricksLoader(
        endpoint_url="https://localhost:8443",
        http_path="/sql",
        access_token="token",
        query="SELECT 1",
        client=object(),
    )
    assert loader._server_hostname == "localhost:8443"


def test_databricks_loader_protects_source_metadata_and_wraps_limit() -> None:
    from synapsekit.loaders.databricks import DatabricksLoader

    class Cursor:
        description = [("source",), ("text",)]

        def __init__(self) -> None:
            self.query = ""

        def execute(self, query: str) -> None:
            self.query = query

        def fetchall(self) -> list[tuple[str, str]]:
            return [("spoofed", "row")]

        def close(self) -> None:
            pass

    class Connection:
        def __init__(self) -> None:
            self.cursor_instance = Cursor()

        def cursor(self) -> Cursor:
            return self.cursor_instance

    connection = Connection()
    docs = DatabricksLoader(
        host="workspace.test",
        http_path="/sql",
        access_token="token",
        query="SELECT * FROM docs LIMIT 100 -- trailing comment",
        limit=2,
        client=connection,
    ).load()

    assert docs[0].metadata["source"] == "databricks"
    assert "SELECT * FROM (" in connection.cursor_instance.query
    assert connection.cursor_instance.query.endswith("LIMIT 2")


def test_linear_loader_applies_project_filter_to_issues() -> None:
    from synapsekit.loaders.linear import LinearLoader

    client = FakeHTTPClient([FakeResponse({"data": {"issues": []}})])
    LinearLoader(api_key="token", project_id="project-1", client=client).load()

    request = client.calls[0][2]["json"]
    assert request["variables"]["projectId"] == "project-1"
    assert "project: {id: {eq: $projectId}}" in request["query"]


def test_iceberg_loader_reads_injected_table() -> None:
    from synapsekit.loaders.iceberg import IcebergLoader

    class ArrowTable:
        def to_pylist(self) -> list[dict[str, str]]:
            return [{"title": "Iceberg", "body": "row"}]

    class Scan:
        def to_arrow(self) -> ArrowTable:
            return ArrowTable()

    class Table:
        def scan(self) -> Scan:
            return Scan()

    docs = IcebergLoader(table=Table(), identifier="catalog.docs").load()

    assert docs[0].metadata["source"] == "iceberg"
    assert "Iceberg" in docs[0].text


def test_iceberg_loader_applies_scan_limit() -> None:
    from synapsekit.loaders.iceberg import IcebergLoader

    class ArrowTable:
        def to_pylist(self) -> list[dict[str, str]]:
            return [{"text": "first"}]

    class Scan:
        def __init__(self) -> None:
            self.requested_limit: int | None = None

        def limit(self, value: int) -> Scan:
            self.requested_limit = value
            return self

        def to_arrow(self) -> ArrowTable:
            return ArrowTable()

    class Table:
        def __init__(self) -> None:
            self.scan_instance = Scan()

        def scan(self) -> Scan:
            return self.scan_instance

    table = Table()
    IcebergLoader(table=table, identifier="catalog.docs", limit=1).load()

    assert table.scan_instance.requested_limit == 1


def test_delta_loader_reads_injected_table() -> None:
    from synapsekit.loaders.delta import DeltaLakeLoader

    class Table:
        def to_pyarrow_table(self) -> Any:
            class ArrowTable:
                def to_pylist(self) -> list[dict[str, str]]:
                    return [{"text": "Delta row"}]

            return ArrowTable()

    docs = DeltaLakeLoader(uri="memory://table", table=Table()).load()

    assert docs[0].text == "text: Delta row"
    assert docs[0].metadata["source"] == "delta_lake"


def test_delta_loader_stops_dataset_scan_at_limit() -> None:
    from synapsekit.loaders.delta import DeltaLakeLoader

    class Batch:
        def __init__(self, rows: list[dict[str, str]]) -> None:
            self.rows = rows

        def to_pylist(self) -> list[dict[str, str]]:
            return self.rows

    class Scanner:
        def to_batches(self) -> list[Batch]:
            return [Batch([{"text": "first"}]), Batch([{"text": "second"}])]

    class Dataset:
        def scanner(self) -> Scanner:
            return Scanner()

    class Table:
        def to_pyarrow_dataset(self) -> Dataset:
            return Dataset()

        def to_pyarrow_table(self) -> Any:
            raise AssertionError("bounded dataset path should be used")

    docs = DeltaLakeLoader(uri="memory://table", table=Table(), limit=1).load()

    assert [doc.text for doc in docs] == ["text: first"]


def test_graphql_loader_extracts_nodes_and_passes_variables() -> None:
    from synapsekit.loaders.graphql import GraphQLLoader

    client = FakeHTTPClient(
        [FakeResponse({"data": {"users": {"nodes": [{"id": "u1", "name": "Ada"}]}}})]
    )
    docs = GraphQLLoader(
        endpoint_url="http://graphql.test",
        query="query Users($limit: Int!) { users { nodes { id name } } }",
        variables={"limit": 1},
        data_path="users",
        client=client,
    ).load()

    assert docs[0].metadata["id"] == "u1"
    assert "Ada" in docs[0].text
    assert client.calls[0][2]["json"]["variables"] == {"limit": 1}


def test_graphql_loader_follows_cursor_pages() -> None:
    from synapsekit.loaders.graphql import GraphQLLoader

    client = FakeHTTPClient(
        [
            FakeResponse(
                {
                    "data": {
                        "users": {
                            "nodes": [{"id": "u1", "name": "Ada"}],
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                        }
                    }
                }
            ),
            FakeResponse(
                {
                    "data": {
                        "users": {
                            "nodes": [{"id": "u2", "name": "Grace"}],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            ),
        ]
    )

    docs = GraphQLLoader(
        endpoint_url="http://graphql.test",
        query="query Users($after: String) { users { nodes { id name } pageInfo { hasNextPage endCursor } } }",
        page_info_path="users.pageInfo",
        limit=2,
        client=client,
    ).load()

    assert [doc.metadata["id"] for doc in docs] == ["u1", "u2"]
    assert client.calls[1][2]["json"]["variables"]["after"] == "cursor-1"


def test_graphql_loader_discovers_nested_nodes_without_data_path() -> None:
    from synapsekit.loaders.graphql import GraphQLLoader

    client = FakeHTTPClient(
        [FakeResponse({"data": {"users": {"nodes": [{"id": "u1", "name": "Ada"}]}}})]
    )

    docs = GraphQLLoader(
        endpoint_url="http://graphql.test",
        query="query { users { nodes { id name } } }",
        client=client,
    ).load()

    assert len(docs) == 1
    assert docs[0].metadata["id"] == "u1"


def test_kafka_loader_is_bounded_and_lifts_message_metadata() -> None:
    from synapsekit.loaders.kafka import KafkaLoader

    class Message:
        topic = "events"
        partition = 2
        offset = 7
        key = b"key"
        value = {"text": "Kafka event"}
        timestamp = 123
        headers = [("kind", b"created")]

    class Consumer:
        def __init__(self) -> None:
            self.calls = 0

        def poll(self, **kwargs: Any) -> dict[Any, list[Message]]:
            self.calls += 1
            return {"partition": [Message()]} if self.calls == 1 else {}

        def close(self) -> None:
            pass

    docs = KafkaLoader(
        bootstrap_servers="localhost:9092", topic="events", max_messages=1, consumer=Consumer()
    ).load()

    assert docs[0].text == '{"text": "Kafka event"}'
    assert docs[0].metadata["offset"] == 7
    assert docs[0].metadata["headers"] == {"kind": "created"}


def test_issue_889_loaders_are_lazy_public_exports() -> None:
    import synapsekit.loaders as loaders

    names = {
        "GmailLoader",
        "LinearLoader",
        "AsanaLoader",
        "ClickUpLoader",
        "MondayLoader",
        "BoxLoader",
        "FigmaLoader",
        "ZoomLoader",
        "ShopifyLoader",
        "DatabricksLoader",
        "IcebergLoader",
        "DeltaLakeLoader",
        "DeltaLoader",
        "GraphQLLoader",
        "KafkaLoader",
    }
    assert names <= set(loaders.__all__)
    assert names <= set(loaders._LOADERS)
