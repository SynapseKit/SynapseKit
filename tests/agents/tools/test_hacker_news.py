import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from synapsekit.agents.tools.hacker_news import HackerNewsTool

ROUTES: dict[str, object] = {
    "/v0/topstories.json": [1, 2, 3],
    "/v0/item/1.json": {"title": "Test Story", "url": "http://example.com", "score": 100},
    "/v0/item/2.json": {"title": "Second Story", "url": "http://example.org", "score": 50},
    "/v0/item/3.json": {"title": "Third Story", "url": "http://example.net", "score": 10},
    "/v0/item/123.json": {"id": 123, "title": "Specific Story", "type": "story"},
    "/v0/item/999.json": None,
}


class _FakeHNHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ROUTES:
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(ROUTES[path]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def hn_base_url() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _FakeHNHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v0"
    finally:
        server.shutdown()
        thread.join()


@pytest.mark.asyncio
async def test_hacker_news_tool_top_stories(hn_base_url: str) -> None:
    tool = HackerNewsTool(base_url=hn_base_url)
    result = await tool.run(action="top_stories")

    assert not result.error
    assert "Test Story" in result.output
    assert "Score: 100" in result.output
    assert "Second Story" in result.output
    assert "Third Story" in result.output


@pytest.mark.asyncio
async def test_hacker_news_tool_get_item(hn_base_url: str) -> None:
    tool = HackerNewsTool(base_url=hn_base_url)
    result = await tool.run(action="get_item", item_id=123)

    assert not result.error
    assert "Specific Story" in result.output


@pytest.mark.asyncio
async def test_hacker_news_tool_get_item_requires_id(hn_base_url: str) -> None:
    tool = HackerNewsTool(base_url=hn_base_url)
    result = await tool.run(action="get_item")

    assert result.error
    assert "item_id is required" in result.error


@pytest.mark.asyncio
async def test_hacker_news_tool_item_not_found(hn_base_url: str) -> None:
    tool = HackerNewsTool(base_url=hn_base_url)
    result = await tool.run(action="get_item", item_id=999)

    assert result.error
    assert "not found" in result.error.lower()
