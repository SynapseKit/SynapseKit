"""Production-grade tests for BrowserTool.

Covers: _ensure_browser ImportError branch, _validate_url edge cases,
_get_html truncation, exception branches in _click/_fill/_select/_submit,
run() ImportError passthrough, no-action guard.

All Playwright calls are mocked — no real browser is started.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.agents.tools.browser import _MAX_HTML_LENGTH, BrowserTool

# ---------------------------------------------------------------------------
# Helpers — build a BrowserTool with a pre-wired mock page
# ---------------------------------------------------------------------------


def _browser_with_mock_page(**kwargs) -> tuple[BrowserTool, MagicMock]:
    """Return a BrowserTool whose _page is already a Mock (skip Playwright launch)."""
    tool = BrowserTool(**kwargs)
    mock_page = MagicMock()
    mock_page.url = "https://example.com"
    # Override coroutine-typed page methods
    mock_page.goto = AsyncMock()
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.inner_text = AsyncMock(return_value="page text")
    mock_page.content = AsyncMock(return_value="<html></html>")
    mock_page.eval_on_selector_all = AsyncMock(return_value=[])
    mock_page.screenshot = AsyncMock(return_value=b"\x89PNG")
    mock_page.click = AsyncMock()
    mock_page.fill = AsyncMock()
    mock_page.select_option = AsyncMock()
    mock_page.eval_on_selector = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.go_back = AsyncMock()
    mock_page.go_forward = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="42")
    tool._page = mock_page
    return tool, mock_page


# ---------------------------------------------------------------------------
# 1. _ensure_browser — ImportError branch (playwright not installed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_browser_playwright_import_error():
    tool = BrowserTool()
    # Simulate playwright not installed by injecting None into sys.modules
    mock_modules = {
        "playwright": None,
        "playwright.async_api": None,
    }
    with patch.dict(sys.modules, mock_modules):
        with pytest.raises(ImportError, match="playwright is required"):
            await tool._ensure_browser()


@pytest.mark.asyncio
async def test_run_playwright_import_error_returns_error_result():
    """run() catches ImportError from _ensure_browser and returns ToolResult with error."""
    tool = BrowserTool()
    with patch.object(
        tool,
        "_ensure_browser",
        side_effect=ImportError("playwright is required: pip install synapsekit[browser]"),
    ):
        result = await tool.run(action="get_text")

    assert result.error is not None
    assert "playwright" in result.error.lower()


# ---------------------------------------------------------------------------
# 2. _validate_url edge cases
# ---------------------------------------------------------------------------


def test_validate_url_no_hostname():
    tool = BrowserTool()
    with pytest.raises(ValueError, match="no hostname"):
        tool._validate_url("http://")


def test_validate_url_non_http_scheme():
    tool = BrowserTool()
    with pytest.raises(ValueError, match="not allowed"):
        tool._validate_url("ftp://example.com/file")


def test_validate_url_file_scheme():
    tool = BrowserTool()
    with pytest.raises(ValueError, match="not allowed"):
        tool._validate_url("file:///etc/passwd")


def test_validate_url_blocked_domain():
    tool = BrowserTool(blocked_domains=["evil.com"])
    with pytest.raises(ValueError, match="blocked"):
        tool._validate_url("https://evil.com/page")


def test_validate_url_blocked_subdomain():
    tool = BrowserTool(blocked_domains=["evil.com"])
    with pytest.raises(ValueError, match="blocked"):
        tool._validate_url("https://sub.evil.com/page")


def test_validate_url_not_in_allowed_domains():
    tool = BrowserTool(allowed_domains=["trusted.com"])
    with pytest.raises(ValueError, match="not in allowed_domains"):
        tool._validate_url("https://untrusted.com/page")


def test_validate_url_allowed_domain_passes():
    tool = BrowserTool(allowed_domains=["trusted.com"])
    tool._validate_url("https://trusted.com/page")  # should not raise


def test_validate_url_allowed_subdomain_passes():
    tool = BrowserTool(allowed_domains=["trusted.com"])
    tool._validate_url("https://www.trusted.com/path")  # subdomain should pass


# ---------------------------------------------------------------------------
# 3. _get_html truncation (line 300)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_html_truncates_long_content():
    tool, mock_page = _browser_with_mock_page()
    long_html = "x" * (_MAX_HTML_LENGTH + 500)
    mock_page.content = AsyncMock(return_value=long_html)

    result = await tool._get_html()
    assert "truncated" in result.output
    assert len(result.output) <= _MAX_HTML_LENGTH + 50  # small overhead for suffix


@pytest.mark.asyncio
async def test_get_html_short_content_not_truncated():
    tool, mock_page = _browser_with_mock_page()
    mock_page.content = AsyncMock(return_value="<html><body>short</body></html>")

    result = await tool._get_html()
    assert "truncated" not in result.output
    assert "short" in result.output


# ---------------------------------------------------------------------------
# 4. Exception branches in interaction methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_click_exception_raises_runtime_error():
    tool, mock_page = _browser_with_mock_page()
    mock_page.click = AsyncMock(side_effect=Exception("element not found"))

    with pytest.raises(RuntimeError, match="Failed to click"):
        await tool._click("#bad-selector", timeout=5.0)


@pytest.mark.asyncio
async def test_fill_exception_raises_runtime_error():
    tool, mock_page = _browser_with_mock_page()
    mock_page.fill = AsyncMock(side_effect=Exception("stale element"))

    with pytest.raises(RuntimeError, match="Failed to fill"):
        await tool._fill("#input", "value", timeout=5.0)


@pytest.mark.asyncio
async def test_select_exception_raises_runtime_error():
    tool, mock_page = _browser_with_mock_page()
    mock_page.select_option = AsyncMock(side_effect=Exception("option not found"))

    with pytest.raises(RuntimeError, match="Failed to select"):
        await tool._select("#dropdown", "opt", timeout=5.0)


@pytest.mark.asyncio
async def test_submit_exception_raises_runtime_error():
    tool, mock_page = _browser_with_mock_page()
    mock_page.eval_on_selector = AsyncMock(side_effect=Exception("form not found"))
    # Wait for load state should suppress; wait_for_timeout is called after
    mock_page.wait_for_load_state = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()

    with pytest.raises(RuntimeError, match="Failed to submit"):
        await tool._submit("#form-btn", timeout=5.0)


# ---------------------------------------------------------------------------
# 5. run() — no action guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_no_action_returns_error():
    tool = BrowserTool()
    result = await tool.run()
    assert result.error is not None
    assert "action" in result.error.lower()


# ---------------------------------------------------------------------------
# 6. run() — close action (no browser needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_close_closes_browser():
    tool, _mock_page = _browser_with_mock_page()
    mock_context = AsyncMock()
    mock_browser = AsyncMock()
    mock_playwright = AsyncMock()
    tool._context = mock_context
    tool._browser = mock_browser
    tool._playwright = mock_playwright

    result = await tool.run(action="close")
    assert result.output == "Browser closed."
    mock_context.close.assert_called_once()
    mock_browser.close.assert_called_once()
    mock_playwright.stop.assert_called_once()


# ---------------------------------------------------------------------------
# 7. run() — exception in dispatch wraps as ToolResult with error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_dispatch_exception_captured():
    tool, mock_page = _browser_with_mock_page()
    mock_page.inner_text = AsyncMock(side_effect=RuntimeError("page crashed"))

    result = await tool.run(action="get_text")
    assert result.error is not None
    assert "page crashed" in result.error or "RuntimeError" in result.error


# ---------------------------------------------------------------------------
# 8. max_pages / max_depth enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_navigate_max_pages_exceeded():
    tool, _mock_page = _browser_with_mock_page(max_pages=2)
    tool._page_count = 2  # already at limit

    with pytest.raises(RuntimeError, match="Max page limit"):
        await tool._navigate("https://example.com", timeout=5.0)


@pytest.mark.asyncio
async def test_navigate_max_depth_exceeded():
    tool, _mock_page = _browser_with_mock_page(max_depth=1)
    tool._depth = 1  # already at limit

    with pytest.raises(RuntimeError, match="Max navigation depth"):
        await tool._navigate("https://example.com", timeout=5.0)


# ---------------------------------------------------------------------------
# 9. JavaScript disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_disabled_returns_error():
    tool, _ = _browser_with_mock_page(allow_javascript=False)
    result = await tool._evaluate("1 + 1")
    assert result.error is not None
    assert "disabled" in result.error.lower()


@pytest.mark.asyncio
async def test_evaluate_enabled_returns_result():
    tool, mock_page = _browser_with_mock_page(allow_javascript=True)
    mock_page.evaluate = AsyncMock(return_value=42)
    result = await tool._evaluate("1 + 41")
    assert result.error is None
    assert "42" in result.output
