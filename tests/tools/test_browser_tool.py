"""Tests for BrowserTool — unit, integration (mocked), security, and negative cases."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.agents.tools.browser import BrowserTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_page(
    *,
    title: str = "Test Page",
    inner_text: str = "Hello world",
    content: str = "<html><body>Hello</body></html>",
    links: list | None = None,
    screenshot_bytes: bytes = b"\x89PNG_FAKE",
) -> AsyncMock:
    """Create a mock Playwright page with sensible defaults."""
    page = AsyncMock()
    page.title.return_value = title
    page.inner_text.return_value = inner_text
    page.content.return_value = content
    page.screenshot.return_value = screenshot_bytes
    page.eval_on_selector_all.return_value = (
        links if links is not None else [{"text": "Example", "href": "https://example.com"}]
    )
    page.evaluate.return_value = 42
    page.goto = AsyncMock()
    page.go_back = AsyncMock()
    page.go_forward = AsyncMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.select_option = AsyncMock()
    page.eval_on_selector = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    return page


def _patch_browser(tool: BrowserTool, page: AsyncMock | None = None) -> None:
    """Inject mock objects so Playwright is never actually launched."""
    page = page or _mock_page()
    tool._playwright = MagicMock()
    tool._browser = AsyncMock()
    tool._context = AsyncMock()
    tool._page = page


# ---------------------------------------------------------------------------
# Unit Tests — Navigation
# ---------------------------------------------------------------------------


class TestNavigate:
    """navigate() calls Playwright correctly."""

    @pytest.mark.asyncio
    async def test_navigate_calls_goto(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"  # set post-navigation URL for redirect check
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://example.com")

        page.goto.assert_awaited_once()
        call_args = page.goto.call_args
        assert call_args[0][0] == "https://example.com"
        assert "Navigated to" in result.output
        assert "Test Page" in result.output
        assert result.error is None

    @pytest.mark.asyncio
    async def test_navigate_missing_url(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = ""
        _patch_browser(tool, page)

        result = await tool.run(action="navigate")

        assert result.is_error
        assert "url" in result.error.lower()

    @pytest.mark.asyncio
    async def test_back(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="back")

        page.go_back.assert_awaited_once()
        assert "back" in result.output.lower()

    @pytest.mark.asyncio
    async def test_forward(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="forward")

        page.go_forward.assert_awaited_once()
        assert "forward" in result.output.lower()


# ---------------------------------------------------------------------------
# Unit Tests — Domain restrictions
# ---------------------------------------------------------------------------


class TestDomainRestrictions:
    """Domain restrictions enforced correctly."""

    @pytest.mark.asyncio
    async def test_allowed_domains_blocks_unlisted(self):
        tool = BrowserTool(
            allowed_domains=["example.com"],
            persistent_session=True,
        )
        page = _mock_page()
        page.url = "https://evil.com"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://evil.com")

        assert result.is_error
        assert "not in allowed_domains" in result.error

    @pytest.mark.asyncio
    async def test_allowed_domains_permits_listed(self):
        tool = BrowserTool(
            allowed_domains=["example.com"],
            persistent_session=True,
        )
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://example.com")

        assert not result.is_error
        page.goto.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_allowed_domains_permits_subdomain(self):
        tool = BrowserTool(
            allowed_domains=["example.com"],
            persistent_session=True,
        )
        page = _mock_page()
        page.url = "https://sub.example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://sub.example.com")

        assert not result.is_error

    @pytest.mark.asyncio
    async def test_blocked_domains_rejects(self):
        tool = BrowserTool(
            blocked_domains=["malware.com"],
            persistent_session=True,
        )
        page = _mock_page()
        page.url = "https://malware.com"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://malware.com")

        assert result.is_error
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocked_subdomain_rejects(self):
        tool = BrowserTool(
            blocked_domains=["malware.com"],
            persistent_session=True,
        )
        page = _mock_page()
        page.url = "https://sub.malware.com/page"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://sub.malware.com/page")

        assert result.is_error
        assert "blocked" in result.error.lower()


# ---------------------------------------------------------------------------
# Unit Tests — Reading
# ---------------------------------------------------------------------------


class TestReading:
    """Reading actions return expected content."""

    @pytest.mark.asyncio
    async def test_get_text_returns_visible_content(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page(inner_text="  Visible    text   here  ")
        _patch_browser(tool, page)

        result = await tool.run(action="get_text")

        page.inner_text.assert_awaited_once_with("body")
        # whitespace should be normalised
        assert result.output == "Visible text here"
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_get_text_truncates_large_content(self):
        tool = BrowserTool(persistent_session=True)
        # After whitespace collapse, each char stays as a single word
        # Use a string that survives join().split() unchanged
        huge = "x" * 200_000
        page = _mock_page(inner_text=huge)
        _patch_browser(tool, page)

        result = await tool.run(action="get_text")

        assert "[truncated]" in result.output
        assert len(result.output) < 200_100  # some slack for the suffix

    @pytest.mark.asyncio
    async def test_get_html(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page(content="<html><body>Hi</body></html>")
        _patch_browser(tool, page)

        result = await tool.run(action="get_html")

        assert "<html>" in result.output

    @pytest.mark.asyncio
    async def test_get_links(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page(
            links=[
                {"text": "Google", "href": "https://google.com"},
                {"text": "GitHub", "href": "https://github.com"},
            ]
        )
        _patch_browser(tool, page)

        result = await tool.run(action="get_links")

        assert "Google" in result.output
        assert "GitHub" in result.output

    @pytest.mark.asyncio
    async def test_get_links_empty(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page(links=[])
        _patch_browser(tool, page)

        result = await tool.run(action="get_links")

        assert "No links" in result.output


# ---------------------------------------------------------------------------
# Unit Tests — Interaction (selector-based)
# ---------------------------------------------------------------------------


class TestInteraction:
    """Selector-based interaction actions work correctly."""

    @pytest.mark.asyncio
    async def test_click_with_selector(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="click", selector="button.submit")

        # wait_for_selector must be called before click (stability fix)
        page.wait_for_selector.assert_awaited()
        page.click.assert_awaited_once()
        assert "button.submit" in page.click.call_args[0][0]
        assert "Clicked" in result.output

    @pytest.mark.asyncio
    async def test_click_missing_selector(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = ""
        _patch_browser(tool, page)

        result = await tool.run(action="click")

        assert result.is_error
        assert "selector" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fill(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(
            action="fill",
            selector="input[name=email]",
            value="test@example.com",
        )

        # wait_for_selector called before fill
        page.wait_for_selector.assert_awaited()
        page.fill.assert_awaited_once_with("input[name=email]", "test@example.com", timeout=30000)
        assert "Filled" in result.output

    @pytest.mark.asyncio
    async def test_select(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="select", selector="select#country", value="US")

        page.wait_for_selector.assert_awaited()
        page.select_option.assert_awaited_once()
        assert "Selected" in result.output

    @pytest.mark.asyncio
    async def test_submit(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="submit", selector="form#login")

        page.wait_for_selector.assert_awaited()
        page.eval_on_selector.assert_awaited_once()
        # Fallback pause must be called
        page.wait_for_timeout.assert_awaited_once_with(500)
        assert "Submitted" in result.output


# ---------------------------------------------------------------------------
# Unit Tests — Screenshot
# ---------------------------------------------------------------------------


class TestScreenshot:
    """screenshot() returns bytes / base64."""

    @pytest.mark.asyncio
    async def test_screenshot_returns_base64(self):
        tool = BrowserTool(persistent_session=True)
        raw = b"\x89PNG_FAKE_SCREENSHOT"
        page = _mock_page(screenshot_bytes=raw)
        _patch_browser(tool, page)

        result = await tool.run(action="screenshot")

        page.screenshot.assert_awaited_once()
        expected_b64 = base64.b64encode(raw).decode()
        assert expected_b64 in result.output
        assert str(len(raw)) in result.output


# ---------------------------------------------------------------------------
# Unit Tests — JavaScript evaluation
# ---------------------------------------------------------------------------


class TestEvaluate:
    """evaluate() works correctly with JS enabled/disabled."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self):
        tool = BrowserTool(allow_javascript=True, persistent_session=True)
        page = _mock_page()
        page.evaluate.return_value = 42
        _patch_browser(tool, page)

        result = await tool.run(action="evaluate", value="1 + 1")

        page.evaluate.assert_awaited_once_with("1 + 1")
        assert "42" in result.output

    @pytest.mark.asyncio
    async def test_evaluate_disabled(self):
        tool = BrowserTool(allow_javascript=False, persistent_session=True)
        _patch_browser(tool)

        result = await tool.run(action="evaluate", value="alert(1)")

        assert result.is_error
        assert "disabled" in result.error.lower()


# ---------------------------------------------------------------------------
# Unit Tests — Waiting
# ---------------------------------------------------------------------------


class TestWaiting:
    """wait_for / wait_for_navigation work."""

    @pytest.mark.asyncio
    async def test_wait_for_selector(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        _patch_browser(tool, page)

        result = await tool.run(action="wait_for", selector="div.loaded")

        page.wait_for_selector.assert_awaited_once()
        assert "div.loaded" in result.output

    @pytest.mark.asyncio
    async def test_wait_for_navigation(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        _patch_browser(tool, page)

        result = await tool.run(action="wait_for_navigation")

        page.wait_for_load_state.assert_awaited_once()
        assert "complete" in result.output.lower() or "title" in result.output.lower()


# ---------------------------------------------------------------------------
# Integration Tests (Mocked) — Agent Workflow
# ---------------------------------------------------------------------------


class TestIntegrationMocked:
    """Simulated agent workflows with mocked Playwright."""

    @pytest.mark.asyncio
    async def test_navigate_and_extract_content(self):
        """Agent navigates to a page, then reads text."""
        tool = BrowserTool(persistent_session=True)
        page = _mock_page(
            title="Docs Page",
            inner_text="Welcome to the documentation.",
        )
        page.url = "https://docs.example.com"
        _patch_browser(tool, page)

        nav = await tool.run(action="navigate", url="https://docs.example.com")
        assert "Docs Page" in nav.output

        text = await tool.run(action="get_text")
        assert "Welcome to the documentation." in text.output

    @pytest.mark.asyncio
    async def test_screenshot_on_action_attaches_image(self):
        """screenshot_on_action=True attaches screenshot to action result."""
        tool = BrowserTool(screenshot_on_action=True, persistent_session=True)
        raw = b"\x89PNG_SHOT"
        page = _mock_page(screenshot_bytes=raw)
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://example.com")

        assert "screenshot attached" in result.output
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_screenshot_off_by_default(self):
        """Without screenshot_on_action, no screenshot is attached."""
        tool = BrowserTool(screenshot_on_action=False, persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://example.com")

        assert "screenshot" not in result.output.lower()


# ---------------------------------------------------------------------------
# Security Tests
# ---------------------------------------------------------------------------


class TestSecurity:
    """Safety controls block dangerous URLs and domains."""

    @pytest.mark.asyncio
    async def test_block_file_scheme(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = ""
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="file:///etc/passwd")

        assert result.is_error
        assert "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_block_javascript_scheme(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = ""
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="javascript:alert(1)")

        assert result.is_error

    @pytest.mark.asyncio
    async def test_block_ftp_scheme(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = ""
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="ftp://files.example.com/data")

        assert result.is_error
        assert "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blocked_domain_prevents_navigation(self):
        tool = BrowserTool(
            blocked_domains=["danger.com"],
            persistent_session=True,
        )
        page = _mock_page()
        page.url = "https://danger.com/path"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://danger.com/path")

        assert result.is_error
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_allowed_domains_rejects_unlisted(self):
        tool = BrowserTool(
            allowed_domains=["safe.com"],
            persistent_session=True,
        )
        page = _mock_page()
        page.url = "https://unsafe.com"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://unsafe.com")

        assert result.is_error
        assert "not in allowed_domains" in result.error


# ---------------------------------------------------------------------------
# Negative Tests
# ---------------------------------------------------------------------------


class TestNegative:
    """Error conditions produce clear, actionable messages."""

    @pytest.mark.asyncio
    async def test_selector_not_found_clear_error(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        # wait_for_selector raises when element absent
        page.wait_for_selector.side_effect = Exception("Timeout: selector 'div.nope' not found")
        _patch_browser(tool, page)

        result = await tool.run(action="click", selector="div.nope")

        assert result.is_error
        assert "div.nope" in result.error

    @pytest.mark.asyncio
    async def test_navigation_timeout(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = ""
        page.goto.side_effect = Exception("Timeout 30000ms exceeded")
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://slow.example.com")

        assert result.is_error
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = ""
        _patch_browser(tool, page)

        result = await tool.run(action="teleport")

        assert result.is_error
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_no_action(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = ""
        _patch_browser(tool, page)

        result = await tool.run()

        assert result.is_error
        assert "action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_evaluate_missing_script(self):
        tool = BrowserTool(allow_javascript=True, persistent_session=True)
        page = _mock_page()
        page.url = ""
        _patch_browser(tool, page)

        result = await tool.run(action="evaluate")

        assert result.is_error
        assert "value" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_close(self):
        tool = BrowserTool(persistent_session=True)
        ctx = AsyncMock()
        browser = AsyncMock()
        pw = MagicMock()
        pw.stop = AsyncMock()
        tool._playwright = pw
        tool._browser = browser
        tool._context = ctx
        tool._page = _mock_page()

        result = await tool.run(action="close")

        assert "closed" in result.output.lower()
        ctx.close.assert_awaited_once()
        browser.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Hardening Tests — New for production fixes
# ---------------------------------------------------------------------------


class TestRedirectBypassPrevention:
    """Post-navigation domain check blocks redirect-based bypasses."""

    @pytest.mark.asyncio
    async def test_redirect_to_blocked_domain_is_caught(self):
        """URL passes pre-check but redirects to a blocked domain."""
        tool = BrowserTool(
            blocked_domains=["evil.com"],
            persistent_session=True,
        )
        page = _mock_page()
        # Simulates a redirect: goto succeeds but page.url is now evil.com
        page.url = "https://evil.com/landing"
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://safe-redirect.com")

        assert result.is_error
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_redirect_outside_allowed_domains_is_caught(self):
        """URL is in allowed_domains but redirect lands outside."""
        tool = BrowserTool(
            allowed_domains=["example.com"],
            persistent_session=True,
        )
        page = _mock_page()
        page.url = "https://other.com/page"  # redirect target
        _patch_browser(tool, page)

        result = await tool.run(action="navigate", url="https://example.com")

        assert result.is_error
        assert "not in allowed_domains" in result.error


class TestLimits:
    """max_pages and max_depth enforcement."""

    @pytest.mark.asyncio
    async def test_max_pages_blocks_after_limit(self):
        tool = BrowserTool(max_pages=1, persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        # First navigation should succeed
        r1 = await tool.run(action="navigate", url="https://example.com")
        assert not r1.is_error

        # Second navigation should be blocked
        r2 = await tool.run(action="navigate", url="https://example.com/page2")
        assert r2.is_error
        assert "max page limit" in r2.error.lower()

    @pytest.mark.asyncio
    async def test_max_depth_blocks_after_limit(self):
        tool = BrowserTool(max_depth=2, persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com"
        _patch_browser(tool, page)

        await tool.run(action="navigate", url="https://example.com")
        await tool.run(action="navigate", url="https://example.com/2")

        r3 = await tool.run(action="navigate", url="https://example.com/3")
        assert r3.is_error
        assert "depth" in r3.error.lower()


class TestErrorsIncludeURL:
    """Errors in run() include current URL for context."""

    @pytest.mark.asyncio
    async def test_interaction_error_includes_url(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        page.url = "https://example.com/form"
        page.wait_for_selector.side_effect = Exception("Element not found")
        _patch_browser(tool, page)

        result = await tool.run(action="click", selector="#missing")

        assert result.is_error
        # URL should appear somewhere in the error chain
        assert "example.com" in result.error


class TestGetTextEmpty:
    """get_text() handles empty pages gracefully."""

    @pytest.mark.asyncio
    async def test_empty_page_returns_placeholder(self):
        tool = BrowserTool(persistent_session=True)
        page = _mock_page(inner_text="   \n\t  ")
        _patch_browser(tool, page)

        result = await tool.run(action="get_text")

        assert not result.is_error
        assert "empty" in result.output.lower()


class TestSessionManagement:
    """persistent_session controls browser lifecycle."""

    @pytest.mark.asyncio
    async def test_non_persistent_closes_after_run(self):
        """Default (non-persistent) session tears down after each run()."""
        tool = BrowserTool(persistent_session=False)
        page = _mock_page()
        ctx = AsyncMock()
        browser = AsyncMock()
        pw = MagicMock()
        pw.stop = AsyncMock()
        tool._playwright = pw
        tool._browser = browser
        tool._context = ctx
        tool._page = page

        await tool.run(action="get_text")

        # Should have cleaned up
        ctx.close.assert_awaited_once()
        browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persistent_keeps_browser(self):
        """persistent_session=True keeps the browser alive after run()."""
        tool = BrowserTool(persistent_session=True)
        page = _mock_page()
        ctx = AsyncMock()
        browser = AsyncMock()
        tool._playwright = MagicMock()
        tool._browser = browser
        tool._context = ctx
        tool._page = page

        await tool.run(action="get_text")

        ctx.close.assert_not_awaited()
        browser.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestToolMeta:
    """BrowserTool exposes correct metadata for agent frameworks."""

    def test_name_and_description(self):
        tool = BrowserTool()
        assert tool.name == "browser"
        assert "web" in tool.description.lower() or "browser" in tool.description.lower()

    def test_schema(self):
        tool = BrowserTool()
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "browser"
        assert "action" in schema["function"]["parameters"]["properties"]

    def test_anthropic_schema(self):
        tool = BrowserTool()
        schema = tool.anthropic_schema()
        assert schema["name"] == "browser"
        assert "action" in schema["input_schema"]["properties"]
