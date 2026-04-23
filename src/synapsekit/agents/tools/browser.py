"""Browser automation tool using Playwright.

Provides selector-based web interaction for agents — navigation, reading,
form filling, and optional screenshot capture for multimodal models.

Requires: ``pip install synapsekit[browser]``
"""

from __future__ import annotations

import base64
import contextlib
from typing import Any
from urllib.parse import urlparse

from ..base import BaseTool, ToolResult

_MAX_HTML_LENGTH = 200_000
_MAX_TEXT_LENGTH = 100_000


class BrowserTool(BaseTool):
    """Interact with web pages using a real browser (Playwright).

    All interactions are **selector-based** (CSS / Playwright selectors).
    Coordinate-based clicking and vision-based detection are intentionally
    not supported.

    Parameters
    ----------
    headless : bool
        Run browser in headless mode (default ``True``).
    allowed_domains : list[str] | None
        If set, only these domains may be navigated to.
    blocked_domains : list[str] | None
        Always deny navigation to these domains.
    screenshot_on_action : bool
        Capture a screenshot after every action and include it in the result.
    timeout : float
        Default timeout in seconds for navigation / waits.
    max_pages : int | None
        Maximum number of navigations allowed per session.
    max_depth : int | None
        Maximum navigation depth allowed per session.
    allow_downloads : bool
        Whether file downloads are permitted.
    allow_javascript : bool
        Whether ``evaluate()`` is permitted.
    incognito : bool
        Use an isolated browser context.
    persistent_session : bool
        Reuse the browser instance across ``run()`` calls.
    """

    name = "browser"
    description = (
        "Interact with web pages using a real browser. "
        "Supports actions: navigate, back, forward, get_text, get_html, "
        "get_links, screenshot, click, fill, select, submit, "
        "wait_for, wait_for_navigation, evaluate, close. "
        "All interactions use CSS/text selectors — no coordinates."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "The browser action to perform. One of: navigate, back, forward, "
                    "get_text, get_html, get_links, screenshot, click, fill, select, "
                    "submit, wait_for, wait_for_navigation, evaluate, close."
                ),
                "enum": [
                    "navigate",
                    "back",
                    "forward",
                    "get_text",
                    "get_html",
                    "get_links",
                    "screenshot",
                    "click",
                    "fill",
                    "select",
                    "submit",
                    "wait_for",
                    "wait_for_navigation",
                    "evaluate",
                    "close",
                ],
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for 'navigate' action).",
            },
            "selector": {
                "type": "string",
                "description": "CSS or Playwright selector (for click/fill/select/submit/wait_for).",
            },
            "value": {
                "type": "string",
                "description": "Value for fill/select actions or JS code for evaluate.",
            },
            "timeout": {
                "type": "number",
                "description": "Override timeout in seconds for this action.",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        *,
        headless: bool = True,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        screenshot_on_action: bool = False,
        timeout: float = 30,
        max_pages: int | None = None,
        max_depth: int | None = None,
        allow_downloads: bool = False,
        allow_javascript: bool = True,
        incognito: bool = True,
        persistent_session: bool = False,
    ) -> None:
        self.headless = headless
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains or []
        self.screenshot_on_action = screenshot_on_action
        self.timeout = timeout
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.allow_downloads = allow_downloads
        self.allow_javascript = allow_javascript
        self.incognito = incognito
        self.persistent_session = persistent_session

        # Playwright objects — lazily initialised
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

        # Navigation counters for max_pages / max_depth enforcement
        self._page_count: int = 0
        self._depth: int = 0

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def _ensure_browser(self) -> None:
        """Start Playwright and open a page if not already running."""
        if self._page is not None:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError("playwright is required: pip install synapsekit[browser]") from None

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            accept_downloads=self.allow_downloads,
        )
        self._page = await self._context.new_page()

    async def _close(self) -> None:
        """Tear down browser resources."""
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self._page = None

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    # NOTE: Domain checks are enforced both before and after navigation to prevent redirect bypass.
    def _validate_url(self, url: str) -> None:
        """Raise ``ValueError`` if *url* violates domain / scheme rules."""
        parsed = urlparse(url)

        # Block dangerous schemes
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"URL scheme {parsed.scheme!r} is not allowed. Only http/https permitted."
            )

        host = (parsed.hostname or "").lower()
        if not host:
            raise ValueError("URL has no hostname.")

        # Allowed-domains whitelist
        if self.allowed_domains is not None and not any(
            host == d.lower() or host.endswith("." + d.lower()) for d in self.allowed_domains
        ):
            raise ValueError(f"Domain {host!r} is not in allowed_domains: {self.allowed_domains}")

        # Blocked-domains blacklist
        if any(host == d.lower() or host.endswith("." + d.lower()) for d in self.blocked_domains):
            raise ValueError(f"Domain {host!r} is blocked.")

    # ------------------------------------------------------------------
    # Screenshot helper
    # ------------------------------------------------------------------

    async def _maybe_screenshot(self) -> str | None:
        """Return a base64-encoded screenshot string if enabled, else ``None``."""
        if not self.screenshot_on_action or self._page is None:
            return None
        raw = await self._page.screenshot(type="png")
        return base64.b64encode(raw).decode()

    def _result_with_screenshot(self, output: str, screenshot_b64: str | None) -> ToolResult:
        """Build a ``ToolResult``, appending screenshot info when present."""
        if screenshot_b64:
            output += f"\n[screenshot attached: {len(screenshot_b64)} chars base64]"
        return ToolResult(output=output, error=None)

    # ------------------------------------------------------------------
    # Navigation actions
    # ------------------------------------------------------------------

    async def _navigate(self, url: str, timeout: float) -> ToolResult:
        # Enforce limits before navigation
        if self.max_pages is not None and self._page_count >= self.max_pages:
            raise RuntimeError(
                f"Max page limit reached ({self.max_pages}). No further navigations allowed."
            )
        if self.max_depth is not None and self._depth >= self.max_depth:
            raise RuntimeError(f"Max navigation depth exceeded ({self.max_depth}).")

        # Pre-navigation domain check
        self._validate_url(url)
        await self._page.goto(url, timeout=int(timeout * 1000), wait_until="domcontentloaded")

        # Post-navigation domain check — prevents redirect-based bypass
        current_url = self._page.url
        self._validate_url(current_url)

        self._page_count += 1
        self._depth += 1

        title = await self._page.title()
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(f"Navigated to {current_url} — title: {title}", shot)

    async def _back(self, timeout: float) -> ToolResult:
        await self._page.go_back(timeout=int(timeout * 1000))
        title = await self._page.title()
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(f"Went back — title: {title}", shot)

    async def _forward(self, timeout: float) -> ToolResult:
        await self._page.go_forward(timeout=int(timeout * 1000))
        title = await self._page.title()
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(f"Went forward — title: {title}", shot)

    # ------------------------------------------------------------------
    # Reading actions
    # ------------------------------------------------------------------

    async def _get_text(self) -> ToolResult:
        text = await self._page.inner_text("body")
        # Collapse runs of whitespace for cleaner output
        text = " ".join(text.split())
        if not text:
            text = "(page appears to be empty)"
        if len(text) > _MAX_TEXT_LENGTH:
            text = text[:_MAX_TEXT_LENGTH] + " ... [truncated]"
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(text, shot)

    async def _get_html(self) -> ToolResult:
        html = await self._page.content()
        if len(html) > _MAX_HTML_LENGTH:
            html = html[:_MAX_HTML_LENGTH] + "\n<!-- truncated -->"
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(html, shot)

    async def _get_links(self) -> ToolResult:
        links = await self._page.eval_on_selector_all(
            "a[href]",
            """els => els.map(e => ({
                text: e.innerText.trim().substring(0, 200),
                href: e.href
            }))""",
        )
        if not links:
            return ToolResult(output="No links found on page.")
        lines = [f"- [{link['text']}]({link['href']})" for link in links[:200]]
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot("\n".join(lines), shot)

    async def _screenshot(self) -> ToolResult:
        raw = await self._page.screenshot(type="png")
        b64 = base64.b64encode(raw).decode()
        return ToolResult(output=f"Screenshot captured ({len(raw)} bytes).\nbase64:{b64}")

    # ------------------------------------------------------------------
    # Interaction actions (selector-based only)
    # ------------------------------------------------------------------

    async def _click(self, selector: str, timeout: float) -> ToolResult:
        # Wait for element to be present before interacting (stability fix)
        await self._page.wait_for_selector(selector, timeout=int(timeout * 1000))
        try:
            await self._page.click(selector, timeout=int(timeout * 1000))
        except Exception as e:
            current_url = getattr(self._page, "url", "unknown")
            raise RuntimeError(
                f"Failed to click on selector '{selector}': {e} (URL: {current_url})"
            ) from e
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(f"Clicked: {selector}", shot)

    async def _fill(self, selector: str, value: str, timeout: float) -> ToolResult:
        # Wait for element to be present before interacting (stability fix)
        await self._page.wait_for_selector(selector, timeout=int(timeout * 1000))
        try:
            await self._page.fill(selector, value, timeout=int(timeout * 1000))
        except Exception as e:
            current_url = getattr(self._page, "url", "unknown")
            raise RuntimeError(
                f"Failed to fill selector '{selector}': {e} (URL: {current_url})"
            ) from e
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(f"Filled {selector} with value", shot)

    async def _select(self, selector: str, value: str, timeout: float) -> ToolResult:
        # Wait for element to be present before interacting (stability fix)
        await self._page.wait_for_selector(selector, timeout=int(timeout * 1000))
        try:
            await self._page.select_option(selector, value, timeout=int(timeout * 1000))
        except Exception as e:
            current_url = getattr(self._page, "url", "unknown")
            raise RuntimeError(
                f"Failed to select '{value}' in selector '{selector}': {e} (URL: {current_url})"
            ) from e
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(f"Selected '{value}' in {selector}", shot)

    async def _submit(self, selector: str, timeout: float) -> ToolResult:
        # Wait for element to be present before interacting (stability fix)
        await self._page.wait_for_selector(selector, timeout=int(timeout * 1000))
        try:
            await self._page.eval_on_selector(
                selector,
                "el => { if (el.form) el.form.submit(); else el.closest('form')?.submit(); }",
            )
        except Exception as e:
            current_url = getattr(self._page, "url", "unknown")
            raise RuntimeError(
                f"Failed to submit via selector '{selector}': {e} (URL: {current_url})"
            ) from e
        with contextlib.suppress(Exception):
            await self._page.wait_for_load_state("domcontentloaded", timeout=int(timeout * 1000))
        # Brief fallback pause for any JS-driven post-submit transitions
        await self._page.wait_for_timeout(500)
        shot = await self._maybe_screenshot()
        return self._result_with_screenshot(f"Submitted form via {selector}", shot)

    # ------------------------------------------------------------------
    # Waiting actions
    # ------------------------------------------------------------------

    async def _wait_for(self, selector: str, timeout: float) -> ToolResult:
        await self._page.wait_for_selector(selector, timeout=int(timeout * 1000))
        return ToolResult(output=f"Element matched: {selector}")

    async def _wait_for_navigation(self, timeout: float) -> ToolResult:
        await self._page.wait_for_load_state("domcontentloaded", timeout=int(timeout * 1000))
        title = await self._page.title()
        return ToolResult(output=f"Navigation complete — title: {title}")

    # ------------------------------------------------------------------
    # JavaScript execution
    # ------------------------------------------------------------------

    async def _evaluate(self, script: str) -> ToolResult:
        if not self.allow_javascript:
            return ToolResult(
                output="", error="JavaScript execution is disabled (allow_javascript=False)."
            )
        result = await self._page.evaluate(script)
        return ToolResult(output=str(result))

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute a browser action.

        Parameters (via *kwargs*)
        -------------------------
        action : str
            Required. The action to perform.
        url : str
            URL for ``navigate``.
        selector : str
            CSS/Playwright selector for interaction actions.
        value : str
            Value for ``fill``/``select`` or JS code for ``evaluate``.
        timeout : float
            Override the default timeout for this call.
        """
        action: str = kwargs.pop("action", "")
        if not action:
            return ToolResult(output="", error="No 'action' provided.")

        timeout = float(kwargs.pop("timeout", self.timeout))

        # Close is special — no browser needed
        if action == "close":
            await self._close()
            return ToolResult(output="Browser closed.")

        # Ensure browser is running for all other actions
        try:
            await self._ensure_browser()
        except ImportError as exc:
            return ToolResult(output="", error=str(exc))

        try:
            return await self._dispatch(action, timeout, **kwargs)
        except Exception as exc:
            current_url = getattr(self._page, "url", "unknown")
            error_msg = f"{type(exc).__name__}: {exc} (URL: {current_url})"
            return ToolResult(output="", error=error_msg)
        finally:
            if not self.persistent_session:
                await self._close()

    async def _dispatch(self, action: str, timeout: float, **kwargs: Any) -> ToolResult:
        """Route *action* to the appropriate handler."""
        if action == "navigate":
            url = kwargs.get("url", "")
            if not url:
                return ToolResult(output="", error="'url' is required for navigate.")
            return await self._navigate(url, timeout)

        if action == "back":
            return await self._back(timeout)

        if action == "forward":
            return await self._forward(timeout)

        if action == "get_text":
            return await self._get_text()

        if action == "get_html":
            return await self._get_html()

        if action == "get_links":
            return await self._get_links()

        if action == "screenshot":
            return await self._screenshot()

        if action in ("click", "fill", "select", "submit", "wait_for"):
            selector = kwargs.get("selector", "")
            if not selector:
                return ToolResult(output="", error=f"'selector' is required for {action}.")
            if action == "click":
                return await self._click(selector, timeout)
            if action == "fill":
                value = kwargs.get("value", "")
                return await self._fill(selector, value, timeout)
            if action == "select":
                value = kwargs.get("value", "")
                return await self._select(selector, value, timeout)
            if action == "submit":
                return await self._submit(selector, timeout)
            # wait_for
            return await self._wait_for(selector, timeout)

        if action == "wait_for_navigation":
            return await self._wait_for_navigation(timeout)

        if action == "evaluate":
            script = kwargs.get("value", "")
            if not script:
                return ToolResult(output="", error="'value' (JS code) is required for evaluate.")
            return await self._evaluate(script)

        return ToolResult(output="", error=f"Unknown action: {action!r}")
