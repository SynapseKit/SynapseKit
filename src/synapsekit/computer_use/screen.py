from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from .types import ComputerAction, ComputerActionType, ComputerObservation, ScreenProvider


class BrowserScreenProvider:
    """Screen provider backed by the existing Playwright ``BrowserTool``."""

    def __init__(self, browser: Any | None = None, **browser_kwargs: Any) -> None:
        if browser is None:
            from synapsekit.agents.tools.browser import BrowserTool

            browser = BrowserTool(**browser_kwargs)
        self.browser = browser

    async def observe(self) -> ComputerObservation:
        text_result = await self.browser.run(action="get_text")
        screenshot_result = await self.browser.run(action="screenshot")
        screenshot = _extract_screenshot_bytes(str(screenshot_result))
        return ComputerObservation(
            screenshot=screenshot,
            text=str(text_result),
            app="browser",
            window_title=None,
            metadata={"tool": "BrowserTool"},
        )

    async def execute(self, action: ComputerAction) -> str:
        action_type = action.action_type
        if action_type == ComputerActionType.NAVIGATE:
            result = await self.browser.run(action="navigate", url=action.url)
        elif action_type == ComputerActionType.CLICK:
            selector = action.text or action.raw.get("selector")
            if not selector:
                raise ValueError("Browser click actions require text or raw['selector'].")
            result = await self.browser.run(action="click", selector=selector)
        elif action_type == ComputerActionType.TYPE_TEXT:
            selector = action.raw.get("selector")
            if not selector:
                raise ValueError("Browser type_text actions require raw['selector'].")
            result = await self.browser.run(
                action="fill", selector=selector, value=action.text or ""
            )
        elif action_type == ComputerActionType.SCREENSHOT:
            result = await self.browser.run(action="screenshot")
        elif action_type == ComputerActionType.WAIT:
            await asyncio.sleep(action.duration or 0)
            return "waited"
        else:
            raise ValueError(f"BrowserScreenProvider cannot execute {action_type.value!r}.")
        if getattr(result, "is_error", False):
            raise RuntimeError(str(result))
        return str(result)

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.browser.run(action="close")


class LocalScreenProvider:
    """Local desktop screen provider using optional ``pyautogui``."""

    def __init__(self, pyautogui_module: Any | None = None) -> None:
        if pyautogui_module is None:
            try:
                import pyautogui as pyautogui_module
            except ImportError:
                raise ImportError(
                    "pyautogui is required for LocalScreenProvider: "
                    "pip install synapsekit[computer-use]"
                ) from None
        self._pyautogui = pyautogui_module

    async def observe(self) -> ComputerObservation:
        screenshot = await asyncio.to_thread(self._pyautogui.screenshot)
        width, height = getattr(screenshot, "size", (None, None))
        raw = _image_to_png_bytes(screenshot)
        return ComputerObservation(screenshot=raw, app="local", width=width, height=height)

    async def execute(self, action: ComputerAction) -> str:
        action_type = action.action_type
        if action_type == ComputerActionType.CLICK:
            await asyncio.to_thread(self._pyautogui.click, action.x, action.y)
        elif action_type == ComputerActionType.DOUBLE_CLICK:
            await asyncio.to_thread(self._pyautogui.doubleClick, action.x, action.y)
        elif action_type == ComputerActionType.RIGHT_CLICK:
            await asyncio.to_thread(self._pyautogui.rightClick, action.x, action.y)
        elif action_type == ComputerActionType.MOVE:
            await asyncio.to_thread(
                self._pyautogui.moveTo, action.x, action.y, duration=action.duration or 0
            )
        elif action_type == ComputerActionType.DRAG:
            await asyncio.to_thread(
                self._pyautogui.dragTo,
                action.end_x,
                action.end_y,
                duration=action.duration or 0,
            )
        elif action_type == ComputerActionType.TYPE_TEXT:
            await asyncio.to_thread(self._pyautogui.write, action.text or "")
        elif action_type == ComputerActionType.KEY:
            await asyncio.to_thread(self._pyautogui.press, action.key or "")
        elif action_type == ComputerActionType.HOTKEY:
            await asyncio.to_thread(self._pyautogui.hotkey, *action.keys)
        elif action_type == ComputerActionType.SCROLL:
            await asyncio.to_thread(self._pyautogui.scroll, action.delta_y)
        elif action_type == ComputerActionType.WAIT:
            await asyncio.sleep(action.duration or 0)
        elif action_type == ComputerActionType.SCREENSHOT:
            await self.observe()
        else:
            raise ValueError(f"LocalScreenProvider cannot execute {action_type.value!r}.")
        return f"executed {action_type.value}"

    async def close(self) -> None:
        return None


class VNCScreenProvider:
    """Remote screen provider wrapper for optional VNC clients."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def observe(self) -> ComputerObservation:
        if not hasattr(self.client, "screenshot"):
            raise TypeError("VNC client must expose screenshot().")
        screenshot = await _maybe_await(self.client.screenshot())
        return ComputerObservation(screenshot=screenshot, app="vnc", metadata={"provider": "vnc"})

    async def execute(self, action: ComputerAction) -> str:
        if not hasattr(self.client, "execute"):
            raise TypeError("VNC client must expose execute(action).")
        result = await _maybe_await(self.client.execute(action))
        return str(result)

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            await _maybe_await(close())


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _image_to_png_bytes(image: Any) -> bytes | None:
    if image is None:
        return None
    if isinstance(image, bytes):
        return image
    if hasattr(image, "save"):
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    return None


def _extract_screenshot_bytes(output: str) -> bytes | None:
    marker = "base64:"
    if marker not in output:
        return None
    import base64

    encoded = output.split(marker, 1)[1].strip()
    return base64.b64decode(encoded)


__all__ = ["BrowserScreenProvider", "LocalScreenProvider", "ScreenProvider", "VNCScreenProvider"]
