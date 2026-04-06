"""WeatherTool: current weather and forecasts via OpenWeatherMap."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, cast
from urllib.error import HTTPError, URLError

from ..base import BaseTool, ToolResult


class WeatherTool(BaseTool):
    """Get current weather and short-term forecast via OpenWeatherMap.

    Auth via ``OPENWEATHERMAP_API_KEY`` env var. Uses stdlib ``urllib``
    only — no extra dependencies.
    """

    name = "weather"
    description = (
        "Get current weather or forecast for a city using OpenWeatherMap. "
        "Actions: current, forecast."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Weather action: 'current' or 'forecast'",
                "enum": ["current", "forecast"],
                "default": "current",
            },
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Delhi' or 'London,UK'",
            },
            "days": {
                "type": "integer",
                "description": "Forecast days (1-5), default 5",
                "default": 5,
            },
        },
        "required": ["city"],
    }

    _base_url = "https://api.openweathermap.org/data/2.5"

    def _get_api_key(self) -> str | None:
        return os.getenv("OPENWEATHERMAP_API_KEY")

    async def _fetch_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{self._base_url}/{endpoint}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "SynapseKit/WeatherTool"})
        loop = asyncio.get_running_loop()

        def _fetch() -> dict[str, Any]:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Unexpected weather API response format")
            return cast(dict[str, Any], payload)

        return await loop.run_in_executor(None, _fetch)

    async def get_current(self, city: str) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(output="", error="OPENWEATHERMAP_API_KEY is not set")

        try:
            data = await self._fetch_json(
                "weather",
                {"q": city, "appid": api_key, "units": "metric"},
            )
            weather = data.get("weather", [{}])[0].get("description", "unknown")
            main = data.get("main", {})
            wind = data.get("wind", {})
            city_name = data.get("name", city)
            country = data.get("sys", {}).get("country", "")
            location = f"{city_name}, {country}".strip(", ")

            return ToolResult(
                output=(
                    f"Current weather for {location}:\n"
                    f"- Condition: {weather}\n"
                    f"- Temperature: {main.get('temp', 'N/A')}°C\n"
                    f"- Humidity: {main.get('humidity', 'N/A')}%\n"
                    f"- Wind: {wind.get('speed', 'N/A')} m/s"
                )
            )
        except HTTPError as e:
            return ToolResult(output="", error=f"Weather API HTTP error: {e.code}")
        except URLError as e:
            return ToolResult(output="", error=f"Weather API connection error: {e.reason}")
        except Exception as e:
            return ToolResult(output="", error=f"Weather lookup failed: {e}")

    async def get_forecast(self, city: str, days: int = 5) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(output="", error="OPENWEATHERMAP_API_KEY is not set")

        days = max(1, min(days, 5))
        try:
            data = await self._fetch_json(
                "forecast",
                {"q": city, "appid": api_key, "units": "metric"},
            )
            city_info = data.get("city", {})
            location = f"{city_info.get('name', city)}, {city_info.get('country', '')}".strip(", ")
            entries = data.get("list", [])
            if not entries:
                return ToolResult(output=f"No forecast data available for {location}.")

            by_day: dict[str, list[dict[str, Any]]] = {}
            for entry in entries:
                dt_txt = entry.get("dt_txt")
                if not dt_txt:
                    continue
                day = dt_txt.split(" ")[0]
                by_day.setdefault(day, []).append(entry)

            lines = [f"{days}-day forecast for {location}:"]
            for day in sorted(by_day.keys())[:days]:
                day_entries = by_day[day]
                temps = [e.get("main", {}).get("temp") for e in day_entries]
                temps = [t for t in temps if isinstance(t, (int, float))]
                weather = day_entries[0].get("weather", [{}])[0].get("description", "unknown")
                wind = day_entries[0].get("wind", {}).get("speed", "N/A")
                humidity = day_entries[0].get("main", {}).get("humidity", "N/A")

                temp_text = f"{min(temps):.1f}°C to {max(temps):.1f}°C" if temps else "N/A"
                readable_day = datetime.strptime(day, "%Y-%m-%d").strftime("%a, %b %d")
                lines.append(
                    f"- {readable_day}: {weather}, Temp: {temp_text}, "
                    f"Humidity: {humidity}%, Wind: {wind} m/s"
                )

            return ToolResult(output="\n".join(lines))
        except HTTPError as e:
            return ToolResult(output="", error=f"Forecast API HTTP error: {e.code}")
        except URLError as e:
            return ToolResult(output="", error=f"Forecast API connection error: {e.reason}")
        except Exception as e:
            return ToolResult(output="", error=f"Forecast lookup failed: {e}")

    async def run(
        self,
        action: str = "current",
        city: str = "",
        days: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        city = city or kwargs.get("input", "")
        if not city:
            return ToolResult(output="", error="City is required.")

        if action == "current":
            return await self.get_current(city)
        if action == "forecast":
            return await self.get_forecast(city, days=days)
        return ToolResult(
            output="",
            error=f"Unknown action: {action!r}. Must be 'current' or 'forecast'.",
        )
