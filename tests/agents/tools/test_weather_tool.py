from __future__ import annotations

from unittest.mock import patch

import pytest

from synapsekit.agents.tools import WeatherTool


@pytest.mark.asyncio
async def test_weather_tool_requires_api_key() -> None:
    tool = WeatherTool()
    with patch.dict("os.environ", {}, clear=True):
        res = await tool.run(action="current", city="Delhi")
    assert res.error == "OPENWEATHERMAP_API_KEY is not set"


@pytest.mark.asyncio
async def test_get_current_formats_output() -> None:
    tool = WeatherTool()
    fake = {
        "name": "Delhi",
        "sys": {"country": "IN"},
        "weather": [{"description": "clear sky"}],
        "main": {"temp": 30.5, "humidity": 40},
        "wind": {"speed": 3.2},
    }
    with patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "k"}):
        with patch.object(tool, "_fetch_json", return_value=fake):
            res = await tool.get_current("Delhi")

    assert res.error is None
    assert "Current weather for Delhi, IN" in res.output
    assert "Temperature: 30.5" in res.output
    assert "Humidity: 40%" in res.output
    assert "Wind: 3.2" in res.output


@pytest.mark.asyncio
async def test_get_forecast_returns_day_by_day() -> None:
    tool = WeatherTool()
    fake = {
        "city": {"name": "Delhi", "country": "IN"},
        "list": [
            {
                "dt_txt": "2026-04-07 09:00:00",
                "weather": [{"description": "haze"}],
                "main": {"temp": 28.0, "humidity": 55},
                "wind": {"speed": 2.0},
            },
            {
                "dt_txt": "2026-04-07 12:00:00",
                "weather": [{"description": "haze"}],
                "main": {"temp": 31.0, "humidity": 50},
                "wind": {"speed": 2.8},
            },
            {
                "dt_txt": "2026-04-08 09:00:00",
                "weather": [{"description": "cloudy"}],
                "main": {"temp": 29.0, "humidity": 60},
                "wind": {"speed": 1.9},
            },
        ],
    }
    with patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "k"}):
        with patch.object(tool, "_fetch_json", return_value=fake):
            res = await tool.get_forecast("Delhi", days=2)

    assert res.error is None
    assert "2-day forecast for Delhi, IN" in res.output
    assert "Tue, Apr 07" in res.output
    assert "Wed, Apr 08" in res.output


@pytest.mark.asyncio
async def test_missing_city() -> None:
    tool = WeatherTool()
    with patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "k"}):
        res = await tool.run(action="current", city="")
    assert res.is_error
    assert "City" in res.error


@pytest.mark.asyncio
async def test_run_unknown_action() -> None:
    tool = WeatherTool()
    with patch.dict("os.environ", {"OPENWEATHERMAP_API_KEY": "k"}):
        res = await tool.run(action="bad", city="Delhi")
    assert "Unknown action" in (res.error or "")


@pytest.mark.asyncio
async def test_top_level_export() -> None:
    from synapsekit import WeatherTool as TopWeatherTool

    assert TopWeatherTool is WeatherTool
