"""Open-Meteo forecast helper (no API key required)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_CACHE_DIR = Path("artifacts/cache/weather")

# Rough coordinates for common destinations (lat, lon)
CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york": (40.71, -74.01),
    "los angeles": (34.05, -118.24),
    "san francisco": (37.77, -122.42),
    "chicago": (41.88, -87.63),
    "miami": (25.76, -80.19),
    "boston": (42.36, -71.06),
    "washington": (38.91, -77.04),
    "washington dc": (38.91, -77.04),
    "seattle": (47.61, -122.33),
    "las vegas": (36.17, -115.14),
    "orlando": (28.54, -81.38),
    "denver": (39.74, -104.99),
    "atlanta": (33.75, -84.39),
    "london": (51.51, -0.13),
    "paris": (48.86, 2.35),
    "rome": (41.90, 12.48),
    "berlin": (52.52, 13.41),
    "madrid": (40.42, -3.70),
    "barcelona": (41.39, 2.16),
    "amsterdam": (52.37, 4.90),
    "dubai": (25.20, 55.27),
    "cairo": (30.04, 31.24),
    "tokyo": (35.68, 139.76),
    "singapore": (1.35, 103.82),
    "bangkok": (13.76, 100.50),
    "sydney": (-33.87, 151.21),
    "toronto": (43.65, -79.38),
    "vancouver": (49.28, -123.12),
    "montreal": (45.50, -73.57),
}


def resolve_coordinates(city_name: str) -> tuple[float, float] | None:
    if not city_name or not str(city_name).strip():
        return None
    key = str(city_name).strip().lower()
    return CITY_COORDS.get(key)


def get_weather(city_name: str, *, forecast_days: int = 7) -> str:
    """Fetch daily max/min temperature and precipitation summary from Open-Meteo."""
    coords = resolve_coordinates(city_name)
    if coords is None:
        return (
            f"[Open-Meteo] No cached coordinates for {city_name!r}. "
            "Add it to CITY_COORDS in weather_tool.py or use a nearby major city name."
        )

    lat, lon = coords
    digest = hashlib.sha1(f"{city_name.lower()}:{forecast_days}".encode("utf-8")).hexdigest()
    cache_file = _CACHE_DIR / f"{digest}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": max(1, min(int(forecast_days), 16)),
        "timezone": "auto",
    }

    try:
        r = httpx.get(OPEN_METEO_URL, params=params, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        tmax = daily.get("temperature_2m_max") or []
        tmin = daily.get("temperature_2m_min") or []
        precip = daily.get("precipitation_sum") or []

        lines = [f"### Weather forecast ({city_name.strip()})\n"]
        for i, day in enumerate(dates):
            mx = tmax[i] if i < len(tmax) else "?"
            mn = tmin[i] if i < len(tmin) else "?"
            pr = precip[i] if i < len(precip) else "?"
            lines.append(f"- {day}: high {mx}°C / low {mn}°C — precip {pr} mm")
        result = "\n".join(lines)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(result, encoding="utf-8")
        return result
    except Exception as e:
        logger.warning("Open-Meteo request failed: %s", e)
        return f"[Open-Meteo] Error: {e}"
