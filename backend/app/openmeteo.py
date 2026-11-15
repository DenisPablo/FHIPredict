"""Async Open-Meteo Forecast client and rolling feature builder."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
import httpx
import numpy as np


@dataclass
class WeatherGridForecast:
    """Weather features for `G` grid points across `D` forecast days."""

    dates: list[str]  # YYYY-MM-DD, length D
    temperature_2m_max: np.ndarray  # (G, D)
    relative_humidity_2m: np.ndarray  # (G, D)
    wind_speed_10m_max: np.ndarray  # (G, D)
    precipitation_sum: np.ndarray  # (G, D)
    shortwave_radiation_sum: np.ndarray  # (G, D)
    lluvia_acum_7d: np.ndarray  # (G, D)
    lluvia_acum_15d: np.ndarray  # (G, D)
    lluvia_acum_30d: np.ndarray  # (G, D)
    dias_consecutivos_sin_lluvia: np.ndarray  # (G, D)
    vpd_approx: np.ndarray  # (G, D)
    wind_direction_10m_dominant: np.ndarray | None = None  # (G, D) degrees [0, 360)
    wind_gusts_10m_max: np.ndarray | None = None  # (G, D) km/h


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum along axis=1 (including current index)."""
    cs = np.cumsum(arr, axis=1, dtype=np.float32)
    out = cs.copy()
    out[:, window:] = cs[:, window:] - cs[:, :-window]
    return out


def _dry_streak(precip: np.ndarray) -> np.ndarray:
    """Consecutive days with precipitation < 1.0 mm per grid point."""
    g, n = precip.shape
    streak = np.zeros((g, n), dtype=np.float32)
    for t in range(n):
        dry = precip[:, t] < 1.0
        if t == 0:
            streak[:, t] = np.where(dry, 1.0, 0.0)
        else:
            streak[:, t] = np.where(dry, streak[:, t - 1] + 1.0, 0.0)
    return streak


def build_weather_features(
    times: list[str],
    tmax: np.ndarray,
    rh: np.ndarray,
    wind: np.ndarray,
    precip: np.ndarray,
    rad: np.ndarray,
    forecast_days: int,
    wind_dir: np.ndarray | None = None,
    wind_gusts: np.ndarray | None = None,
) -> WeatherGridForecast:
    """Compute rolling rainfall, dry-streak, VPD, and directional wind features."""
    # Replace any occasional API nulls with safe local defaults
    tmax = np.nan_to_num(tmax, nan=22.0).astype(np.float32)
    rh = np.clip(np.nan_to_num(rh, nan=55.0), 0.0, 100.0).astype(np.float32)
    wind = np.nan_to_num(wind, nan=12.0).astype(np.float32)
    precip = np.maximum(np.nan_to_num(precip, nan=0.0), 0.0).astype(np.float32)
    rad = np.maximum(np.nan_to_num(rad, nan=15.0), 0.0).astype(np.float32)

    if wind_dir is None:
        wdir_clean = np.zeros_like(wind, dtype=np.float32)
    else:
        wdir_clean = (np.nan_to_num(wind_dir, nan=0.0) % 360.0).astype(np.float32)

    if wind_gusts is None:
        wgust_clean = (wind * 1.35).astype(np.float32)
    else:
        wgust_clean = np.maximum(np.nan_to_num(wind_gusts, nan=0.0).astype(np.float32), wind)

    r7 = _rolling_sum(precip, 7)
    r15 = _rolling_sum(precip, 15)
    r30 = _rolling_sum(precip, 30)
    streak = _dry_streak(precip)
    vpd = (0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3)) * (1.0 - rh / 100.0)).astype(np.float32)

    # Identify forecast start index (first date >= today, fallback to last `forecast_days`)
    today_iso = date.today().isoformat()
    start_idx = next((i for i, d in enumerate(times) if d >= today_iso), max(0, len(times) - forecast_days))
    end_idx = min(len(times), start_idx + forecast_days)

    return WeatherGridForecast(
        dates=times[start_idx:end_idx],
        temperature_2m_max=tmax[:, start_idx:end_idx],
        relative_humidity_2m=rh[:, start_idx:end_idx],
        wind_speed_10m_max=wind[:, start_idx:end_idx],
        precipitation_sum=precip[:, start_idx:end_idx],
        shortwave_radiation_sum=rad[:, start_idx:end_idx],
        lluvia_acum_7d=r7[:, start_idx:end_idx],
        lluvia_acum_15d=r15[:, start_idx:end_idx],
        lluvia_acum_30d=r30[:, start_idx:end_idx],
        dias_consecutivos_sin_lluvia=streak[:, start_idx:end_idx],
        vpd_approx=vpd[:, start_idx:end_idx],
        wind_direction_10m_dominant=wdir_clean[:, start_idx:end_idx],
        wind_gusts_10m_max=wgust_clean[:, start_idx:end_idx],
    )


async def fetch_openmeteo_grid(
    coords: list[tuple[float, float]],
    openmeteo_url: str,
    past_days: int,
    forecast_days: int,
    chunk_size: int = 80,
    max_concurrency: int = 4,
) -> WeatherGridForecast:
    """Fetch daily past+forecast weather for all grid coordinates in concurrent chunks."""
    sem = asyncio.Semaphore(max_concurrency)
    daily_vars = ",".join([
        "temperature_2m_max",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant",
        "wind_gusts_10m_max",
        "precipitation_sum",
        "shortwave_radiation_sum",
    ])

    async def fetch_chunk(client: httpx.AsyncClient, chunk: list[tuple[float, float]]) -> list[dict]:
        params = {
            "latitude": ",".join(f"{lat:.4f}" for lat, _ in chunk),
            "longitude": ",".join(f"{lon:.4f}" for _, lon in chunk),
            "daily": daily_vars,
            "past_days": str(past_days),
            "forecast_days": str(forecast_days),
            "timezone": "America/Argentina/Buenos_Aires",
        }
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.get(openmeteo_url, params=params, timeout=60.0)
                except httpx.TimeoutException:
                    if attempt < 4:
                        await asyncio.sleep(2.0 * (2**attempt))
                        continue
                    raise
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = float(retry_after) + 1.0 if retry_after and retry_after.isdigit() else 2.0 * (2**attempt)
                    await asyncio.sleep(min(wait_s, 65.0))
                    continue
                resp.raise_for_status()
                payload = resp.json()
                await asyncio.sleep(0.35)
                return payload if isinstance(payload, list) else [payload]
        raise RuntimeError("Open-Meteo request failed after retries")

    chunks = [coords[i : i + chunk_size] for i in range(0, len(coords), chunk_size)]
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch_chunk(client, c) for c in chunks))

    items = [entry for group in results for entry in group]
    if len(items) != len(coords):
        raise RuntimeError(f"Expected {len(coords)} Open-Meteo locations, got {len(items)}")

    times: list[str] = items[0]["daily"]["time"]
    n_days = len(times)
    n_pts = len(items)

    def extract(key: str) -> np.ndarray:
        arr = np.empty((n_pts, n_days), dtype=np.float32)
        for idx, item in enumerate(items):
            vals = item["daily"].get(key)
            if vals is None:
                arr[idx] = np.nan
            else:
                arr[idx] = [np.nan if v is None else float(v) for v in vals]
        return arr

    return build_weather_features(
        times=times,
        tmax=extract("temperature_2m_max"),
        rh=extract("relative_humidity_2m_mean"),
        wind=extract("wind_speed_10m_max"),
        precip=extract("precipitation_sum"),
        rad=extract("shortwave_radiation_sum"),
        forecast_days=forecast_days,
        wind_dir=extract("wind_direction_10m_dominant"),
        wind_gusts=extract("wind_gusts_10m_max"),
    )
