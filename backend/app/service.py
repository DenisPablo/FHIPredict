"""Inference orchestration and TTL cache for all 64,653 Argentina cells."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
import duckdb
import joblib
import numpy as np
from .config import Settings
from .openmeteo import WeatherGridForecast, fetch_openmeteo_grid
from .predictor import FireRiskPredictor

logger = logging.getLogger(__name__)


RISK_ORDER = {"Bajo": 0, "Moderado": 1, "Alto": 2, "Extremo": 3}


@dataclass
class CellBaseline:
    cell_id: np.ndarray  # (N,) int32
    lat_centro: np.ndarray  # (N,) float32
    lon_centro: np.ndarray  # (N,) float32
    elevacion_m: np.ndarray  # (N,) float32
    focos_ultimos_7d: np.ndarray  # (N,) float32
    focos_ultimos_30d: np.ndarray  # (N,) float32
    frp_ultimos_7d: np.ndarray  # (N,) float32
    anios_desde_ultimo_fuego: np.ndarray  # (N,) float32
    clim_months: np.ndarray  # (N, 12) float32
    grid_coords: list[tuple[float, float]]  # length G
    cell_to_grid: np.ndarray  # (N,) int32


@dataclass
class CachedForecast:
    generated_at: str
    expires_at: float
    dates: list[str]  # length D
    probs: np.ndarray  # (N, D) float32
    weather: WeatherGridForecast


def load_baseline(parquet_path: str) -> CellBaseline:
    con = duckdb.connect(":memory:")
    tbl = con.sql(f"SELECT * FROM read_parquet('{parquet_path}') ORDER BY cell_id").fetchnumpy()
    lat_c = tbl["lat_clima"].astype(np.float32)
    lon_c = tbl["lon_clima"].astype(np.float32)

    grid_map: dict[tuple[float, float], int] = {}
    grid_coords: list[tuple[float, float]] = []
    cell_to_grid = np.empty(len(lat_c), dtype=np.int32)
    for i, (la, lo) in enumerate(zip(lat_c.tolist(), lon_c.tolist())):
        key = (round(la, 4), round(lo, 4))
        idx = grid_map.get(key)
        if idx is None:
            idx = len(grid_coords)
            grid_map[key] = idx
            grid_coords.append(key)
        cell_to_grid[i] = idx

    clim = np.column_stack([tbl[f"m{m:02d}"].astype(np.float32) for m in range(1, 13)])
    return CellBaseline(
        cell_id=tbl["cell_id"].astype(np.int32),
        lat_centro=tbl["lat_centro"].astype(np.float32),
        lon_centro=tbl["lon_centro"].astype(np.float32),
        elevacion_m=tbl["elevacion_m"].astype(np.float32),
        focos_ultimos_7d=tbl["focos_ultimos_7d"].astype(np.float32),
        focos_ultimos_30d=tbl["focos_ultimos_30d"].astype(np.float32),
        frp_ultimos_7d=tbl["frp_ultimos_7d"].astype(np.float32),
        anios_desde_ultimo_fuego=tbl["anios_desde_ultimo_fuego"].astype(np.float32),
        clim_months=clim,
        grid_coords=grid_coords,
        cell_to_grid=cell_to_grid,
    )


class ForecastService:
    """Computes and caches 15-day fire-risk predictions for all cells."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.predictor = FireRiskPredictor.from_bundle(settings.models_dir / "fire_risk_bundle.joblib")
        self.baseline = load_baseline((settings.models_dir / "cell_baseline.parquet").as_posix())
        self._cache_path: Path = settings.models_dir / ".forecast_cache.joblib"
        self._cache: CachedForecast | None = self._load_disk_cache()
        self._lock = asyncio.Lock()

    def _load_disk_cache(self) -> CachedForecast | None:
        if not self._cache_path.exists():
            return None
        try:
            payload = joblib.load(self._cache_path)
            saved_epoch = float(payload["saved_epoch"])
            age_s = max(0.0, time.time() - saved_epoch)
            remaining = max(0.0, self.settings.cache_ttl_s - age_s)
            return CachedForecast(
                generated_at=payload["generated_at"],
                expires_at=time.monotonic() + remaining,
                dates=payload["dates"],
                probs=payload["probs"],
                weather=payload["weather"],
            )
        except Exception as exc:
            logger.warning("Ignoring corrupt forecast disk cache: %s", exc)
            return None

    def _save_disk_cache(self, cached: CachedForecast) -> None:
        try:
            tmp_path = self._cache_path.with_suffix(".tmp")
            joblib.dump(
                {
                    "saved_epoch": time.time(),
                    "generated_at": cached.generated_at,
                    "dates": cached.dates,
                    "probs": cached.probs,
                    "weather": cached.weather,
                },
                tmp_path,
                compress=3,
            )
            tmp_path.replace(self._cache_path)
        except Exception as exc:
            logger.warning("Failed to persist forecast disk cache: %s", exc)

    async def get_forecast(self, force_refresh: bool = False) -> CachedForecast:
        now = time.monotonic()
        if not force_refresh and self._cache is not None and now < self._cache.expires_at:
            return self._cache

        async with self._lock:
            now = time.monotonic()
            if not force_refresh and self._cache is not None and now < self._cache.expires_at:
                return self._cache

            try:
                weather = await fetch_openmeteo_grid(
                    coords=self.baseline.grid_coords,
                    openmeteo_url=self.settings.openmeteo_url,
                    past_days=self.settings.past_days,
                    forecast_days=self.settings.forecast_days,
                    chunk_size=self.settings.chunk_size,
                    max_concurrency=self.settings.max_concurrency,
                )
            except Exception as exc:
                if self._cache is not None:
                    logger.warning("Open-Meteo refresh failed (%s); serving stale cached forecast", exc)
                    self._cache.expires_at = time.monotonic() + 60.0
                    return self._cache
                raise

            probs = self._score_all_days(weather)
            self._cache = CachedForecast(
                generated_at=datetime.now(timezone.utc).isoformat(),
                expires_at=time.monotonic() + self.settings.cache_ttl_s,
                dates=weather.dates,
                probs=probs,
                weather=weather,
            )
            self._save_disk_cache(self._cache)
            return self._cache

    def _score_all_days(self, weather: WeatherGridForecast) -> np.ndarray:
        b = self.baseline
        n_cells = len(b.cell_id)
        n_days = len(weather.dates)
        probs = np.empty((n_cells, n_days), dtype=np.float32)
        g_idx = b.cell_to_grid

        for d_idx, date_str in enumerate(weather.dates):
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            mes_val = float(dt.month)
            doy_val = float(dt.timetuple().tm_yday)

            X = np.column_stack([
                b.lat_centro,
                b.lon_centro,
                np.full(n_cells, mes_val, dtype=np.float32),
                np.full(n_cells, doy_val, dtype=np.float32),
                b.focos_ultimos_7d,
                b.focos_ultimos_30d,
                b.frp_ultimos_7d,
                b.clim_months[:, dt.month - 1],
                b.anios_desde_ultimo_fuego,
                b.elevacion_m,
                weather.temperature_2m_max[g_idx, d_idx],
                weather.relative_humidity_2m[g_idx, d_idx],
                weather.wind_speed_10m_max[g_idx, d_idx],
                weather.precipitation_sum[g_idx, d_idx],
                weather.shortwave_radiation_sum[g_idx, d_idx],
                weather.lluvia_acum_7d[g_idx, d_idx],
                weather.lluvia_acum_15d[g_idx, d_idx],
                weather.lluvia_acum_30d[g_idx, d_idx],
                weather.dias_consecutivos_sin_lluvia[g_idx, d_idx],
                weather.vpd_approx[g_idx, d_idx],
            ])
            probs[:, d_idx] = self.predictor.predict_probabilities(X)

        return probs

    def query_geojson(
        self,
        cached: CachedForecast,
        mode: str = "window",
        days: int = 1,
        day_offset: int = 0,
        min_risk: str = "Moderado",
        bbox: tuple[float, float, float, float] | None = None,
    ) -> dict:
        b = self.baseline
        n_avail = len(cached.dates)

        if mode == "day":
            idx = max(0, min(day_offset, n_avail - 1))
            prob = cached.probs[:, idx]
            w_idx = idx
            fecha_desde = cached.dates[idx]
            fecha_hasta = cached.dates[idx]
        else:
            span = max(1, min(days, n_avail))
            sub = cached.probs[:, :span]
            # Cumulative probability of >=1 fire event in [0..span-1]
            prob = (1.0 - np.prod(1.0 - sub, axis=1)).astype(np.float32)
            # Representative weather from the peak-risk day for each cell
            w_idx_arr = np.argmax(sub, axis=1)
            w_idx = int(0)  # handled below if array
            fecha_desde = cached.dates[0]
            fecha_hasta = cached.dates[span - 1]

        # Categorize using daily cutoffs (consistent scale)
        cats = self.predictor.categorize(prob)

        # Summary counts across ALL Argentina cells
        counts = {
            "Bajo": int(np.sum(cats == "Bajo")),
            "Moderado": int(np.sum(cats == "Moderado")),
            "Alto": int(np.sum(cats == "Alto")),
            "Extremo": int(np.sum(cats == "Extremo")),
        }

        min_rank = RISK_ORDER.get(min_risk, 1)
        ranks = np.where(cats == "Extremo", 3, np.where(cats == "Alto", 2, np.where(cats == "Moderado", 1, 0)))
        mask = ranks >= min_rank

        if bbox is not None:
            west, south, east, north = bbox
            mask &= (
                (b.lon_centro >= west)
                & (b.lon_centro <= east)
                & (b.lat_centro >= south)
                & (b.lat_centro <= north)
            )

        sel = np.nonzero(mask)[0]
        # Cap to 20,000 highest-risk cells to keep MapLibre GeoJSON snappy
        if len(sel) > 20_000:
            top = np.argpartition(prob[sel], -20_000)[-20_000:]
            sel = sel[top]

        g_sel = b.cell_to_grid[sel]
        if mode == "day":
            d_sel = np.full(len(sel), w_idx, dtype=np.int32)
        else:
            d_sel = w_idx_arr[sel]

        w = cached.weather
        tmax_s = w.temperature_2m_max[g_sel, d_sel]
        rh_s = w.relative_humidity_2m[g_sel, d_sel]
        wind_s = w.wind_speed_10m_max[g_sel, d_sel]
        rain_s = w.precipitation_sum[g_sel, d_sel]
        rain30_s = w.lluvia_acum_30d[g_sel, d_sel]
        dry_s = w.dias_consecutivos_sin_lluvia[g_sel, d_sel]
        vpd_s = w.vpd_approx[g_sel, d_sel]

        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(float(b.lon_centro[i]), 4), round(float(b.lat_centro[i]), 4)],
                },
                "properties": {
                    "cell_id": int(b.cell_id[i]),
                    "riesgo_pct": round(float(prob[i]) * 100.0, 2),
                    "categoria": str(cats[i]),
                    "tmax_c": round(float(tmax_s[k]), 1),
                    "rh_pct": round(float(rh_s[k]), 0),
                    "wind_kmh": round(float(wind_s[k]), 1),
                    "rain_mm": round(float(rain_s[k]), 1),
                    "rain_30d_mm": round(float(rain30_s[k]), 1),
                    "dry_days": int(dry_s[k]),
                    "vpd_kpa": round(float(vpd_s[k]), 2),
                    "elev_m": int(round(float(b.elevacion_m[i]))),
                    "years_since_fire": round(float(b.anios_desde_ultimo_fuego[i]), 1),
                },
            }
            for k, i in enumerate(sel.tolist())
        ]

        return {
            "type": "FeatureCollection",
            "meta": {
                "model_version": self.predictor.model_version,
                "generated_at": cached.generated_at,
                "mode": mode,
                "fecha_desde": fecha_desde,
                "fecha_hasta": fecha_hasta,
                "dates_available": cached.dates,
                "counts": counts,
                "returned": len(features),
            },
            "features": features,
        }
