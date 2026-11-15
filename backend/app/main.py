"""FastAPI entry point for FHIPredict."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import get_settings
from .service import ForecastService


_service: ForecastService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = ForecastService(get_settings())
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="FHIPredict API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    async def health():
        assert _service is not None
        return {
            "status": "ok",
            "model_version": _service.predictor.model_version,
            "spread_model_version": (
                _service.spread_predictor.model_version if _service.spread_predictor is not None else None
            ),
            "cells_total": int(len(_service.baseline.cell_id)),
            "grid_points": int(len(_service.baseline.grid_coords)),
            "forecast_cached": _service._cache is not None,
        }

    @app.get("/api/v1/spread")
    async def spread(
        horizon_hours: int = Query(default=24, ge=24, le=72),
        day_offset: int = Query(default=0, ge=0, le=15),
        min_risk: Literal["Bajo", "Moderado", "Alto", "Extremo"] = "Moderado",
        bbox: str | None = None,
        refresh: bool = False,
    ):
        assert _service is not None
        if horizon_hours not in (24, 48, 72):
            raise HTTPException(status_code=400, detail="horizon_hours must be 24, 48, or 72")
        if _service.spread_predictor is None:
            raise HTTPException(status_code=503, detail="Spread model is not enabled or bundle is missing")

        parsed_bbox = None
        if bbox:
            try:
                parts = [float(x) for x in bbox.split(",")]
                if len(parts) != 4:
                    raise ValueError
                parsed_bbox = (parts[0], parts[1], parts[2], parts[3])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid bbox; expected west,south,east,north") from exc

        try:
            cached = await _service.get_forecast(force_refresh=refresh)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Forecast upstream unavailable (Open-Meteo rate limit or network error): {exc}",
            ) from exc

        return _service.query_spread_geojson(
            cached=cached,
            horizon_hours=horizon_hours,
            day_offset=day_offset,
            min_risk=min_risk,
            bbox=parsed_bbox,
        )

    @app.get("/api/v1/forecast")
    async def forecast(
        mode: Literal["window", "day"] = "window",
        days: int = Query(default=1, ge=1, le=16),
        day_offset: int = Query(default=0, ge=0, le=15),
        min_risk: Literal["Bajo", "Moderado", "Alto", "Extremo"] = "Moderado",
        bbox: str | None = None,
        refresh: bool = False,
    ):
        assert _service is not None
        parsed_bbox = None
        if bbox:
            try:
                parts = [float(x) for x in bbox.split(",")]
                if len(parts) != 4:
                    raise ValueError
                parsed_bbox = (parts[0], parts[1], parts[2], parts[3])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid bbox; expected west,south,east,north") from exc

        try:
            cached = await _service.get_forecast(force_refresh=refresh)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Forecast upstream unavailable (Open-Meteo rate limit or network error): {exc}",
            ) from exc
        return _service.query_geojson(
            cached=cached,
            mode=mode,
            days=days,
            day_offset=day_offset,
            min_risk=min_risk,
            bbox=parsed_bbox,
        )

    if settings.static_dir and settings.static_dir.is_dir():
        static_root = settings.static_dir.resolve()
        assets_dir = static_root / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            if full_path:
                candidate = (static_root / full_path).resolve()
                if candidate.is_relative_to(static_root) and candidate.is_file():
                    return FileResponse(candidate)
            return FileResponse(static_root / "index.html")

    return app


app = create_app()
