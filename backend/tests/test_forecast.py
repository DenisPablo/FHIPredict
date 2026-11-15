"""Tests for FHIPredict backend inference and rolling weather features."""

from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np

from app.config import Settings
from app.main import create_app
from app.openmeteo import build_weather_features
from app.predictor import FireRiskPredictor


def test_rolling_weather_features():
    times = [f"2026-11-{d:02d}" for d in range(1, 31)] + [f"2026-12-{d:02d}" for d in range(1, 16)]
    n = len(times)
    precip = np.zeros((2, n), dtype=np.float32)
    precip[0, -16] = 5.0  # rain 1 day before forecast starts
    tmax = np.full((2, n), 32.0, dtype=np.float32)
    rh = np.full((2, n), 25.0, dtype=np.float32)
    wind = np.full((2, n), 28.0, dtype=np.float32)
    rad = np.full((2, n), 22.0, dtype=np.float32)

    wf = build_weather_features(times, tmax, rh, wind, precip, rad, forecast_days=15)
    assert len(wf.dates) == 15
    assert wf.temperature_2m_max.shape == (2, 15)
    assert wf.vpd_approx[0, 0] > 2.5


def test_bundle_loads_and_predicts():
    models_dir = Path(__file__).resolve().parents[1] / "models"
    predictor = FireRiskPredictor.from_bundle(models_dir / "fire_risk_bundle.joblib")
    assert predictor.model_version == "lgbm-risk-24h-v2"
    X = np.zeros((4, len(predictor.feature_columns)), dtype=np.float32)
    probs = predictor.predict_probabilities(X)
    assert probs.shape == (4,)
    assert np.all((probs > 0.0) & (probs < 1.0))


def test_health_endpoint():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["cells_total"] == 64653
        assert body["grid_points"] == 971


def test_predict_with_stability_and_confidence_decay():
    models_dir = Path(__file__).resolve().parents[1] / "models"
    predictor = FireRiskPredictor.from_bundle(models_dir / "fire_risk_bundle.joblib")
    X = np.zeros((4, len(predictor.feature_columns)), dtype=np.float32)

    probs_ref = predictor.predict_probabilities(X)
    probs, stability = predictor.predict_with_stability(X)

    assert np.allclose(probs, probs_ref, atol=1e-6)
    assert stability.shape == (4,)
    assert np.all((stability >= 0.2) & (stability <= 1.0))

    from app.service import ForecastService

    svc = ForecastService(Settings(models_dir=models_dir))
    stab = np.array([0.95, 0.95], dtype=np.float32)
    pct_day0, lvl_day0 = svc._compute_confidence(stab, np.array([0, 0], dtype=np.int32), is_stale=False)
    pct_day14, lvl_day14 = svc._compute_confidence(stab, np.array([14, 14], dtype=np.int32), is_stale=False)
    pct_stale, _ = svc._compute_confidence(stab, np.array([0, 0], dtype=np.int32), is_stale=True)

    assert np.all(pct_day0 > pct_day14)
    assert np.all(pct_day0 > pct_stale)
    assert lvl_day0[0] == "Alta"
    assert lvl_day14[0] in ("Media", "Baja")


def test_periodic_refresh_recovers_from_error_and_stops():
    import asyncio
    from app.service import ForecastService

    models_dir = Path(__file__).resolve().parents[1] / "models"
    svc = ForecastService(Settings(models_dir=models_dir))

    calls = 0

    async def fake_refresh(force_refresh: bool = False):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient 429")
        return svc._cache

    svc._refresh_forecast = fake_refresh  # type: ignore[method-assign]

    async def runner():
        task = asyncio.create_task(svc.run_periodic_refresh(interval_s=0.01))
        for _ in range(30):
            if calls >= 2:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(runner())
    assert calls >= 2
