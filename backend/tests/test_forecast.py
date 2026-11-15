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
