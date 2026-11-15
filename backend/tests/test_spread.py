"""Unit tests for spatial hotspot propagation (`app.spread`) and directional weather features."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np

from app.main import create_app
from app.openmeteo import build_weather_features
from app.spread import (
    SPREAD_FEATURE_COLUMNS,
    SpreadPredictor,
    aggregate_target_probabilities,
    build_neighbor_graph,
    compute_bearing_and_distance,
    compute_wind_alignment,
    fuse_risk_and_spread,
)


def test_weather_features_include_wind_direction_and_gusts():
    times = [f"2026-11-{d:02d}" for d in range(1, 31)] + [f"2026-12-{d:02d}" for d in range(1, 16)]
    n = len(times)
    tmax = np.full((2, n), 30.0, dtype=np.float32)
    rh = np.full((2, n), 30.0, dtype=np.float32)
    wind = np.full((2, n), 25.0, dtype=np.float32)
    precip = np.zeros((2, n), dtype=np.float32)
    rad = np.full((2, n), 20.0, dtype=np.float32)
    wind_dir = np.full((2, n), 180.0, dtype=np.float32)
    wind_gusts = np.full((2, n), 40.0, dtype=np.float32)

    wf = build_weather_features(
        times, tmax, rh, wind, precip, rad, forecast_days=15, wind_dir=wind_dir, wind_gusts=wind_gusts
    )
    assert wf.wind_direction_10m_dominant is not None
    assert wf.wind_gusts_10m_max is not None
    assert wf.wind_direction_10m_dominant.shape == (2, 15)
    assert np.allclose(wf.wind_direction_10m_dominant, 180.0)
    assert np.allclose(wf.wind_gusts_10m_max, 40.0)


def test_bearing_distance_and_wind_alignment():
    # Source at (-31.50, -64.50), target due North at (-31.45, -64.50)
    lat_s = np.array([-31.50], dtype=np.float32)
    lon_s = np.array([-64.50], dtype=np.float32)
    lat_t = np.array([-31.45], dtype=np.float32)
    lon_t = np.array([-64.50], dtype=np.float32)

    bearing_deg, sin_b, cos_b, dist_km = compute_bearing_and_distance(lat_s, lon_s, lat_t, lon_t)
    assert np.allclose(bearing_deg, 0.0, atol=1e-2)
    assert np.allclose(sin_b, 0.0, atol=1e-3)
    assert np.allclose(cos_b, 1.0, atol=1e-3)
    assert 5.0 < float(dist_km[0]) < 6.0

    # Wind FROM South (180 deg) blows North (0 deg) -> aligned (+1.0) with target due North
    align_downwind = compute_wind_alignment(np.array([180.0], dtype=np.float32), bearing_deg)
    # Wind FROM North (0 deg) blows South (180 deg) -> opposite (-1.0) to target due North
    align_upwind = compute_wind_alignment(np.array([0.0], dtype=np.float32), bearing_deg)
    assert float(align_downwind[0]) > 0.99
    assert float(align_upwind[0]) < -0.99


def test_neighbor_graph_and_probability_aggregation():
    # 3x3 regular ~5km grid around (-31.5, -64.5)
    lats = []
    lons = []
    elevs = []
    cid = 1
    cell_ids = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            cell_ids.append(cid)
            cid += 1
            lats.append(-31.50 + di * 0.045)
            lons.append(-64.50 + dj * 0.050)
            elevs.append(500.0 + di * 50.0)

    graph = build_neighbor_graph(
        cell_id=np.array(cell_ids, dtype=np.int32),
        lat_centro=np.array(lats, dtype=np.float32),
        lon_centro=np.array(lons, dtype=np.float32),
        elevacion_m=np.array(elevs, dtype=np.float32),
        max_dist_km=8.5,
    )
    # Center cell (index 4) has 8 neighbors
    center_out = np.sum(graph.src_idx == 4)
    assert center_out == 8

    # Two active sources pointing to target cell 4 with probs 0.2 and 0.5 -> 1 - (0.8 * 0.5) = 0.6
    agg = aggregate_target_probabilities(
        n_cells=9,
        dst_idx=np.array([4, 4, 2], dtype=np.int32),
        pair_probs=np.array([0.2, 0.5, 0.3], dtype=np.float32),
    )
    assert np.isclose(agg[4], 0.6, atol=1e-5)
    assert np.isclose(agg[2], 0.3, atol=1e-5)
    assert np.isclose(agg[0], 0.0, atol=1e-5)

    fused = fuse_risk_and_spread(
        p_risk=np.array([0.1, 0.0], dtype=np.float32),
        p_spread=np.array([0.5, 0.4], dtype=np.float32),
    )
    assert np.isclose(fused[0], 0.55, atol=1e-5)
    assert np.isclose(fused[1], 0.40, atol=1e-5)
