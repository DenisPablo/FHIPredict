"""Spatial hotspot propagation predictor (`lgbm-spread-v1`) and 8-neighbor graph utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np


SPREAD_FEATURE_COLUMNS: list[str] = [
    "lat_centro_t",
    "lon_centro_t",
    "mes",
    "dia_anio",
    "horizon_days",
    "dist_km",
    "sin_bearing",
    "cos_bearing",
    "delta_elev_m",
    "slope_deg",
    "n_harm_s",
    "phi_harm_s",
    "focos_7d_s",
    "focos_ultimos_7d_t",
    "focos_ultimos_30d_t",
    "frp_ultimos_7d_t",
    "mediana_historica_mes_t",
    "anios_desde_ultimo_fuego_t",
    "active_neighbors_count_t",
    "neighbors_focos_sum_t",
    "neighbors_frp_sum_t",
    "elevacion_m_t",
    "temperature_2m_max_t",
    "relative_humidity_2m_t",
    "wind_speed_10m_max_t",
    "precipitation_sum_t",
    "shortwave_radiation_sum_t",
    "lluvia_acum_7d_t",
    "lluvia_acum_15d_t",
    "lluvia_acum_30d_t",
    "dias_consecutivos_sin_lluvia_t",
    "vpd_approx_t",
]


@dataclass
class NeighborGraph:
    """Directed 8-neighborhood graph over the 5 km fire cells."""

    src_idx: np.ndarray  # (E,) int32 index into cell arrays
    dst_idx: np.ndarray  # (E,) int32 index into cell arrays
    dist_km: np.ndarray  # (E,) float32
    bearing_deg: np.ndarray  # (E,) float32 [0, 360)
    sin_bearing: np.ndarray  # (E,) float32
    cos_bearing: np.ndarray  # (E,) float32
    delta_elev_m: np.ndarray  # (E,) float32 (elev_t - elev_s)
    slope_deg: np.ndarray  # (E,) float32


def compute_bearing_and_distance(
    lat_s: np.ndarray,
    lon_s: np.ndarray,
    lat_t: np.ndarray,
    lon_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(bearing_deg, sin_bearing, cos_bearing, dist_km)` from source `s` to target `t`."""
    lat_s_r = np.radians(lat_s.astype(np.float64))
    lat_t_r = np.radians(lat_t.astype(np.float64))
    dlat_r = lat_t_r - lat_s_r
    dlon_r = np.radians((lon_t - lon_s).astype(np.float64))

    mean_lat_r = 0.5 * (lat_s_r + lat_t_r)
    dx = dlon_r * np.cos(mean_lat_r) * 6371.0
    dy = dlat_r * 6371.0
    dist_km = np.hypot(dx, dy).astype(np.float32)

    bearing_rad = np.arctan2(dx, dy)
    bearing_deg = (np.degrees(bearing_rad) % 360.0).astype(np.float32)
    sin_b = np.sin(bearing_rad).astype(np.float32)
    cos_b = np.cos(bearing_rad).astype(np.float32)
    return bearing_deg, sin_b, cos_b, dist_km


def compute_wind_alignment(
    wind_from_deg: np.ndarray,
    bearing_s_to_t_deg: np.ndarray,
) -> np.ndarray:
    """Cosine alignment in `[-1, 1]` between wind push vector (`wind_from + 180`) and `s -> t` bearing."""
    wind_to_deg = (wind_from_deg.astype(np.float64) + 180.0) % 360.0
    diff_rad = np.radians(wind_to_deg - bearing_s_to_t_deg.astype(np.float64))
    return np.cos(diff_rad).astype(np.float32)


def build_neighbor_graph(
    cell_id: np.ndarray,
    lat_centro: np.ndarray,
    lon_centro: np.ndarray,
    elevacion_m: np.ndarray,
    max_dist_km: float = 8.5,
) -> NeighborGraph:
    """Build directed Moore-neighborhood edges `(s -> t)` within `max_dist_km` using spatial hashing."""
    n = len(cell_id)
    bin_deg = 0.055  # ~6 km bucket size
    lat_bins = np.floor(lat_centro / bin_deg).astype(np.int32)
    lon_bins = np.floor(lon_centro / bin_deg).astype(np.int32)

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(n):
        buckets[(int(lat_bins[idx]), int(lon_bins[idx]))].append(idx)

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []

    # Prepare shifted bucket arrays for all 9 (di, dj) offsets
    keys = list(buckets.keys())
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            s_list: list[int] = []
            t_list: list[int] = []
            for b_lat, b_lon in keys:
                nbrs = buckets.get((b_lat + di, b_lon + dj))
                if not nbrs:
                    continue
                srcs = buckets[(b_lat, b_lon)]
                for s in srcs:
                    for t in nbrs:
                        if s != t:
                            s_list.append(s)
                            t_list.append(t)
            if s_list:
                src_chunks.append(np.asarray(s_list, dtype=np.int32))
                dst_chunks.append(np.asarray(t_list, dtype=np.int32))

    if not src_chunks:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float32)
        return NeighborGraph(empty_i, empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f)

    src_all = np.concatenate(src_chunks)
    dst_all = np.concatenate(dst_chunks)

    bearing_deg, sin_b, cos_b, dist_km = compute_bearing_and_distance(
        lat_centro[src_all], lon_centro[src_all], lat_centro[dst_all], lon_centro[dst_all]
    )
    keep = (dist_km > 0.1) & (dist_km <= max_dist_km)
    src_idx = src_all[keep]
    dst_idx = dst_all[keep]
    dist_km = dist_km[keep]
    bearing_deg = bearing_deg[keep]
    sin_b = sin_b[keep]
    cos_b = cos_b[keep]

    delta_elev_m = (elevacion_m[dst_idx] - elevacion_m[src_idx]).astype(np.float32)
    slope_deg = np.degrees(np.arctan2(delta_elev_m, np.maximum(dist_km * 1000.0, 1.0))).astype(np.float32)

    return NeighborGraph(
        src_idx=src_idx,
        dst_idx=dst_idx,
        dist_km=dist_km,
        bearing_deg=bearing_deg,
        sin_bearing=sin_b,
        cos_bearing=cos_b,
        delta_elev_m=delta_elev_m,
        slope_deg=slope_deg,
    )


def aggregate_target_probabilities(
    n_cells: int,
    dst_idx: np.ndarray,
    pair_probs: np.ndarray,
) -> np.ndarray:
    """Aggregate directed spread probabilities `p(s -> t)` into `P_spread(t) = 1 - prod_s (1 - p(s -> t))`."""
    if len(dst_idx) == 0:
        return np.zeros(n_cells, dtype=np.float32)
    clipped = np.clip(pair_probs.astype(np.float64), 0.0, 1.0 - 1e-7)
    log_comp = np.log1p(-clipped)
    sum_log_comp = np.zeros(n_cells, dtype=np.float64)
    np.add.at(sum_log_comp, dst_idx, log_comp)
    return (-np.expm1(sum_log_comp)).astype(np.float32)


def fuse_risk_and_spread(
    p_risk: np.ndarray,
    p_spread: np.ndarray,
    weight_spread: float = 1.0,
) -> np.ndarray:
    """Combine independent baseline ignition risk `p_risk` and neighbor propagation `p_spread`."""
    pr = np.clip(p_risk.astype(np.float64), 0.0, 1.0)
    ps = np.clip(p_spread.astype(np.float64) * float(weight_spread), 0.0, 1.0)
    return (1.0 - (1.0 - pr) * (1.0 - ps)).astype(np.float32)


@dataclass
class SpreadPredictor:
    """Self-contained neighbor propagation predictor loaded from `spread_bundle.joblib`."""

    model_version: str
    feature_columns: list[str]
    logit_offset: float
    risk_cutoffs: dict[str, float]
    model_str: str
    wind_directional_gain: float = 0.35

    def __post_init__(self) -> None:
        self._booster = lgb.Booster(model_str=self.model_str)

    @classmethod
    def from_bundle(cls, bundle_path: Path) -> SpreadPredictor:
        payload = joblib.load(bundle_path)
        return cls(
            model_version=str(payload["model_version"]),
            feature_columns=list(payload["feature_columns"]),
            logit_offset=float(payload["logit_offset"]),
            risk_cutoffs={k: float(v) for k, v in payload["risk_cutoffs"].items()},
            model_str=str(payload["model_str"]),
            wind_directional_gain=float(payload.get("wind_directional_gain", 0.35)),
        )

    def predict_pair_probabilities(
        self,
        X: np.ndarray,
        wind_alignment: np.ndarray | None = None,
        wind_speed_kmh: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict calibrated `p(s -> t, H)` for candidate pair feature matrix `X`."""
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        raw = self._booster.predict(np.asarray(X, dtype=np.float32), raw_score=True)
        z = raw + self.logit_offset
        if wind_alignment is not None and wind_speed_kmh is not None:
            wind_factor = np.clip(wind_speed_kmh.astype(np.float64) / 30.0, 0.0, 2.5)
            z = z + self.wind_directional_gain * wind_alignment.astype(np.float64) * wind_factor
        z = np.clip(z, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def categorize(self, prob: np.ndarray) -> np.ndarray:
        """Map spread probabilities to (`Extremo`, `Alto`, `Moderado`, `Bajo`)."""
        c = self.risk_cutoffs
        return np.where(
            prob >= c["Extremo"],
            "Extremo",
            np.where(
                prob >= c["Alto"],
                "Alto",
                np.where(prob >= c["Moderado"], "Moderado", "Bajo"),
            ),
        )
