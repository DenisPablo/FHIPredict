"""Train the multi-horizon (24h/48h/72h) spatial hotspot propagation model (`lgbm-spread-v1`).

Builds leak-free directed neighbor pairs `(source s -> target t, forecast date d)`
where source cell `s` had a harmonized fire event on `d - 1` and target cell `t`
is in the 8-neighborhood of `s`. Evaluates cumulative propagation labels over
`H in {1, 2, 3}` days (24h, 48h, 72h) on the same temporal splits as `lgbm-risk-24h-v2`:
- Train: 2018-01-01 .. 2023-12-31
- Val:   2024-01-01 .. 2025-12-31
- Test:  2026-01-01 .. 2026-12-31
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pyarrow as pa

# Ensure `app` is importable when run as `python scripts/train_spread.py`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.service import load_baseline
from app.spread import SPREAD_FEATURE_COLUMNS, build_neighbor_graph


RISK_CATEGORY_TOP_FRACTIONS = {"Extremo": 0.01, "Alto": 0.05, "Moderado": 0.15}
TOP_K_FRACTIONS = (0.001, 0.01, 0.05)


def prior_logit_offset(neg_sample_permille: int, scale_pos_weight: float = 1.0) -> float:
    return float(np.log(neg_sample_permille / 1000.0) - np.log(scale_pos_weight))


def to_real_probability(raw_margin: np.ndarray, logit_offset: float) -> np.ndarray:
    z = np.clip(np.asarray(raw_margin, dtype=np.float64) + logit_offset, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def population_weights(y_true: np.ndarray, neg_sample_permille: int) -> np.ndarray:
    return np.where(y_true == 1, 1.0, 1000.0 / neg_sample_permille)


def weighted_top_cutoff(prob: np.ndarray, weights: np.ndarray, top_fraction: float) -> float:
    order = np.argsort(-prob)
    cum_w = np.cumsum(weights[order])
    k = min(int(np.searchsorted(cum_w, top_fraction * cum_w[-1])), len(order) - 1)
    return float(prob[order][k])


def derive_risk_cutoffs(prob: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        name: weighted_top_cutoff(prob, weights, frac)
        for name, frac in RISK_CATEGORY_TOP_FRACTIONS.items()
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
    risk_cutoffs: dict[str, float] | None = None,
) -> dict[str, Any]:
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    pos = (y_true == 1).astype(np.float64)
    total_w = float(w.sum())
    total_pos = float((pos * w).sum())
    total_neg = max(total_w - total_pos, 1e-12)
    base_rate = total_pos / max(total_w, 1e-12)
    mean_pred = float(np.sum(y_prob * w) / max(total_w, 1e-12))

    order = np.argsort(-y_prob, kind="mergesort")
    w_ord = w[order]
    pos_ord = pos[order] * w_ord
    neg_ord = (1.0 - pos[order]) * w_ord
    cum_w = np.cumsum(w_ord)
    cum_pos = np.cumsum(pos_ord)
    cum_neg = np.cumsum(neg_ord)

    tpr = np.concatenate(([0.0], cum_pos / max(total_pos, 1e-12)))
    fpr = np.concatenate(([0.0], cum_neg / total_neg))
    roc_auc = float(np.trapezoid(tpr, fpr))

    prec_curve = cum_pos / np.maximum(cum_w, 1e-12)
    pr_auc = float(np.sum(prec_curve * (pos_ord / max(total_pos, 1e-12))))

    brier = float(np.sum(w * (y_prob - pos) ** 2) / max(total_w, 1e-12))
    p_clip = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
    logloss = float(
        -np.sum(w * (pos * np.log(p_clip) + (1.0 - pos) * np.log(1.0 - p_clip))) / max(total_w, 1e-12)
    )

    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "positive_rate": base_rate,
        "mean_predicted_probability": mean_pred,
        "calibration_ratio": mean_pred / max(base_rate, 1e-12),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": logloss,
        "top_k": {},
    }

    for frac in TOP_K_FRACTIONS:
        k = min(int(np.searchsorted(cum_w, frac * total_w)), len(order) - 1)
        w_k = float(cum_w[k])
        tp_w = float(cum_pos[k])
        prec = tp_w / max(w_k, 1e-12)
        rec = tp_w / max(total_pos, 1e-12)
        lift = prec / max(base_rate, 1e-12)
        label = f"top_{frac * 100:g}pct"
        metrics["top_k"][label] = {
            "precision": prec,
            "recall": rec,
            "lift": lift,
        }

    if risk_cutoffs is not None:
        cats: dict[str, Any] = {}
        labels = np.full(len(y_prob), "Bajo", dtype=object)
        for name in ("Moderado", "Alto", "Extremo"):
            labels[y_prob >= risk_cutoffs[name]] = name
        for name in ("Extremo", "Alto", "Moderado", "Bajo"):
            m = labels == name
            if not np.any(m):
                cats[name] = {
                    "share_of_pairs": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "lift": 0.0,
                    "mean_predicted_probability": 0.0,
                }
                continue
            w_c = float(w[m].sum())
            pos_c = float((pos[m] * w[m]).sum())
            prec_c = pos_c / max(w_c, 1e-12)
            cats[name] = {
                "share_of_pairs": w_c / max(total_w, 1e-12),
                "precision": prec_c,
                "recall": pos_c / max(total_pos, 1e-12),
                "lift": prec_c / max(base_rate, 1e-12),
                "mean_predicted_probability": float(np.sum(y_prob[m] * w[m]) / max(w_c, 1e-12)),
            }
        metrics["categories"] = cats

    return metrics


def build_spread_splits(
    baseline_parquet: Path,
    firms_parquet: Path,
    weather_glob: str,
    train_end: str = "2023-12-31",
    val_start: str = "2024-01-01",
    val_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
    test_end: str = "2026-12-31",
    neg_sample_permille: int = 200,
    only_years: list[int] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build `(X, y, dates)` for `train`, `val`, and `test` across horizons `H in {1, 2, 3}`."""
    t0 = time.perf_counter()
    baseline = load_baseline(baseline_parquet.as_posix())
    graph = build_neighbor_graph(
        cell_id=baseline.cell_id,
        lat_centro=baseline.lat_centro,
        lon_centro=baseline.lon_centro,
        elevacion_m=baseline.elevacion_m,
    )
    print(
        f"[1/3] Built 8-neighbor graph over {len(baseline.cell_id):,} cells: "
        f"{len(graph.src_idx):,} directed edges ({time.perf_counter() - t0:.2f}s)"
    )

    edges_table = pa.table(
        {
            "src_cell_id": baseline.cell_id[graph.src_idx],
            "dst_cell_id": baseline.cell_id[graph.dst_idx],
            "dist_km": graph.dist_km,
            "sin_bearing": graph.sin_bearing,
            "cos_bearing": graph.cos_bearing,
            "delta_elev_m": graph.delta_elev_m,
            "slope_deg": graph.slope_deg,
        }
    )
    cells_table = pa.table(
        {
            "cell_id": baseline.cell_id,
            "lat_centro": baseline.lat_centro,
            "lon_centro": baseline.lon_centro,
            "elevacion_m": baseline.elevacion_m,
            "lat_clima": np.round(baseline.lat_centro * 2.0) / 2.0,
            "lon_clima": np.round(baseline.lon_centro / 0.625) * 0.625,
        }
    )

    base_feature_cols = [c for c in SPREAD_FEATURE_COLUMNS if c != "horizon_days"]

    threads = min(8, os.cpu_count() or 4)
    splits: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    with tempfile.TemporaryDirectory(prefix="duckdb_spread_") as spill_dir:
        con = duckdb.connect(database=":memory:")
        con.execute(f"SET threads TO {threads};")
        con.execute("SET memory_limit = '2GB';")
        con.execute(f"SET temp_directory = '{Path(spill_dir).as_posix()}';")

        con.register("edges_arrow", edges_table)
        con.register("cells_arrow", cells_table)

        con.execute(f"""
        CREATE TEMP TABLE neighbor_edges AS SELECT * FROM edges_arrow;
        CREATE TEMP TABLE active_cells AS SELECT * FROM cells_arrow;

        CREATE TEMP TABLE pos_events AS
        SELECT
            cell_id::INT AS cell_id,
            fecha::DATE AS fecha,
            n_harmonized::FLOAT AS n_harm,
            phi_harmonized::FLOAT AS phi_harm
        FROM read_parquet('{firms_parquet.as_posix()}')
        WHERE n_harmonized >= 1
          AND phi_harmonized >= 10.0
          AND fecha BETWEEN DATE '2011-01-01' AND DATE '{test_end}';

        CREATE TEMP TABLE weather_features AS
        WITH weather_base AS (
            SELECT
                lat_clima::FLOAT AS lat_clima,
                lon_clima::FLOAT AS lon_clima,
                fecha::DATE AS fecha,
                temperature_2m_max::FLOAT AS temperature_2m_max,
                relative_humidity_2m::FLOAT AS relative_humidity_2m,
                wind_speed_10m_max::FLOAT AS wind_speed_10m_max,
                precipitation_sum::FLOAT AS precipitation_sum,
                shortwave_radiation_sum::FLOAT AS shortwave_radiation_sum,
                CASE WHEN COALESCE(precipitation_sum, 0.0) >= 1.0 THEN 1 ELSE 0 END AS is_rain_day
            FROM read_parquet('{weather_glob}')
            WHERE fecha BETWEEN DATE '2018-01-01' AND DATE '{test_end}'
        ),
        weather_streaks AS (
            SELECT
                *,
                SUM(is_rain_day) OVER (
                    PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ) AS rain_group
            FROM weather_base
        )
        SELECT
            lat_clima,
            lon_clima,
            fecha,
            temperature_2m_max,
            relative_humidity_2m,
            wind_speed_10m_max,
            precipitation_sum,
            shortwave_radiation_sum,
            SUM(COALESCE(precipitation_sum, 0.0)) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_7d,
            SUM(COALESCE(precipitation_sum, 0.0)) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_15d,
            SUM(COALESCE(precipitation_sum, 0.0)) OVER (
                PARTITION BY lat_clima, lon_clima ORDER BY fecha
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            )::FLOAT AS lluvia_acum_30d,
            CASE
                WHEN is_rain_day = 1 THEN 0::FLOAT
                ELSE (ROW_NUMBER() OVER (
                    PARTITION BY lat_clima, lon_clima, rain_group ORDER BY fecha
                ) - 1)::FLOAT
            END AS dias_consecutivos_sin_lluvia,
            (
                0.6108 * EXP((17.27 * temperature_2m_max) / (temperature_2m_max + 237.3))
                * (1.0 - LEAST(GREATEST(relative_humidity_2m, 0.0), 100.0) / 100.0)
            )::FLOAT AS vpd_approx
        FROM weather_streaks;
        """)

        years = [
            int(r[0])
            for r in con.sql(
                "SELECT DISTINCT EXTRACT(YEAR FROM fecha)::INT AS y FROM weather_features ORDER BY y"
            ).fetchall()
        ]
        if only_years is not None:
            allowed = set(only_years)
            years = [y for y in years if y in allowed]

        print(f"[2/3] Extracting directed propagation pairs across {len(years)} yearly batches...")
        for year in years:
            y_start = f"{year}-01-01"
            y_end = min(f"{year}-12-31", test_end)
            query = f"""
            WITH cell_month_clim AS (
                SELECT
                    cell_id,
                    EXTRACT(MONTH FROM fecha)::SMALLINT AS mes,
                    MEDIAN(n_harm)::FLOAT AS mediana_historica_mes
                FROM pos_events
                WHERE fecha < DATE '{y_start}'
                GROUP BY cell_id, EXTRACT(MONTH FROM fecha)
            ),
            active_sources AS (
                -- Source cell s active on d - 1 => forecast window starts on fecha = d
                SELECT
                    p.cell_id AS src_cell_id,
                    (p.fecha + INTERVAL 1 DAY)::DATE AS fecha,
                    p.n_harm AS n_harm_s,
                    p.phi_harm AS phi_harm_s,
                    COALESCE((
                        SELECT SUM(p2.n_harm)
                        FROM pos_events p2
                        WHERE p2.cell_id = p.cell_id
                          AND p2.fecha BETWEEN p.fecha - INTERVAL 6 DAY AND p.fecha
                    ), p.n_harm)::FLOAT AS focos_7d_s
                FROM pos_events p
                WHERE (p.fecha + INTERVAL 1 DAY)::DATE BETWEEN DATE '{y_start}' AND DATE '{y_end}'
            ),
            raw_pairs AS (
                SELECT
                    s.src_cell_id,
                    e.dst_cell_id,
                    s.fecha,
                    e.dist_km,
                    e.sin_bearing,
                    e.cos_bearing,
                    e.delta_elev_m,
                    e.slope_deg,
                    s.n_harm_s,
                    s.phi_harm_s,
                    s.focos_7d_s,
                    COUNT(*) OVER (PARTITION BY e.dst_cell_id, s.fecha)::FLOAT AS active_neighbors_count_t,
                    SUM(s.n_harm_s) OVER (PARTITION BY e.dst_cell_id, s.fecha)::FLOAT AS neighbors_focos_sum_t,
                    SUM(s.phi_harm_s) OVER (PARTITION BY e.dst_cell_id, s.fecha)::FLOAT AS neighbors_frp_sum_t,
                    CASE WHEN EXISTS (
                        SELECT 1 FROM pos_events pt
                        WHERE pt.cell_id = e.dst_cell_id AND pt.fecha = s.fecha
                    ) THEN 1::TINYINT ELSE 0::TINYINT END AS y_h1,
                    CASE WHEN EXISTS (
                        SELECT 1 FROM pos_events pt
                        WHERE pt.cell_id = e.dst_cell_id
                          AND pt.fecha BETWEEN s.fecha AND s.fecha + INTERVAL 1 DAY
                    ) THEN 1::TINYINT ELSE 0::TINYINT END AS y_h2,
                    CASE WHEN EXISTS (
                        SELECT 1 FROM pos_events pt
                        WHERE pt.cell_id = e.dst_cell_id
                          AND pt.fecha BETWEEN s.fecha AND s.fecha + INTERVAL 2 DAY
                    ) THEN 1::TINYINT ELSE 0::TINYINT END AS y_h3
                FROM active_sources s
                JOIN neighbor_edges e ON s.src_cell_id = e.src_cell_id
            ),
            sampled_pairs AS (
                SELECT *
                FROM raw_pairs
                WHERE y_h3 = 1
                   OR (ABS(HASH(src_cell_id, dst_cell_id, fecha)) % 1000) < {neg_sample_permille}
            ),
            distinct_targets AS (
                SELECT DISTINCT dst_cell_id AS cell_id, fecha
                FROM sampled_pairs
            ),
            target_memory AS (
                SELECT
                    dt.cell_id,
                    dt.fecha,
                    COALESCE(SUM(CASE
                        WHEN p.fecha BETWEEN dt.fecha - INTERVAL 7 DAY AND dt.fecha - INTERVAL 1 DAY
                        THEN p.n_harm ELSE 0.0
                    END), 0.0)::FLOAT AS focos_ultimos_7d_t,
                    COALESCE(SUM(CASE
                        WHEN p.fecha BETWEEN dt.fecha - INTERVAL 30 DAY AND dt.fecha - INTERVAL 1 DAY
                        THEN p.n_harm ELSE 0.0
                    END), 0.0)::FLOAT AS focos_ultimos_30d_t,
                    COALESCE(SUM(CASE
                        WHEN p.fecha BETWEEN dt.fecha - INTERVAL 7 DAY AND dt.fecha - INTERVAL 1 DAY
                        THEN p.phi_harm ELSE 0.0
                    END), 0.0)::FLOAT AS frp_ultimos_7d_t,
                    COALESCE(
                        DATEDIFF('day', MAX(CASE WHEN p.fecha < dt.fecha THEN p.fecha END), dt.fecha) / 365.25,
                        10.0
                    )::FLOAT AS anios_desde_ultimo_fuego_t
                FROM distinct_targets dt
                LEFT JOIN pos_events p
                    ON dt.cell_id = p.cell_id
                   AND p.fecha < dt.fecha
                GROUP BY dt.cell_id, dt.fecha
            )
            SELECT
                sp.fecha,
                sp.y_h1,
                sp.y_h2,
                sp.y_h3,
                ac.lat_centro AS lat_centro_t,
                ac.lon_centro AS lon_centro_t,
                EXTRACT(MONTH FROM sp.fecha)::FLOAT AS mes,
                EXTRACT(DOY FROM sp.fecha)::FLOAT AS dia_anio,
                sp.dist_km,
                sp.sin_bearing,
                sp.cos_bearing,
                sp.delta_elev_m,
                sp.slope_deg,
                sp.n_harm_s,
                sp.phi_harm_s,
                sp.focos_7d_s,
                tm.focos_ultimos_7d_t,
                tm.focos_ultimos_30d_t,
                tm.frp_ultimos_7d_t,
                COALESCE(cmc.mediana_historica_mes, 0.0)::FLOAT AS mediana_historica_mes_t,
                tm.anios_desde_ultimo_fuego_t,
                sp.active_neighbors_count_t,
                sp.neighbors_focos_sum_t,
                sp.neighbors_frp_sum_t,
                ac.elevacion_m AS elevacion_m_t,
                wf.temperature_2m_max AS temperature_2m_max_t,
                wf.relative_humidity_2m AS relative_humidity_2m_t,
                wf.wind_speed_10m_max AS wind_speed_10m_max_t,
                wf.precipitation_sum AS precipitation_sum_t,
                wf.shortwave_radiation_sum AS shortwave_radiation_sum_t,
                wf.lluvia_acum_7d AS lluvia_acum_7d_t,
                wf.lluvia_acum_15d AS lluvia_acum_15d_t,
                wf.lluvia_acum_30d AS lluvia_acum_30d_t,
                wf.dias_consecutivos_sin_lluvia AS dias_consecutivos_sin_lluvia_t,
                wf.vpd_approx AS vpd_approx_t
            FROM sampled_pairs sp
            JOIN target_memory tm
              ON sp.dst_cell_id = tm.cell_id AND sp.fecha = tm.fecha
            JOIN active_cells ac
              ON sp.dst_cell_id = ac.cell_id
            JOIN weather_features wf
              ON ac.lat_clima = wf.lat_clima
             AND ac.lon_clima = wf.lon_clima
             AND sp.fecha = wf.fecha
            LEFT JOIN cell_month_clim cmc
              ON sp.dst_cell_id = cmc.cell_id
             AND EXTRACT(MONTH FROM sp.fecha)::SMALLINT = cmc.mes
            """
            tbl = con.sql(query).to_arrow_table()
            if tbl.num_rows == 0:
                continue

            fechas = tbl["fecha"].to_numpy()
            y_by_h = {
                1: tbl["y_h1"].to_numpy().astype(np.int32),
                2: tbl["y_h2"].to_numpy().astype(np.int32),
                3: tbl["y_h3"].to_numpy().astype(np.int32),
            }
            col_arrays = {c: tbl[c].to_numpy().astype(np.float32) for c in base_feature_cols}

            # Expand across H in {1, 2, 3}
            for h in (1, 2, 3):
                y_h = y_by_h[h]
                # Keep all positives for horizon h, and sampled negatives (where y_h == 0)
                if h < 3:
                    # Rows with y_h3 == 1 but y_h == 0 were included unconditionally; downsample them at the same rate
                    # using deterministic index mod so negative sampling rate is uniform across horizons.
                    idx_arr = np.arange(len(y_h), dtype=np.int64)
                    keep_h = (y_h == 1) | (tbl["y_h3"].to_numpy() == 0) | ((idx_arr * 2654435761 % 1000) < neg_sample_permille)
                else:
                    keep_h = np.ones(len(y_h), dtype=bool)

                n_k = int(np.sum(keep_h))
                if n_k == 0:
                    continue
                h_col = np.full(n_k, float(h), dtype=np.float32)
                x_cols = []
                for c in SPREAD_FEATURE_COLUMNS:
                    if c == "horizon_days":
                        x_cols.append(h_col)
                    else:
                        x_cols.append(col_arrays[c][keep_h])
                X_h = np.column_stack(x_cols)
                y_k = y_h[keep_h]
                f_k = fechas[keep_h]

                tr_m = f_k <= np.datetime64(train_end)
                va_m = (f_k >= np.datetime64(val_start)) & (f_k <= np.datetime64(val_end))
                te_m = (f_k >= np.datetime64(test_start)) & (f_k <= np.datetime64(test_end))

                if np.any(tr_m):
                    splits["train"].append((X_h[tr_m], y_k[tr_m], f_k[tr_m]))
                if np.any(va_m):
                    splits["val"].append((X_h[va_m], y_k[va_m], f_k[va_m]))
                if np.any(te_m):
                    splits["test"].append((X_h[te_m], y_k[te_m], f_k[te_m]))

            print(f"  - year {year}: {tbl.num_rows:,} base pairs processed")

        con.close()

    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k, chunks in splits.items():
        if chunks:
            out[k] = (
                np.vstack([c[0] for c in chunks]),
                np.concatenate([c[1] for c in chunks]),
                np.concatenate([c[2] for c in chunks]),
            )
        else:
            out[k] = (
                np.empty((0, len(SPREAD_FEATURE_COLUMNS)), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype="datetime64[D]"),
            )
    return out


def train_and_export(
    baseline_parquet: Path,
    firms_parquet: Path,
    weather_glob: str,
    out_bundle: Path,
    out_metadata: Path,
    neg_sample_permille: int = 200,
) -> dict[str, Any]:
    splits = build_spread_splits(
        baseline_parquet=baseline_parquet,
        firms_parquet=firms_parquet,
        weather_glob=weather_glob,
        neg_sample_permille=neg_sample_permille,
    )
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, _ = splits["test"]

    print(
        f"[3/3] Training lgbm-spread-v1 | "
        f"train={len(y_train):,} (pos={int(y_train.sum()):,}) | "
        f"val={len(y_val):,} (pos={int(y_val.sum()):,}) | "
        f"test={len(y_test):,} (pos={int(y_test.sum()):,})"
    )

    # Monotone constraint +1 on `horizon_days` so P(24h) <= P(48h) <= P(72h)
    monotone = [1 if c == "horizon_days" else 0 for c in SPREAD_FEATURE_COLUMNS]

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=SPREAD_FEATURE_COLUMNS)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=SPREAD_FEATURE_COLUMNS, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "monotone_constraints": monotone,
        "n_jobs": min(8, os.cpu_count() or 4),
        "verbose": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )

    logit_offset = prior_logit_offset(neg_sample_permille, 1.0)
    val_prob = to_real_probability(booster.predict(X_val, raw_score=True), logit_offset)
    val_w = population_weights(y_val, neg_sample_permille)
    cutoffs = derive_risk_cutoffs(val_prob, val_w)

    val_metrics = evaluate_predictions(y_val, val_prob, val_w, cutoffs)
    test_prob = to_real_probability(booster.predict(X_test, raw_score=True), logit_offset)
    test_w = population_weights(y_test, neg_sample_permille)
    test_metrics = evaluate_predictions(y_test, test_prob, test_w, cutoffs)

    # Per-horizon breakdown on test
    h_col_idx = SPREAD_FEATURE_COLUMNS.index("horizon_days")
    horizon_test_metrics: dict[str, Any] = {}
    for h in (1, 2, 3):
        m_h = X_test[:, h_col_idx] == float(h)
        if np.any(m_h):
            horizon_test_metrics[f"{h * 24}h"] = evaluate_predictions(
                y_test[m_h], test_prob[m_h], test_w[m_h], cutoffs
            )

    model_str = booster.model_to_string()
    bundle_payload = {
        "model_version": "lgbm-spread-v1",
        "feature_columns": SPREAD_FEATURE_COLUMNS,
        "logit_offset": logit_offset,
        "risk_cutoffs": cutoffs,
        "wind_directional_gain": 0.35,
        "model_str": model_str,
    }
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle_payload, out_bundle, compress=3)

    importance_split = {
        col: int(v)
        for col, v in zip(SPREAD_FEATURE_COLUMNS, booster.feature_importance(importance_type="split"))
    }
    metadata = {
        "model_version": "lgbm-spread-v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizons_hours": [24, 48, 72],
        "positive_label_rule": "target neighbor t has (n_harmonized >= 1 AND phi_harmonized >= 10.0) in [d, d + H - 1] given source s active on d - 1",
        "split": {
            "train": "2018-01-01..2023-12-31",
            "val": "2024-01-01..2025-12-31",
            "test": "2026-01-01..2026-12-31",
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "train_pos": int(y_train.sum()),
            "val_pos": int(y_val.sum()),
            "test_pos": int(y_test.sum()),
            "neg_sample_permille": neg_sample_permille,
            "best_iteration": int(booster.best_iteration or booster.current_iteration()),
        },
        "probability": {
            "meaning": "Real-world probability that active source cell s propagates fire to neighbor cell t within H days",
            "method": "prior_correction",
            "neg_sample_permille": neg_sample_permille,
            "logit_offset": logit_offset,
        },
        "risk_cutoffs": cutoffs,
        "features": SPREAD_FEATURE_COLUMNS,
        "feature_importance_split": importance_split,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_horizon": horizon_test_metrics,
    }
    out_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Saved {out_bundle.name} ({out_bundle.stat().st_size:,} bytes) | "
        f"val PR-AUC={val_metrics['pr_auc']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f} | "
        f"test PR-AUC={test_metrics['pr_auc']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lgbm-spread-v1 propagation model.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BACKEND_ROOT / "models" / "cell_baseline.parquet",
    )
    parser.add_argument(
        "--firms",
        type=Path,
        default=Path("C:/Users/Denis/Documents/codigo/ModeloPredictivo/firms_argentina_harmonized.parquet"),
    )
    parser.add_argument(
        "--weather-glob",
        type=str,
        default="C:/Users/Denis/Documents/codigo/ModeloPredictivo/data/weather/nasapower/nasapower_*.parquet",
    )
    parser.add_argument(
        "--out-bundle",
        type=Path,
        default=BACKEND_ROOT / "models" / "spread_bundle.joblib",
    )
    parser.add_argument(
        "--out-metadata",
        type=Path,
        default=BACKEND_ROOT / "models" / "spread_metadata.json",
    )
    parser.add_argument("--neg-sample-permille", type=int, default=200)
    args = parser.parse_args()

    train_and_export(
        baseline_parquet=args.baseline,
        firms_parquet=args.firms,
        weather_glob=args.weather_glob,
        out_bundle=args.out_bundle,
        out_metadata=args.out_metadata,
        neg_sample_permille=args.neg_sample_permille,
    )


if __name__ == "__main__":
    main()
