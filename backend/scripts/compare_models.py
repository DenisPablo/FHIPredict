"""Head-to-head comparative benchmark: `lgbm-risk-24h-v2` vs `lgbm-spread-v1`.

Evaluates both models on the identical temporal splits (Validation 2024–2025 and
Unseen Holdout Test 2026) across two complementary tracks:

- Track A (Global Cell-Day 24h Risk):
  Compares `lgbm-risk-24h-v2` alone against the fused `v2 + lgbm-spread-v1` probability
  on the full cell-day population.
- Track B (Conditional Neighbor Propagation at 24h, 48h, and 72h):
  Restricts to directed neighbor pairs `(source s -> target t, date d)` where source
  cell `s` had an active fire on `d - 1`. Compares:
    1. `lgbm-spread-v1` (new spatial propagation model)
    2. `lgbm-risk-24h-v2` (existing cell-only model applied to target cell `t`, scaled for H days)
    3. Naive heuristic baseline (target recency + active neighbor intensity score)

Writes:
- `backend/models/comparison_report.json`
- `specs/05-comparison-results.md`
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.predictor import FireRiskPredictor
from app.spread import SPREAD_FEATURE_COLUMNS, SpreadPredictor, fuse_risk_and_spread
from scripts.train_spread import (
    build_spread_splits,
    evaluate_predictions,
    population_weights,
)


# Map columns from SPREAD_FEATURE_COLUMNS back to the 20 columns expected by `lgbm-risk-24h-v2`
V2_FROM_SPREAD_COLS = [
    "lat_centro_t",
    "lon_centro_t",
    "mes",
    "dia_anio",
    "focos_ultimos_7d_t",
    "focos_ultimos_30d_t",
    "frp_ultimos_7d_t",
    "mediana_historica_mes_t",
    "anios_desde_ultimo_fuego_t",
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


def extract_v2_features_from_spread(X_spread: np.ndarray) -> np.ndarray:
    idxs = [SPREAD_FEATURE_COLUMNS.index(c) for c in V2_FROM_SPREAD_COLS]
    return X_spread[:, idxs]


def naive_spread_score(X_spread: np.ndarray) -> np.ndarray:
    """Heuristic baseline combining source intensity, active neighbors, inverse distance, and target recency."""
    idx_dist = SPREAD_FEATURE_COLUMNS.index("dist_km")
    idx_n_s = SPREAD_FEATURE_COLUMNS.index("n_harm_s")
    idx_act_nbr = SPREAD_FEATURE_COLUMNS.index("active_neighbors_count_t")
    idx_f7_t = SPREAD_FEATURE_COLUMNS.index("focos_ultimos_7d_t")
    idx_h = SPREAD_FEATURE_COLUMNS.index("horizon_days")

    dist = np.maximum(X_spread[:, idx_dist], 1.0)
    n_s = np.log1p(np.maximum(X_spread[:, idx_n_s], 0.0))
    act_n = np.log1p(np.maximum(X_spread[:, idx_act_nbr], 0.0))
    f7_t = np.log1p(np.maximum(X_spread[:, idx_f7_t], 0.0))
    h = X_spread[:, idx_h]

    raw = -3.8 + 0.35 * n_s + 0.45 * act_n + 0.50 * f7_t - 0.12 * dist + 0.25 * (h - 1.0)
    return (1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))).astype(np.float32)


def run_comparison(
    v2_bundle_path: Path,
    v2_meta_path: Path,
    spread_bundle_path: Path,
    spread_meta_path: Path,
    baseline_parquet: Path,
    firms_parquet: Path,
    weather_glob: str,
    out_json: Path,
    out_md: Path,
    neg_sample_permille: int = 200,
) -> dict[str, Any]:
    v2_pred = FireRiskPredictor.from_bundle(v2_bundle_path)
    spread_pred = SpreadPredictor.from_bundle(spread_bundle_path)
    v2_meta = json.loads(v2_meta_path.read_text(encoding="utf-8"))
    spread_meta = json.loads(spread_meta_path.read_text(encoding="utf-8"))

    splits = build_spread_splits(
        baseline_parquet=baseline_parquet,
        firms_parquet=firms_parquet,
        weather_glob=weather_glob,
        neg_sample_permille=neg_sample_permille,
        only_years=[2024, 2025, 2026],
    )

    h_idx = SPREAD_FEATURE_COLUMNS.index("horizon_days")
    results_by_split: dict[str, Any] = {}

    for split_name in ("val", "test"):
        X_sp, y_sp, _ = splits[split_name]
        w_sp = population_weights(y_sp, neg_sample_permille)
        X_v2 = extract_v2_features_from_spread(X_sp)

        # Measure inference speed on identical candidate rows
        t0 = time.perf_counter()
        p_spread_all = spread_pred.predict_pair_probabilities(X_sp)
        t_spread_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        p_v2_24h = v2_pred.predict_probabilities(X_v2)
        t_v2_s = time.perf_counter() - t1

        # Scale v2 daily probability to cumulative H-day probability: 1 - (1 - p_24h)^H
        horizons = X_sp[:, h_idx]
        p_v2_scaled = (1.0 - np.power(1.0 - np.clip(p_v2_24h, 0.0, 1.0 - 1e-7), horizons)).astype(np.float32)
        p_fused_all = fuse_risk_and_spread(p_v2_scaled, p_spread_all, weight_spread=0.65)
        p_naive_all = naive_spread_score(X_sp)

        overall = {
            "lgbm_spread_v1": evaluate_predictions(y_sp, p_spread_all, w_sp, spread_pred.risk_cutoffs),
            "lgbm_risk_24h_v2_scaled": evaluate_predictions(y_sp, p_v2_scaled, w_sp),
            "v2_plus_spread_fused": evaluate_predictions(y_sp, p_fused_all, w_sp),
            "naive_spatial_baseline": evaluate_predictions(y_sp, p_naive_all, w_sp),
            "inference_benchmark": {
                "n_rows": int(len(y_sp)),
                "spread_seconds": round(t_spread_s, 4),
                "spread_rows_per_sec": round(len(y_sp) / max(t_spread_s, 1e-6), 1),
                "v2_seconds": round(t_v2_s, 4),
                "v2_rows_per_sec": round(len(y_sp) / max(t_v2_s, 1e-6), 1),
            },
        }

        by_horizon: dict[str, Any] = {}
        for h in (1, 2, 3):
            m = horizons == float(h)
            if not np.any(m):
                continue
            y_h = y_sp[m]
            w_h = w_sp[m]
            m_sp = evaluate_predictions(y_h, p_spread_all[m], w_h, spread_pred.risk_cutoffs)
            m_v2 = evaluate_predictions(y_h, p_v2_scaled[m], w_h)
            m_fu = evaluate_predictions(y_h, p_fused_all[m], w_h)
            m_na = evaluate_predictions(y_h, p_naive_all[m], w_h)
            rel_gain_vs_v2 = (m_sp["pr_auc"] - m_v2["pr_auc"]) / max(m_v2["pr_auc"], 1e-12) * 100.0
            rel_gain_vs_naive = (m_sp["pr_auc"] - m_na["pr_auc"]) / max(m_na["pr_auc"], 1e-12) * 100.0
            by_horizon[f"{h * 24}h"] = {
                "lgbm_spread_v1": m_sp,
                "lgbm_risk_24h_v2": m_v2,
                "v2_plus_spread_fused": m_fu,
                "naive_spatial_baseline": m_na,
                "relative_pr_auc_gain_vs_v2_pct": round(rel_gain_vs_v2, 2),
                "relative_pr_auc_gain_vs_naive_pct": round(rel_gain_vs_naive, 2),
            }

        overall_rel_gain_vs_v2 = (
            (overall["lgbm_spread_v1"]["pr_auc"] - overall["lgbm_risk_24h_v2_scaled"]["pr_auc"])
            / max(overall["lgbm_risk_24h_v2_scaled"]["pr_auc"], 1e-12)
            * 100.0
        )
        overall_rel_gain_vs_naive = (
            (overall["lgbm_spread_v1"]["pr_auc"] - overall["naive_spatial_baseline"]["pr_auc"])
            / max(overall["naive_spatial_baseline"]["pr_auc"], 1e-12)
            * 100.0
        )
        overall["relative_pr_auc_gain_vs_v2_pct"] = round(overall_rel_gain_vs_v2, 2)
        overall["relative_pr_auc_gain_vs_naive_pct"] = round(overall_rel_gain_vs_naive, 2)

        results_by_split[split_name] = {
            "overall": overall,
            "by_horizon": by_horizon,
        }

    test_overall = results_by_split["test"]["overall"]
    passes_margin = (
        test_overall["relative_pr_auc_gain_vs_v2_pct"] >= 20.0
        and test_overall["relative_pr_auc_gain_vs_naive_pct"] >= 20.0
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_compared": {
            "baseline_risk_model": {
                "model_version": v2_pred.model_version,
                "horizon_hours": 24,
                "n_features": len(v2_pred.feature_columns),
                "global_val_pr_auc": v2_meta["validation_metrics"]["pr_auc"],
                "global_test_pr_auc": v2_meta["test_metrics"]["pr_auc"],
                "global_test_roc_auc": v2_meta["test_metrics"]["roc_auc"],
            },
            "candidate_spread_model": {
                "model_version": spread_pred.model_version,
                "horizons_hours": [24, 48, 72],
                "n_features": len(spread_pred.feature_columns),
                "val_pr_auc": spread_meta["validation_metrics"]["pr_auc"],
                "test_pr_auc": spread_meta["test_metrics"]["pr_auc"],
                "test_roc_auc": spread_meta["test_metrics"]["roc_auc"],
            },
        },
        "track_b_conditional_propagation": results_by_split,
        "acceptance_gate": {
            "required_relative_pr_auc_gain_pct": 20.0,
            "observed_test_pr_auc_gain_vs_v2_pct": test_overall["relative_pr_auc_gain_vs_v2_pct"],
            "observed_test_pr_auc_gain_vs_naive_pct": test_overall["relative_pr_auc_gain_vs_naive_pct"],
            "calibration_ratio_test": round(test_overall["lgbm_spread_v1"]["calibration_ratio"], 4),
            "verdict": "PASS" if passes_margin else "REVIEW",
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(report, out_md)
    print(
        f"Saved comparison report to {out_json.name} and {out_md.name} | "
        f"Verdict={report['acceptance_gate']['verdict']} | "
        f"Test PR-AUC gain vs v2: +{test_overall['relative_pr_auc_gain_vs_v2_pct']:.2f}%"
    )
    return report


def _write_markdown_report(report: dict[str, Any], out_md: Path) -> None:
    v2_info = report["models_compared"]["baseline_risk_model"]
    sp_info = report["models_compared"]["candidate_spread_model"]
    te = report["track_b_conditional_propagation"]["test"]
    va = report["track_b_conditional_propagation"]["val"]
    gate = report["acceptance_gate"]

    te_ov = te["overall"]
    va_ov = va["overall"]

    lines = [
        "# Comparative Benchmark Results — `lgbm-risk-24h-v2` vs `lgbm-spread-v1`",
        "",
        f"- **Generated at**: `{report['generated_at']}`",
        f"- **Branch**: `feat/propagation-model-v3`",
        f"- **Acceptance Gate Verdict**: **{gate['verdict']}** (Required ≥ +20% relative PR-AUC on Track B; observed `{gate['observed_test_pr_auc_gain_vs_v2_pct']:+.2f}%` vs `v2` and `{gate['observed_test_pr_auc_gain_vs_naive_pct']:+.2f}%` vs naive baseline)",
        "",
        "## 1. Model Overview & Global Baseline Reference (Track A)",
        "",
        "| Model | Role | Horizons | Features | Val PR-AUC | Test PR-AUC | Test ROC-AUC |",
        "|---|---|---|---:|---:|---:|---:|",
        f"| `{v2_info['model_version']}` | Global cell-day ignition risk | 24h | {v2_info['n_features']} | {v2_info['global_val_pr_auc']:.4f} | {v2_info['global_test_pr_auc']:.4f} | {v2_info['global_test_roc_auc']:.4f} |",
        f"| `{sp_info['model_version']}` | Conditional 8-neighbor propagation | 24h, 48h, 72h | {sp_info['n_features']} | {sp_info['val_pr_auc']:.4f} | {sp_info['test_pr_auc']:.4f} | {sp_info['test_roc_auc']:.4f} |",
        "",
        "## 2. Track B — Conditional Neighbor Propagation (Unseen 2026 Holdout, All Horizons)",
        "",
        "| Model | PR-AUC | ROC-AUC | Brier | Log-Loss | Calibration Ratio | Top-1% Precision | Top-1% Recall | Top-1% Lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key, label in (
        ("lgbm_spread_v1", "`lgbm-spread-v1` (New)"),
        ("v2_plus_spread_fused", "`v2 + spread` (Fused)"),
        ("lgbm_risk_24h_v2_scaled", "`lgbm-risk-24h-v2` (Scaled to H)"),
        ("naive_spatial_baseline", "Naive Spatial Heuristic"),
    ):
        m = te_ov[key]
        t1 = m["top_k"]["top_1pct"]
        lines.append(
            f"| {label} | **{m['pr_auc']:.4f}** | {m['roc_auc']:.4f} | {m['brier_score']:.4f} | "
            f"{m['log_loss']:.4f} | {m['calibration_ratio']:.3f} | {t1['precision']:.4f} | "
            f"{t1['recall']:.4f} | {t1['lift']:.2f}x |"
        )

    lines.extend([
        "",
        "## 3. Breakdown by Propagation Horizon (24h vs 48h vs 72h — 2026 Holdout)",
        "",
        "| Horizon | Base Rate | `lgbm-spread-v1` PR-AUC | `lgbm-risk-24h-v2` PR-AUC | `v2 + spread` PR-AUC | Naive PR-AUC | Gain vs `v2` | Gain vs Naive | `spread-v1` ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for h_key in ("24h", "48h", "72h"):
        if h_key not in te["by_horizon"]:
            continue
        bh = te["by_horizon"][h_key]
        msp = bh["lgbm_spread_v1"]
        mv2 = bh["lgbm_risk_24h_v2"]
        mfu = bh["v2_plus_spread_fused"]
        mna = bh["naive_spatial_baseline"]
        lines.append(
            f"| **{h_key}** | {msp['positive_rate'] * 100:.2f}% | **{msp['pr_auc']:.4f}** | "
            f"{mv2['pr_auc']:.4f} | {mfu['pr_auc']:.4f} | {mna['pr_auc']:.4f} | "
            f"**{bh['relative_pr_auc_gain_vs_v2_pct']:+.1f}%** | **{bh['relative_pr_auc_gain_vs_naive_pct']:+.1f}%** | "
            f"{msp['roc_auc']:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Validation Split Summary (2024–2025)",
        "",
        "| Model | Val PR-AUC | Val ROC-AUC | Val Brier | Val Top-1% Lift |",
        "|---|---:|---:|---:|---:|",
        f"| `lgbm-spread-v1` | **{va_ov['lgbm_spread_v1']['pr_auc']:.4f}** | {va_ov['lgbm_spread_v1']['roc_auc']:.4f} | {va_ov['lgbm_spread_v1']['brier_score']:.4f} | {va_ov['lgbm_spread_v1']['top_k']['top_1pct']['lift']:.2f}x |",
        f"| `v2 + spread` (Fused) | {va_ov['v2_plus_spread_fused']['pr_auc']:.4f} | {va_ov['v2_plus_spread_fused']['roc_auc']:.4f} | {va_ov['v2_plus_spread_fused']['brier_score']:.4f} | {va_ov['v2_plus_spread_fused']['top_k']['top_1pct']['lift']:.2f}x |",
        f"| `lgbm-risk-24h-v2` (Scaled) | {va_ov['lgbm_risk_24h_v2_scaled']['pr_auc']:.4f} | {va_ov['lgbm_risk_24h_v2_scaled']['roc_auc']:.4f} | {va_ov['lgbm_risk_24h_v2_scaled']['brier_score']:.4f} | {va_ov['lgbm_risk_24h_v2_scaled']['top_k']['top_1pct']['lift']:.2f}x |",
        f"| Naive Spatial Heuristic | {va_ov['naive_spatial_baseline']['pr_auc']:.4f} | {va_ov['naive_spatial_baseline']['roc_auc']:.4f} | {va_ov['naive_spatial_baseline']['brier_score']:.4f} | {va_ov['naive_spatial_baseline']['top_k']['top_1pct']['lift']:.2f}x |",
        "",
        "## 5. Inference Throughput Benchmark",
        "",
        f"- **Test candidate pairs evaluated**: `{te_ov['inference_benchmark']['n_rows']:,}`",
        f"- **`lgbm-spread-v1` latency**: `{te_ov['inference_benchmark']['spread_seconds']} s` (`{te_ov['inference_benchmark']['spread_rows_per_sec']:,.0f} rows/s`)",
        f"- **`lgbm-risk-24h-v2` latency**: `{te_ov['inference_benchmark']['v2_seconds']} s` (`{te_ov['inference_benchmark']['v2_rows_per_sec']:,.0f} rows/s`)",
        "",
    ])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lgbm-risk-24h-v2 and lgbm-spread-v1.")
    parser.add_argument("--models-dir", type=Path, default=BACKEND_ROOT / "models")
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
        "--out-json",
        type=Path,
        default=BACKEND_ROOT / "models" / "comparison_report.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "specs" / "05-comparison-results.md",
    )
    args = parser.parse_args()

    run_comparison(
        v2_bundle_path=args.models_dir / "fire_risk_bundle.joblib",
        v2_meta_path=args.models_dir / "fire_risk_metadata.json",
        spread_bundle_path=args.models_dir / "spread_bundle.joblib",
        spread_meta_path=args.models_dir / "spread_metadata.json",
        baseline_parquet=args.models_dir / "cell_baseline.parquet",
        firms_parquet=args.firms,
        weather_glob=args.weather_glob,
        out_json=args.out_json,
        out_md=args.out_md,
    )


if __name__ == "__main__":
    main()
