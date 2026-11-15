# Comparative Benchmark Results — `lgbm-risk-24h-v2` vs `lgbm-spread-v1`

- **Generated at**: `2026-11-15T18:43:55.750049+00:00`
- **Branch**: `feat/propagation-model-v3`
- **Acceptance Gate Verdict**: **REVIEW** (Required ≥ +20% relative PR-AUC on Track B; observed `+6.19%` vs `v2` and `+28.72%` vs naive baseline)

## 1. Model Overview & Global Baseline Reference (Track A)

| Model | Role | Horizons | Features | Val PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| `lgbm-risk-24h-v2` | Global cell-day ignition risk | 24h | 20 | 0.0691 | 0.0785 | 0.9196 |
| `lgbm-spread-v1` | Conditional 8-neighbor propagation | 24h, 48h, 72h | 32 | 0.2922 | 0.3945 | 0.8749 |

## 2. Track B — Conditional Neighbor Propagation (Unseen 2026 Holdout, All Horizons)

| Model | PR-AUC | ROC-AUC | Brier | Log-Loss | Calibration Ratio | Top-1% Precision | Top-1% Recall | Top-1% Lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lgbm-spread-v1` (New) | **0.3933** | 0.8747 | 0.0333 | 0.1287 | 0.875 | 0.6947 | 0.1577 | 15.77x |
| `v2 + spread` (Fused) | **0.3972** | 0.8720 | 0.0330 | 0.1284 | 1.086 | 0.7272 | 0.1651 | 16.51x |
| `lgbm-risk-24h-v2` (Scaled to H) | **0.3703** | 0.8273 | 0.0340 | 0.1507 | 0.618 | 0.7059 | 0.1604 | 16.02x |
| Naive Spatial Heuristic | **0.3055** | 0.8058 | 0.0367 | 0.1481 | 0.973 | 0.5954 | 0.1352 | 13.52x |

## 3. Breakdown by Propagation Horizon (24h vs 48h vs 72h — 2026 Holdout)

| Horizon | Base Rate | `lgbm-spread-v1` PR-AUC | `lgbm-risk-24h-v2` PR-AUC | `v2 + spread` PR-AUC | Naive PR-AUC | Gain vs `v2` | Gain vs Naive | `spread-v1` ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **24h** | 3.51% | **0.4148** | 0.4079 | 0.4277 | 0.2970 | **+1.7%** | **+39.7%** | 0.8929 |
| **48h** | 4.54% | **0.3869** | 0.3825 | 0.4020 | 0.3006 | **+1.1%** | **+28.7%** | 0.8702 |
| **72h** | 5.16% | **0.3863** | 0.3815 | 0.4009 | 0.3193 | **+1.3%** | **+21.0%** | 0.8601 |

## 4. Validation Split Summary (2024–2025)

| Model | Val PR-AUC | Val ROC-AUC | Val Brier | Val Top-1% Lift |
|---|---:|---:|---:|---:|
| `lgbm-spread-v1` | **0.2927** | 0.8343 | 0.0418 | 11.25x |
| `v2 + spread` (Fused) | 0.2804 | 0.8310 | 0.0438 | 10.45x |
| `lgbm-risk-24h-v2` (Scaled) | 0.2571 | 0.7973 | 0.0439 | 9.93x |
| Naive Spatial Heuristic | 0.2113 | 0.7599 | 0.0452 | 8.36x |

## 5. Inference Throughput Benchmark

- **Test candidate pairs evaluated**: `65,111`
- **`lgbm-spread-v1` latency**: `0.1527 s` (`426,487 rows/s`)
- **`lgbm-risk-24h-v2` latency**: `0.206 s` (`316,116 rows/s`)
