# FHIPredict — Technical Architecture (SDD)

## 1. Repository Structure
```text
FHIPredict/
├── specs/
│   ├── 01-spec.md
│   ├── 02-architecture.md
│   └── 03-tasks.md
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, CORS, health, forecast endpoints
│   │   ├── config.py          # Settings (model path, cache TTL, Open-Meteo URL)
│   │   ├── predictor.py       # FireRiskPredictor loader (from fire_risk_bundle.joblib)
│   │   ├── openmeteo.py       # Async Open-Meteo client + 30d/15d feature builder
│   │   └── cache.py           # In-memory TTL cache for forecast matrix (e.g. 1h)
│   ├── models/
│   │   ├── fire_risk_bundle.joblib
│   │   ├── fire_risk_metadata.json
│   │   └── cell_baseline.parquet   # 64,653 cells: cell_id, lat/lon, elevacion_m, fire memory
│   ├── tests/
│   └── pyproject.toml
└── frontend/
    ├── src/
    │   ├── api/               # Typed client for /api/v1/forecast & /api/v1/health
    │   ├── components/        # RiskMap (MapLibre GL), TimeRangeBar, RiskLegend, CellTooltip
    │   ├── styles/            # FuegoContinuo tokens (Tailwind + CSS vars)
    │   └── App.tsx
    ├── index.html
    ├── package.json
    └── vite.config.ts         # Dev proxy /api -> http://localhost:8000
```

---

## 2. Inference & Open-Meteo Data Flow

### Model Features Required (`lgbm-risk-24h-v2`, 20 features)
1. **Static / Cell Baseline** (from `cell_baseline.parquet`, pre-exported from `ModeloPredictivo`):
   - `lat_centro`, `lon_centro`, `elevacion_m`
   - `focos_ultimos_7d`, `focos_ultimos_30d`, `frp_ultimos_7d`, `mediana_historica_mes` (12-month lookup or current), `anios_desde_ultimo_fuego`
2. **Calendar** (from target forecast date `d`):
   - `mes`, `dia_anio`
3. **Dynamic Weather** (from Open-Meteo `/v1/forecast`):
   - Query params: `daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum,shortwave_radiation_sum`, `past_days=30`, `forecast_days=15`, `timezone=America/Argentina/Buenos_Aires`.
   - From the 45-day series (`-30d .. +15d`), compute per weather grid point and forecast day:
     - `temperature_2m_max`, `relative_humidity_2m`, `wind_speed_10m_max`, `precipitation_sum`, `shortwave_radiation_sum`
     - `lluvia_acum_7d`, `lluvia_acum_15d`, `lluvia_acum_30d`
     - `dias_consecutivos_sin_lluvia` (streak of days with `precipitation_sum < 1.0`)
     - `vpd_approx` (Magnus formula using `temperature_2m_max` and `relative_humidity_2m`)

### Weather Grid Strategy (Performance)
- `ModeloPredictivo` maps each 5×5 km cell to a `0.5° × 0.625°` weather grid (`lat_clima`, `lon_clima`).
- Across Argentina, there are ~900–1,100 unique `(lat_clima, lon_clima)` points.
- Open-Meteo supports **multi-coordinate batch requests** (comma-separated `latitude=...&longitude=...`, e.g., chunks of 100 coordinates per HTTP call → ~10 parallel async requests = < 2 seconds total!).
- Results are cached in memory (`TTL = 3600s` by default) and scored in batch with LightGBM across all 64,653 cells × 15 days (~970k rows, < 1.5s in NumPy/LightGBM).

---

## 3. Backend API Contract
- `GET /api/v1/health` → `{ status, model_version, cells_total, forecast_ready, generated_at }`
- `GET /api/v1/forecast?mode=window&days=1|3|7|15` or `?mode=day&day_offset=0..14&min_risk=Moderado`
  - Returns GeoJSON `FeatureCollection` (or compact JSON array of cells above threshold / within optional `bbox`) with `cell_id`, `lat`, `lon`, `riesgo_pct`, `categoria_riesgo`, and weather summary.
