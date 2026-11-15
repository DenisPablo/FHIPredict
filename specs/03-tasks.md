# FHIPredict — Implementation Tasks (SDD)

- [x] **T1 — Project Scaffold & Model Artifacts Migration**
  - Initialize `FHIPredict/` (`git init`, `.gitignore`, `README.md`).
  - Copy `fire_risk_bundle.joblib` and `fire_risk_metadata.json` from `ModeloPredictivo/models/`.
  - Export `cell_baseline.parquet` (64,653 active cells with `cell_id`, `lat_centro`, `lon_centro`, `lat_clima`, `lon_clima`, `elevacion_m`, fire memory & monthly climatology) from `ModeloPredictivo` so `FHIPredict` has zero runtime dependency on `ModeloPredictivo`.

- [x] **T2 — FastAPI Inference Backend + Open-Meteo Client**
  - Implement `FireRiskPredictor` loader, Open-Meteo batch async client (`past_days=30`, `forecast_days=15`), rolling feature builder, and 1h TTL cache.
  - Expose `GET /api/v1/health` and `GET /api/v1/forecast` (supporting `1d`, `3d`, `7d`, `15d` windows and `day_offset=0..14`, plus `bbox` / `min_risk` filtering).
  - Add pytest suite (mocked Open-Meteo + real bundle inference).

- [x] **T3 — Minimal MapLibre Frontend (FuegoContinuo Identity)**
  - Scaffold Vite + React + TypeScript + Tailwind CSS + MapLibre GL with `DESIGN.md` tokens (`#07100F`, `#0D1816`, `#B8EE68`, `Manrope`/`DM Sans`/`DM Mono`).
  - Full-screen map with risk cell layer, floating header, time-range selector (`24h`, `3d`, `7d`, `15d` + day-by-day slider `Día 1..15`), risk summary chips, and cell inspection popup.
  - Add Vitest unit tests + `tsc --noEmit` check.

- [x] **T4 — End-to-End Verification**
  - Verify backend + frontend locally against live Open-Meteo data, confirm all 15 forecast days and time-range presets render cleanly, and run full test suites.
