# FHIPredict — Product & Functional Specification (SDD)

## 1. Purpose
Standalone, lightweight web application for **live fire-risk prediction across Argentina** using the pre-trained `lgbm-risk-24h-v2` LightGBM model (exported from `ModeloPredictivo`) combined with real-time weather forecasts from **Open-Meteo**.

The application is intentionally minimal: **a single full-viewport interactive map** with the **FuegoContinuo** visual identity (`DESIGN.md`), focused exclusively on visualizing predicted fire-risk cells across selectable time horizons.

---

## 2. Scope & Non-Goals

### In Scope
- Independent repository at `C:\Users\Denis\Documents\codigo\FHIPredict` (no runtime dependency on `FireHistoryInterface`, `FireHistoryInterfaceAPI`, or `ModeloPredictivo`).
- Self-contained inference using `fire_risk_bundle.joblib` (`lgbm-risk-24h-v2`) and a compact static cell baseline table (`cell_baseline.parquet`) exported once from `ModeloPredictivo`.
- Real-time weather ingestion from Open-Meteo Forecast API (`past_days=30` + `forecast_days=15`) to compute daily features and rolling 7d/15d/30d precipitation & dry-streak features.
- Single-screen MapLibre GL map (IGN Argenmap / dark basemap) using FuegoContinuo visual tokens.
- Temporal controls:
  - **Window presets**: `24h (1 día)`, `3 días`, `7 días`, `15 días` (aggregated risk over the selected horizon).
  - **Day-by-day selector**: step/slider from `Día 1` (today/+24h) through `Día 15` to inspect any individual 24h slice.
- Cell inspection on hover/click: `cell_id`, coordinates, predicted probability (`riesgo_pct`), risk category (`Bajo`, `Moderado`, `Alto`, `Extremo`), and key weather drivers (`T° max`, `HR`, `Viento`, `Lluvia`, `Días sin lluvia`).
- Legend, data freshness indicator (Open-Meteo fetch timestamp + model version), and risk category filter/summary counters.

### Out of Scope (Non-Goals)
- Historical fire calendar, 3D satellite schematic, CSV bulk downloads, province/polygon analytics dashboard, authentication, or multi-page navigation.

---

## 3. Time Granularity Contract
- **Native model resolution**: **24 hours (1 day)** (`horizon_hours: 24`, trained on daily weather aggregates).
- **Minimum selectable range**: **1 day (24h)**.
- **Maximum selectable range**: **15 days** (within Open-Meteo's 16-day `forecast_days` limit).
- **Multi-day aggregation rule** (for `3d`, `7d`, `15d` windows):
  - Cumulative horizon probability: $P_{\text{range}} = 1 - \prod_{d=1}^{k}(1 - p_d)$ (probability of at least one positive fire event in the window), plus peak daily category/probability available in cell details.

---

## 4. Visual Identity (FuegoContinuo)
Reuses the exact design tokens from `FireHistoryInterface/DESIGN.md`:
- **Surfaces**: `--space-base: #07100F`, `--space-surface: #0D1816`, `--space-elevated: #111E1B`, `--space-border: #21312D`.
- **Accent**: `--primary: #B8EE68`, `--primary-hover: #CBF58C`.
- **Text**: `--text-primary: #EFF5ED`, `--text-secondary: #93A39C`, `--text-muted: #687771`.
- **Typography**: `Manrope` (headings), `DM Sans` (body), `DM Mono` (coordinates, percentages, dates).
- **Risk Color Scale**:
  - `Bajo`: `#3D786E` / hidden or subtle low opacity at country zoom
  - `Moderado`: `#68D9D0`
  - `Alto`: `#FF9C55`
  - `Extremo`: `#FF6759`
