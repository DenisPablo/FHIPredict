# FHIPredict

Standalone fire-risk forecast map for Argentina powered by the pre-trained `lgbm-risk-24h-v2` LightGBM model and live Open-Meteo weather forecasts (up to 15 days).

## Architecture
- `backend/`: FastAPI + LightGBM + DuckDB + Open-Meteo async batch client (`past_days=30`, `forecast_days=15`).
- `frontend/`: Vite + React + TypeScript + Tailwind CSS + MapLibre GL (`FuegoContinuo` visual identity).
- `specs/`: SDD specification, architecture, and task tracking.

## Quick Start
### Backend
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```powershell
cd frontend
npm run dev
```
