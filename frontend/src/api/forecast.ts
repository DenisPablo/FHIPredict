export type RiskCategory = 'Bajo' | 'Moderado' | 'Alto' | 'Extremo';

export interface CellProperties {
  cell_id: number;
  riesgo_pct: number;
  categoria: RiskCategory;
  tmax_c: number;
  rh_pct: number;
  wind_kmh: number;
  rain_mm: number;
  rain_30d_mm?: number;
  dry_days: number;
  vpd_kpa?: number;
  elev_m?: number;
  years_since_fire?: number;
}

export interface ForecastFeature {
  type: 'Feature';
  geometry: {
    type: 'Point';
    coordinates: [number, number];
  };
  properties: CellProperties;
}

export interface ForecastResponse {
  type: 'FeatureCollection';
  meta: {
    model_version: string;
    generated_at: string;
    mode: 'window' | 'day';
    fecha_desde: string;
    fecha_hasta: string;
    dates_available: string[];
    counts: Record<RiskCategory, number>;
    returned: number;
  };
  features: ForecastFeature[];
}

export interface ForecastQuery {
  mode: 'window' | 'day';
  days: number;
  dayOffset: number;
  minRisk: RiskCategory;
  refresh?: boolean;
}

export async function fetchForecast(q: ForecastQuery, signal?: AbortSignal): Promise<ForecastResponse> {
  const params = new URLSearchParams({
    mode: q.mode,
    days: String(q.days),
    day_offset: String(q.dayOffset),
    min_risk: q.minRisk,
  });
  if (q.refresh) params.set('refresh', 'true');

  const res = await fetch(`/api/v1/forecast?${params.toString()}`, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `API error (${res.status})`);
  }
  return res.json() as Promise<ForecastResponse>;
}
