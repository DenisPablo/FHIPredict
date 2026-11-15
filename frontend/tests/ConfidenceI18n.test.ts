import { describe, it, expect } from 'vitest';
import { COPY } from '../src/i18n';
import type { CellProperties } from '../src/api/forecast';

describe('Confidence copy & types', () => {
  it('incluye etiquetas y niveles de confianza en ES y EN', () => {
    for (const lang of ['es', 'en'] as const) {
      const popup = COPY[lang].popup;
      expect(popup.confidence).toBeTruthy();
      expect(popup.confidenceHint).toBeTruthy();
      expect(popup.confidenceLevels.Alta).toBeTruthy();
      expect(popup.confidenceLevels.Media).toBeTruthy();
      expect(popup.confidenceLevels.Baja).toBeTruthy();
    }
  });

  it('acepta propiedades de confianza opcionales en CellProperties', () => {
    const withConf: CellProperties = {
      cell_id: 1,
      riesgo_pct: 12.5,
      categoria: 'Alto',
      confianza_pct: 82,
      confianza_nivel: 'Alta',
      tmax_c: 33,
      rh_pct: 22,
      wind_kmh: 25,
      rain_mm: 0,
      dry_days: 10,
    };
    expect(withConf.confianza_pct).toBe(82);
  });
});
