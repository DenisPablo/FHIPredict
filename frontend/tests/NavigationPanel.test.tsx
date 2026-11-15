import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { NavigationPanel } from '../src/components/NavigationPanel';

describe('NavigationPanel', () => {
  it('permite navegar por provincia y por puntos de mayor riesgo', () => {
    const onFlyTo = vi.fn();
    render(
      <NavigationPanel
        lang="es"
        onFlyTo={onFlyTo}
        features={[
          {
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [-65.5, -33.2] },
            properties: {
              cell_id: 101,
              riesgo_pct: 9.4,
              categoria: 'Extremo',
              tmax_c: 34,
              rh_pct: 20,
              wind_kmh: 30,
              rain_mm: 0,
              dry_days: 12,
            },
          },
        ]}
      />,
    );

    fireEvent.click(screen.getByText('Córdoba'));
    expect(onFlyTo).toHaveBeenCalledWith([-63.8, -32.1], 6.4, '14');

    fireEvent.click(screen.getByText('Top riesgo'));
    fireEvent.click(screen.getByText('#101'));
    expect(onFlyTo).toHaveBeenCalledWith([-65.5, -33.2], 9.2, 'cell-101');
  });
});
