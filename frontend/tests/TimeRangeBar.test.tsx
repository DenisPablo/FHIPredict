import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TimeRangeBar } from '../src/components/TimeRangeBar';

describe('TimeRangeBar', () => {
  it('permite elegir presets de 24h a 15 días y deslizar día puntual', () => {
    const onSelectWindow = vi.fn();
    const onSelectDay = vi.fn();
    render(
      <TimeRangeBar
        lang="es"
        mode="window"
        days={1}
        dayOffset={0}
        minRisk="Moderado"
        datesAvailable={['2026-11-15', '2026-11-16', '2026-11-17']}
        fechaDesde="2026-11-15"
        fechaHasta="2026-11-15"
        isFetching={false}
        onSelectWindow={onSelectWindow}
        onSelectDay={onSelectDay}
        onChangeMinRisk={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByText('15 días'));
    expect(onSelectWindow).toHaveBeenCalledWith(15);

    fireEvent.change(screen.getByLabelText('Día puntual'), { target: { value: '2' } });
    expect(onSelectDay).toHaveBeenCalledWith(2);
  });

  it('soporta inglés en etiquetas y presets', () => {
    render(
      <TimeRangeBar
        lang="en"
        mode="window"
        days={7}
        dayOffset={0}
        minRisk="Moderado"
        datesAvailable={['2026-11-15']}
        isFetching={false}
        onSelectWindow={vi.fn()}
        onSelectDay={vi.fn()}
        onChangeMinRisk={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText('15 days')).toBeTruthy();
    expect(screen.getByText('Single day')).toBeTruthy();
  });
});
