import { Calendar, RefreshCw } from 'lucide-react';
import type { RiskCategory } from '../api/forecast';
import { COPY, type Lang } from '../i18n';

const WINDOW_DAYS = [1, 3, 7, 15] as const;
const RISK_VALUES: RiskCategory[] = ['Moderado', 'Alto', 'Extremo', 'Bajo'];

interface TimeRangeBarProps {
  lang: Lang;
  mode: 'window' | 'day';
  days: number;
  dayOffset: number;
  minRisk: RiskCategory;
  datesAvailable: string[];
  fechaDesde?: string;
  fechaHasta?: string;
  isFetching: boolean;
  onSelectWindow: (days: number) => void;
  onSelectDay: (offset: number) => void;
  onChangeMinRisk: (risk: RiskCategory) => void;
  onRefresh: () => void;
}

export function TimeRangeBar({
  lang,
  mode,
  days,
  dayOffset,
  minRisk,
  datesAvailable,
  fechaDesde,
  fechaHasta,
  isFetching,
  onSelectWindow,
  onSelectDay,
  onChangeMinRisk,
  onRefresh,
}: TimeRangeBarProps) {
  const t = COPY[lang];
  const maxOffset = Math.max(0, (datesAvailable.length || 15) - 1);
  const activeDayDate = datesAvailable[dayOffset] ?? `${t.dayWord} ${dayOffset + 1}`;

  return (
    <div className="pointer-events-auto flex flex-col gap-3 rounded-2xl border border-space-border bg-space-surface/95 p-3.5 shadow-2xl backdrop-blur-md">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 inline-flex items-center gap-1.5 font-mono text-xs uppercase tracking-wider text-fcText-secondary">
            <Calendar className="h-3.5 w-3.5 text-lime-brand" />
            {t.horizon}
          </span>
          {WINDOW_DAYS.map((d) => {
            const active = mode === 'window' && days === d;
            return (
              <button
                key={d}
                type="button"
                onClick={() => onSelectWindow(d)}
                className={
                  active
                    ? 'rounded-lg bg-lime-brand px-3 py-1.5 text-xs font-semibold text-space-base transition'
                    : 'rounded-lg border border-space-border bg-space-elevated px-3 py-1.5 text-xs font-medium text-fcText-primary hover:border-space-borderLight transition'
                }
              >
                {t.presets[d]}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <select
            aria-label={t.showPrefix}
            value={minRisk}
            onChange={(e) => onChangeMinRisk(e.target.value as RiskCategory)}
            className="rounded-lg border border-space-border bg-space-elevated px-2.5 py-1.5 text-xs text-fcText-primary outline-none focus:border-lime-brand"
          >
            {RISK_VALUES.map((val) => (
              <option key={val} value={val}>
                {t.showPrefix}: {t.minRisk[val]}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={onRefresh}
            disabled={isFetching}
            title={t.refreshTitle}
            className="inline-flex items-center gap-1.5 rounded-lg border border-space-border bg-space-elevated px-2.5 py-1.5 text-xs text-fcText-secondary hover:text-fcText-primary disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isFetching ? 'animate-spin text-lime-brand' : ''}`} />
            Open-Meteo
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 border-t border-space-border/80 pt-2.5">
        <span className="font-mono text-[11px] uppercase tracking-wider text-fcText-muted">
          {t.singleDay}
        </span>
        <input
          type="range"
          aria-label={t.singleDay}
          min={0}
          max={maxOffset}
          value={Math.min(dayOffset, maxOffset)}
          onChange={(e) => onSelectDay(Number(e.target.value))}
          className="h-1.5 flex-1 min-w-[160px] cursor-pointer accent-[#B8EE68]"
        />
        <span className="rounded-md bg-space-elevated px-2.5 py-1 font-mono text-xs text-fcText-primary">
          {mode === 'day'
            ? `${t.dayWord} ${dayOffset + 1} · ${activeDayDate}`
            : `${fechaDesde ?? '—'} → ${fechaHasta ?? '—'}`}
        </span>
      </div>
    </div>
  );
}
