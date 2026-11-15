import { useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, Flame, MapPin } from 'lucide-react';
import type { ForecastFeature } from '../api/forecast';
import { PROVINCE_TARGETS } from '../data/provinces';
import { COPY, type Lang } from '../i18n';

interface NavigationPanelProps {
  lang: Lang;
  features: ForecastFeature[];
  onFlyTo: (center: [number, number], zoom: number, id: string) => void;
  activeId?: string;
}

export function NavigationPanel({ lang, features, onFlyTo, activeId }: NavigationPanelProps) {
  const [open, setOpen] = useState(true);
  const [tab, setTab] = useState<'provinces' | 'hotspots'>('provinces');
  const t = COPY[lang];

  const topHotspots = useMemo(() => {
    if (tab !== 'hotspots' || features.length === 0) return [];
    const top: ForecastFeature[] = [];
    for (const feat of features) {
      const score = feat.properties.riesgo_pct;
      if (top.length === 30 && score <= top[29].properties.riesgo_pct) continue;
      let idx = top.length;
      while (idx > 0 && top[idx - 1].properties.riesgo_pct < score) {
        idx--;
      }
      top.splice(idx, 0, feat);
      if (top.length > 30) top.pop();
    }
    return top;
  }, [features, tab]);

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={t.nav.open}
        className="pointer-events-auto inline-flex items-center gap-1.5 rounded-xl border border-space-border bg-space-surface/95 px-3 py-2 text-xs font-medium text-fcText-primary shadow-xl backdrop-blur-md hover:border-space-borderLight"
      >
        <MapPin className="h-3.5 w-3.5 text-lime-brand" />
        {t.nav.title}
        <ChevronRight className="h-3.5 w-3.5 text-fcText-secondary" />
      </button>
    );
  }

  return (
    <aside
      aria-label={t.nav.title}
      className="pointer-events-auto flex max-h-[calc(100vh-12rem)] w-64 flex-col rounded-2xl border border-space-border bg-space-surface/95 shadow-2xl backdrop-blur-md"
    >
      <div className="flex items-center justify-between border-b border-space-border px-3 py-2.5">
        <span className="inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-fcText-secondary">
          <MapPin className="h-3.5 w-3.5 text-lime-brand" />
          {t.nav.title}
        </span>
        <button
          type="button"
          onClick={() => setOpen(false)}
          aria-label={t.nav.close}
          className="rounded-md p-1 text-fcText-secondary hover:bg-space-elevated hover:text-fcText-primary"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-1 border-b border-space-border p-1.5">
        <button
          type="button"
          onClick={() => setTab('provinces')}
          className={
            tab === 'provinces'
              ? 'rounded-lg bg-lime-brand px-2 py-1 text-xs font-semibold text-space-base'
              : 'rounded-lg px-2 py-1 text-xs text-fcText-secondary hover:text-fcText-primary'
          }
        >
          {t.nav.provinces}
        </button>
        <button
          type="button"
          onClick={() => setTab('hotspots')}
          className={
            tab === 'hotspots'
              ? 'rounded-lg bg-lime-brand px-2 py-1 text-xs font-semibold text-space-base'
              : 'rounded-lg px-2 py-1 text-xs text-fcText-secondary hover:text-fcText-primary'
          }
        >
          {t.nav.hotspots}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-1.5">
        {tab === 'provinces' ? (
          <ul className="space-y-0.5">
            {PROVINCE_TARGETS.map((prov) => {
              const selected = activeId === prov.id;
              return (
                <li key={prov.id}>
                  <button
                    type="button"
                    onClick={() => onFlyTo(prov.center, prov.zoom, prov.id)}
                    className={
                      selected
                        ? 'flex w-full items-center justify-between rounded-lg bg-space-elevated px-2.5 py-1.5 text-left text-xs font-semibold text-lime-brand'
                        : 'flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left text-xs text-fcText-primary hover:bg-space-elevated/70'
                    }
                  >
                    <span className="truncate">{prov.id === 'arg' ? t.nav.country : prov.name}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        ) : topHotspots.length === 0 ? (
          <p className="px-2 py-4 text-center text-xs text-fcText-muted">{t.nav.emptyHotspots}</p>
        ) : (
          <ul className="space-y-1">
            {topHotspots.map((feat) => {
              const p = feat.properties;
              const id = `cell-${p.cell_id}`;
              const color =
                p.categoria === 'Extremo'
                  ? 'text-[#FF6759]'
                  : p.categoria === 'Alto'
                    ? 'text-[#FF9C55]'
                    : 'text-[#68D9D0]';
              return (
                <li key={id}>
                  <button
                    type="button"
                    onClick={() => onFlyTo(feat.geometry.coordinates, 9.2, id)}
                    className={`flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-xs hover:bg-space-elevated/80 ${
                      activeId === id ? 'bg-space-elevated ring-1 ring-lime-brand/40' : ''
                    }`}
                  >
                    <span className="inline-flex items-center gap-1.5 truncate font-mono text-[11px] text-fcText-primary">
                      <Flame className={`h-3 w-3 shrink-0 ${color}`} />
                      #{p.cell_id}
                    </span>
                    <span className={`font-mono text-[11px] font-semibold ${color}`}>
                      {p.riesgo_pct}%
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}
