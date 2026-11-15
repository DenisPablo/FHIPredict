import { COPY, type Lang } from '../i18n';
import type { RiskCategory } from '../api/forecast';

const RISK_SCALE: { key: RiskCategory; range: string; color: string }[] = [
  { key: 'Extremo', range: '≥ 5.79%', color: '#FF6759' },
  { key: 'Alto', range: '1.29 – 5.79%', color: '#FF9C55' },
  { key: 'Moderado', range: '0.35 – 1.29%', color: '#68D9D0' },
  { key: 'Bajo', range: '< 0.35%', color: '#3D786E' },
];

export function RiskLegend({ lang }: { lang: Lang }) {
  const t = COPY[lang];
  return (
    <aside
      aria-label={t.legendTitle}
      className="pointer-events-auto max-w-[260px] rounded-2xl border border-space-border bg-space-surface/95 px-3.5 py-2.5 shadow-xl backdrop-blur-md"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <p className="font-mono text-[10px] uppercase tracking-wider text-fcText-secondary">
          {t.legendTitle}
        </p>
        <span className="rounded-full border border-[#FF9C55]/40 bg-[#FF9C55]/10 px-2 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wide text-[#FF9C55]">
          {t.demoBadge}
        </span>
      </div>
      <ul className="space-y-1.5">
        {RISK_SCALE.map((item) => (
          <li key={item.key} className="flex items-center justify-between gap-4 text-xs">
            <span className="inline-flex items-center gap-2 text-fcText-primary">
              <span
                className="h-2.5 w-2.5 rounded-full ring-1 ring-space-base"
                style={{ backgroundColor: item.color }}
              />
              {t.categories[item.key]}
            </span>
            <span className="font-mono text-[11px] text-fcText-secondary">{item.range}</span>
          </li>
        ))}
      </ul>
      <p className="mt-2.5 border-t border-space-border pt-2 text-[10px] leading-relaxed text-fcText-muted">
        {t.disclaimer}
      </p>
    </aside>
  );
}
