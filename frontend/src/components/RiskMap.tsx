import { useEffect, useRef, useState } from 'react';
import { Mountain, Route } from 'lucide-react';
import * as maplibregl from 'maplibre-gl';
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { ForecastResponse, RiskCategory } from '../api/forecast';
import { COPY, type Lang } from '../i18n';

maplibregl.setWorkerUrl(maplibreWorkerUrl);

export interface FlyTarget {
  center: [number, number];
  zoom: number;
  nonce: number;
}

interface RiskMapProps {
  data?: ForecastResponse;
  lang: Lang;
  flyTarget?: FlyTarget | null;
}

const EMPTY_FC: GeoJSON.FeatureCollection = {
  type: 'FeatureCollection',
  features: [],
};

const SOURCE_ID = 'fire-risk-cells';
const LAYER_ID = 'fire-risk-circles';
const TERRAIN_SOURCE_ID = 'terrain-dem';
const HILLSHADE_LAYER_ID = 'terrain-hillshade';
const ROADS_SOURCE_ID = 'carto-dark-roads';
const ROADS_LAYER_ID = 'carto-dark-roads-layer';

export function RiskMap({ data, lang, flyTarget }: RiskMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const readyRef = useRef(false);
  const latestDataRef = useRef<ForecastResponse | undefined>(data);
  latestDataRef.current = data;
  const langRef = useRef<Lang>(lang);
  langRef.current = lang;
  const [is3D, setIs3D] = useState(false);
  const [showRoads, setShowRoads] = useState(false);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          argenmapOscuro: {
            type: 'raster',
            tiles: [
              'https://wms.ign.gob.ar/geoserver/gwc/service/tms/1.0.0/argenmap_oscuro@EPSG%3A3857@png/{z}/{x}/{y}.png',
            ],
            scheme: 'tms',
            tileSize: 256,
            attribution: '© IGN Argentina · Argenmap Oscuro',
          },
        },
        layers: [
          {
            id: 'argenmap-oscuro-base',
            type: 'raster',
            source: 'argenmapOscuro',
            paint: { 'raster-opacity': 0.92 },
          },
        ],
      },
      center: [-64.0, -37.5],
      zoom: 4.1,
      minZoom: 3,
      maxZoom: 12,
      maxPitch: 75,
    });

    map.addControl(
      new maplibregl.NavigationControl({ showCompass: true, visualizePitch: true }),
      'bottom-left',
    );

    map.on('load', () => {
      map.addSource(SOURCE_ID, {
        type: 'geojson',
        data: EMPTY_FC,
      });

      map.addLayer({
        id: LAYER_ID,
        type: 'circle',
        source: SOURCE_ID,
        paint: {
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['zoom'],
            3.5, 3.8,
            6, 6.5,
            9, 10.5,
            12, 16.0,
          ],
          'circle-color': [
            'match',
            ['get', 'categoria'],
            'Extremo', '#FF6759',
            'Alto', '#FF9C55',
            'Moderado', '#68D9D0',
            '#3D786E',
          ],
          'circle-opacity': [
            'match',
            ['get', 'categoria'],
            'Extremo', 0.92,
            'Alto', 0.82,
            'Moderado', 0.65,
            0.32,
          ],
          'circle-stroke-width': 0.5,
          'circle-stroke-color': '#07100F',
        },
      });

      map.on('mouseenter', LAYER_ID, () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', LAYER_ID, () => {
        map.getCanvas().style.cursor = '';
      });

      map.on('click', LAYER_ID, (e: maplibregl.MapLayerMouseEvent) => {
        const feat = e.features?.[0];
        if (!feat || feat.geometry.type !== 'Point') return;
        const p = feat.properties as Record<string, unknown>;
        const coords = feat.geometry.coordinates.slice() as [number, number];

        const badgeColor =
          p.categoria === 'Extremo'
            ? '#FF6759'
            : p.categoria === 'Alto'
              ? '#FF9C55'
              : p.categoria === 'Moderado'
                ? '#68D9D0'
                : '#3D786E';

        popupRef.current?.remove();
        const t = COPY[langRef.current];
        const cat = p.categoria as RiskCategory;
        const [lon, lat] = coords;
        const latStr = lat.toFixed(4);
        const lonStr = lon.toFixed(4);
        const mapsUrl = `https://www.google.com/maps?q=${latStr},${lonStr}`;
        const meta = latestDataRef.current?.meta;
        const periodLabel = meta
          ? meta.fecha_desde === meta.fecha_hasta
            ? meta.fecha_desde
            : `${meta.fecha_desde} → ${meta.fecha_hasta}`
          : '';
        popupRef.current = new maplibregl.Popup({ closeButton: true, maxWidth: '300px' })
          .setLngLat(coords)
          .setHTML(`
            <div style="font-family:'DM Sans',sans-serif;font-size:12px;line-height:1.45;">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:6px;">
                <span style="font-family:'DM Mono',monospace;color:#93A39C;">${t.popup.cell} #${p.cell_id}</span>
                <span style="background:${badgeColor}22;color:${badgeColor};border:1px solid ${badgeColor}66;padding:2px 8px;border-radius:999px;font-weight:600;font-size:11px;">
                  ${t.categories[cat] ?? cat} · ${p.riesgo_pct}%
                </span>
              </div>
              ${periodLabel ? `<div style="font-family:'DM Mono',monospace;font-size:10px;color:#93A39C;margin-bottom:6px;">${t.popup.period}: ${periodLabel}</div>` : ''}
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#EFF5ED;">
                <div>${t.popup.tmax}: <b>${p.tmax_c}°C</b></div>
                <div>${t.popup.humidity}: <b>${p.rh_pct}%</b></div>
                <div>${t.popup.wind}: <b>${p.wind_kmh} km/h</b></div>
                <div>${t.popup.rain}: <b>${p.rain_mm} mm</b></div>
                <div>${t.popup.rain30d}: <b>${p.rain_30d_mm ?? '—'} mm</b></div>
                <div>${t.popup.vpd}: <b>${p.vpd_kpa ?? '—'} kPa</b></div>
                <div>${t.popup.elev}: <b>${p.elev_m ?? '—'} m</b></div>
                <div>${t.popup.yearsSinceFire}: <b>${p.years_since_fire ?? '—'} ${t.popup.yearsUnit}</b></div>
                <div style="grid-column:span 2;color:#93A39C;">${t.popup.dryDays}: <b style="color:#EFF5ED;">${p.dry_days}</b></div>
              </div>
              <div style="margin-top:8px;padding-top:8px;border-top:1px solid #21312D;display:flex;align-items:center;justify-content:space-between;gap:8px;font-family:'DM Mono',monospace;font-size:11px;">
                <span style="color:#93A39C;">${latStr}, ${lonStr}</span>
                <a href="${mapsUrl}" target="_blank" rel="noopener noreferrer" style="color:#B8EE68;text-decoration:none;font-weight:600;">
                  ${t.popup.openMaps} ↗
                </a>
              </div>
            </div>
          `)
          .addTo(map);
      });

      readyRef.current = true;
      if (latestDataRef.current) {
        (map.getSource(SOURCE_ID) as maplibregl.GeoJSONSource)?.setData(
          latestDataRef.current as unknown as GeoJSON.FeatureCollection,
        );
      }
    });

    mapRef.current = map;
    return () => {
      popupRef.current?.remove();
      map.remove();
      mapRef.current = null;
      readyRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current || !readyRef.current || !data) return;
    const src = mapRef.current.getSource(SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    src?.setData(data as unknown as GeoJSON.FeatureCollection);
  }, [data]);

  useEffect(() => {
    if (!mapRef.current || !flyTarget) return;
    mapRef.current.flyTo({
      center: flyTarget.center,
      zoom: flyTarget.zoom,
      duration: 850,
    });
  }, [flyTarget]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    if (is3D) {
      if (!map.getSource(TERRAIN_SOURCE_ID)) {
        map.addSource(TERRAIN_SOURCE_ID, {
          type: 'raster-dem',
          tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
          encoding: 'terrarium',
          tileSize: 256,
          maxzoom: 15,
        });
      }
      if (!map.getLayer(HILLSHADE_LAYER_ID)) {
        map.addLayer(
          {
            id: HILLSHADE_LAYER_ID,
            type: 'hillshade',
            source: TERRAIN_SOURCE_ID,
            paint: {
              'hillshade-exaggeration': 0.85,
              'hillshade-shadow-color': '#020605',
              'hillshade-highlight-color': '#4BA396',
              'hillshade-accent-color': '#18302B',
              'hillshade-illumination-direction': 315,
            },
          },
          LAYER_ID,
        );
      } else {
        map.setLayoutProperty(HILLSHADE_LAYER_ID, 'visibility', 'visible');
      }
      map.setTerrain({ source: TERRAIN_SOURCE_ID, exaggeration: 2.5 });
      map.easeTo({ pitch: 55, duration: 700 });
    } else {
      if (map.getLayer(HILLSHADE_LAYER_ID)) {
        map.setLayoutProperty(HILLSHADE_LAYER_ID, 'visibility', 'none');
      }
      map.setTerrain(null);
      map.easeTo({ pitch: 0, bearing: 0, duration: 500 });
    }
  }, [is3D]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    if (showRoads) {
      if (!map.getSource(ROADS_SOURCE_ID)) {
        map.addSource(ROADS_SOURCE_ID, {
          type: 'raster',
          tiles: ['https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'],
          tileSize: 256,
          attribution: '© OpenStreetMap · © CARTO',
        });
      }
      if (!map.getLayer(ROADS_LAYER_ID)) {
        const beforeId = map.getLayer(HILLSHADE_LAYER_ID) ? HILLSHADE_LAYER_ID : LAYER_ID;
        map.addLayer(
          {
            id: ROADS_LAYER_ID,
            type: 'raster',
            source: ROADS_SOURCE_ID,
            paint: { 'raster-opacity': 0.85 },
          },
          beforeId,
        );
      } else {
        map.setLayoutProperty(ROADS_LAYER_ID, 'visibility', 'visible');
      }
    } else if (map.getLayer(ROADS_LAYER_ID)) {
      map.setLayoutProperty(ROADS_LAYER_ID, 'visibility', 'none');
    }
  }, [showRoads]);

  const t = COPY[lang];

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />
      <div className="absolute right-4 top-20 z-10 flex items-center gap-2">
        <button
          type="button"
          onClick={() => setShowRoads((prev) => !prev)}
          aria-pressed={showRoads}
          className={
            showRoads
              ? 'inline-flex items-center gap-1.5 rounded-xl border border-lime-brand bg-lime-brand px-3 py-1.5 font-mono text-xs font-semibold text-space-base shadow-xl transition'
              : 'inline-flex items-center gap-1.5 rounded-xl border border-space-border bg-space-surface/95 px-3 py-1.5 font-mono text-xs font-medium text-fcText-primary shadow-xl backdrop-blur-md hover:border-space-borderLight transition'
          }
        >
          <Route className="h-3.5 w-3.5" />
          {t.roadsBtn}
        </button>
        <button
          type="button"
          onClick={() => setIs3D((prev) => !prev)}
          aria-pressed={is3D}
          className={
            is3D
              ? 'inline-flex items-center gap-1.5 rounded-xl border border-lime-brand bg-lime-brand px-3 py-1.5 font-mono text-xs font-semibold text-space-base shadow-xl transition'
              : 'inline-flex items-center gap-1.5 rounded-xl border border-space-border bg-space-surface/95 px-3 py-1.5 font-mono text-xs font-medium text-fcText-primary shadow-xl backdrop-blur-md hover:border-space-borderLight transition'
          }
        >
          <Mountain className="h-3.5 w-3.5" />
          {is3D ? t.disable3d : t.enable3d}
        </button>
      </div>
    </div>
  );
}
