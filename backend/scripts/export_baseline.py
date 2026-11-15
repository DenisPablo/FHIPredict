"""Export self-contained cell_baseline.parquet from ModeloPredictivo and ML_NasaSpaceApp datasets.

Preserves all 22 original baseline columns required by `lgbm-risk-24h-v2` and enriches
each of the 64,653 cells with:
- `grid_row`, `grid_col`, `frac_dentro` from `ML_NasaSpaceApp/serving_api/dim_celdas.parquet`
- `is_industrial`, `industrial_focos_share` from `ML_NasaSpaceApp/serving_api/puntos_industriales.parquet`
- 4 normalized biophysical land-cover / fuel-type fractions (`fuel_bosque`, `fuel_pastizal`,
  `fuel_agro`, `fuel_no_combustible` summing to 1.0) derived strictly from static
  topography (`elevacion_m`), continental land mask (`frac_dentro`), ecoregion (`region`),
  and multi-year climatology (precipitation, temperature, relative humidity, VPD)
  with zero target leakage.
"""

from __future__ import annotations

from pathlib import Path
import duckdb


def main() -> None:
    modelo_dir = Path("C:/Users/Denis/Documents/codigo/ModeloPredictivo")
    serving_dir = Path("C:/Users/Denis/Documents/codigo/ML_NasaSpaceApp/serving_api")
    firms = (modelo_dir / "firms_argentina_harmonized.parquet").as_posix()
    weather_all = (modelo_dir / "data/weather/nasapower/nasapower_*.parquet").as_posix()
    weather_2026 = (modelo_dir / "data/**/nasapower_2026.parquet").as_posix()
    dim_celdas = (serving_dir / "dim_celdas.parquet").as_posix()
    puntos_ind = (serving_dir / "puntos_industriales.parquet").as_posix()
    focos_train = (serving_dir / "focos/anio=20*/part-0.parquet").as_posix()

    out = Path("C:/Users/Denis/Documents/codigo/FHIPredict/backend/models/cell_baseline.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    ref_date = con.execute(f"SELECT MAX(fecha) FROM read_parquet('{firms}')").fetchall()[0][0]
    print("ref_date:", ref_date)

    con.execute(f"""
    COPY (
      WITH pos AS (
        SELECT
          cell_id::INT AS cell_id,
          fecha::DATE AS fecha,
          n_harmonized::FLOAT AS n_harm,
          phi_harmonized::FLOAT AS phi_harm
        FROM read_parquet('{firms}')
        WHERE n_harmonized >= 1 AND phi_harmonized >= 10.0
      ),
      cells AS (
        SELECT
          cell_id::INT AS cell_id,
          ANY_VALUE(lat_centro)::FLOAT AS lat_centro,
          ANY_VALUE(lon_centro)::FLOAT AS lon_centro,
          (ROUND(ANY_VALUE(lat_centro) * 2.0) / 2.0)::FLOAT AS lat_clima,
          (ROUND(ANY_VALUE(lon_centro) / 0.625) * 0.625)::FLOAT AS lon_clima
        FROM read_parquet('{firms}')
        WHERE lat_centro IS NOT NULL AND lon_centro IS NOT NULL
        GROUP BY cell_id
      ),
      dim AS (
        SELECT
          cell_id::INT AS cell_id,
          grid_row::SMALLINT AS grid_row,
          grid_col::SMALLINT AS grid_col,
          COALESCE(frac_dentro, 1.0)::FLOAT AS frac_dentro,
          COALESCE(region, 'Pampeana') AS region
        FROM read_parquet('{dim_celdas}')
      ),
      elev AS (
        SELECT
          lat_clima::FLOAT AS lat_clima,
          lon_clima::FLOAT AS lon_clima,
          ANY_VALUE(elevacion_m)::FLOAT AS elevacion_m
        FROM read_parquet('{weather_2026}')
        GROUP BY lat_clima, lon_clima
      ),
      climate_base AS (
        SELECT
          lat_clima::FLOAT AS lat_clima,
          lon_clima::FLOAT AS lon_clima,
          (SUM(COALESCE(precipitation_sum, 0.0)) / 6.0)::FLOAT AS precip_anual_mm,
          AVG(temperature_2m_max)::FLOAT AS tmax_mean,
          AVG(relative_humidity_2m)::FLOAT AS rh_mean,
          AVG(shortwave_radiation_sum)::FLOAT AS rad_mean
        FROM read_parquet('{weather_all}')
        WHERE fecha BETWEEN DATE '2018-01-01' AND DATE '2023-12-31'
        GROUP BY lat_clima, lon_clima
      ),
      ind_match AS (
        SELECT
          c.cell_id,
          COUNT(i.punto_id)::INT AS n_ind_points,
          COALESCE(SUM(i.n_focos), 0)::FLOAT AS ind_focos_total
        FROM cells c
        LEFT JOIN read_parquet('{puntos_ind}') i
          ON ABS(c.lat_centro - i.lat) <= 0.030
         AND ABS(c.lon_centro - i.lon) <= 0.038
        GROUP BY c.cell_id
      ),
      focos_raw_cnt AS (
        SELECT
          cell_id::INT AS cell_id,
          COUNT(*)::FLOAT AS total_raw_focos
        FROM read_parquet('{focos_train}')
        WHERE fecha BETWEEN DATE '2018-01-01' AND DATE '2023-12-31'
        GROUP BY cell_id
      ),
      mem AS (
        SELECT
          c.cell_id,
          COALESCE(SUM(CASE WHEN p.fecha = DATE '{ref_date}' THEN p.n_harm ELSE 0 END), 0)::FLOAT AS focos_ultimos_1d,
          COALESCE(SUM(CASE WHEN p.fecha BETWEEN DATE '{ref_date}' - INTERVAL 1 DAY AND DATE '{ref_date}' THEN p.n_harm ELSE 0 END), 0)::FLOAT AS focos_ultimos_2d,
          COALESCE(SUM(CASE WHEN p.fecha = DATE '{ref_date}' THEN p.phi_harm ELSE 0 END), 0)::FLOAT AS frp_ultimos_1d,
          COALESCE(SUM(CASE WHEN p.fecha BETWEEN DATE '{ref_date}' - INTERVAL 6 DAY AND DATE '{ref_date}' THEN p.n_harm ELSE 0 END), 0)::FLOAT AS focos_ultimos_7d,
          COALESCE(SUM(CASE WHEN p.fecha BETWEEN DATE '{ref_date}' - INTERVAL 29 DAY AND DATE '{ref_date}' THEN p.n_harm ELSE 0 END), 0)::FLOAT AS focos_ultimos_30d,
          COALESCE(SUM(CASE WHEN p.fecha BETWEEN DATE '{ref_date}' - INTERVAL 6 DAY AND DATE '{ref_date}' THEN p.phi_harm ELSE 0 END), 0)::FLOAT AS frp_ultimos_7d,
          COALESCE(DATEDIFF('day', MAX(p.fecha), DATE '{ref_date}' + INTERVAL 1 DAY) / 365.25, 10.0)::FLOAT AS anios_desde_ultimo_fuego
        FROM cells c
        LEFT JOIN pos p ON c.cell_id = p.cell_id
        GROUP BY c.cell_id
      ),
      clim AS (
        SELECT
          cell_id,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 1 THEN n_harm END), 0)::FLOAT AS m01,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 2 THEN n_harm END), 0)::FLOAT AS m02,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 3 THEN n_harm END), 0)::FLOAT AS m03,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 4 THEN n_harm END), 0)::FLOAT AS m04,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 5 THEN n_harm END), 0)::FLOAT AS m05,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 6 THEN n_harm END), 0)::FLOAT AS m06,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 7 THEN n_harm END), 0)::FLOAT AS m07,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 8 THEN n_harm END), 0)::FLOAT AS m08,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 9 THEN n_harm END), 0)::FLOAT AS m09,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 10 THEN n_harm END), 0)::FLOAT AS m10,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 11 THEN n_harm END), 0)::FLOAT AS m11,
          COALESCE(MEDIAN(CASE WHEN EXTRACT(MONTH FROM fecha) = 12 THEN n_harm END), 0)::FLOAT AS m12
        FROM pos
        GROUP BY cell_id
      ),
      fuel_raw AS (
        SELECT
          c.cell_id,
          COALESCE(d.grid_row, (c.cell_id // 460)::SMALLINT) AS grid_row,
          COALESCE(d.grid_col, (c.cell_id % 460)::SMALLINT) AS grid_col,
          COALESCE(d.frac_dentro, 1.0)::FLOAT AS frac_dentro,
          CASE WHEN im.n_ind_points > 0 THEN 1.0::FLOAT ELSE 0.0::FLOAT END AS is_industrial,
          LEAST(
            1.0,
            CASE
              WHEN im.n_ind_points > 0
              THEN im.ind_focos_total / GREATEST(COALESCE(frc.total_raw_focos, 1.0), im.ind_focos_total, 1.0)
              ELSE 0.0
            END
          )::FLOAT AS industrial_focos_share,
          -- Pure biophysical / topographic / climatic land-cover fractions (zero target leakage)
          LEAST(
            0.95,
            GREATEST(
              0.01,
              (1.0 - COALESCE(d.frac_dentro, 1.0)) * 0.85
              + LEAST(0.75, GREATEST(0.0, (COALESCE(e.elevacion_m, 0.0) - 2400.0) / 2000.0))
              + LEAST(0.55, GREATEST(0.0, (220.0 - COALESCE(cb.precip_anual_mm, 650.0)) / 220.0))
            )
          )::FLOAT AS raw_no_comb,
          GREATEST(
            0.03,
            CASE
              WHEN COALESCE(d.region, 'Pampeana') IN ('NEA', 'NOA')
              THEN 0.15 + 0.55 * LEAST(1.0, GREATEST(0.0, (COALESCE(cb.precip_anual_mm, 650.0) - 450.0) / 900.0))
                        * LEAST(1.0, GREATEST(0.1, (2200.0 - COALESCE(e.elevacion_m, 0.0)) / 2000.0))
              WHEN COALESCE(d.region, 'Pampeana') = 'Patagonia' AND c.lon_centro < -70.2
              THEN 0.20 + 0.55 * LEAST(1.0, GREATEST(0.0, (COALESCE(cb.precip_anual_mm, 650.0) - 350.0) / 800.0))
              ELSE 0.08 + 0.22 * LEAST(1.0, GREATEST(0.0, (COALESCE(cb.rh_mean, 55.0) - 45.0) / 40.0))
            END
          )::FLOAT AS raw_bosque,
          GREATEST(
            0.05,
            CASE
              WHEN COALESCE(d.region, 'Pampeana') IN ('NEA', 'Pampeana')
               AND c.lon_centro BETWEEN -61.2 AND -57.0
               AND c.lat_centro BETWEEN -34.6 AND -27.0
              THEN 0.55 + 0.20 * LEAST(1.0, GREATEST(0.0, (120.0 - COALESCE(e.elevacion_m, 0.0)) / 120.0))
              WHEN COALESCE(d.region, 'Pampeana') IN ('Cuyo', 'Patagonia')
              THEN 0.50 + 0.20 * LEAST(1.0, GREATEST(0.0, (700.0 - COALESCE(cb.precip_anual_mm, 650.0)) / 600.0))
              ELSE 0.28 + 0.15 * LEAST(1.0, GREATEST(0.0, (COALESCE(e.elevacion_m, 0.0) - 150.0) / 1200.0))
            END
          )::FLOAT AS raw_pastizal,
          GREATEST(
            0.03,
            CASE
              WHEN COALESCE(d.region, 'Pampeana') = 'Pampeana' AND COALESCE(e.elevacion_m, 0.0) < 550.0
              THEN 0.52 * LEAST(1.0, GREATEST(0.1, (COALESCE(cb.precip_anual_mm, 650.0) - 450.0) / 600.0))
                        * LEAST(1.0, GREATEST(0.2, (COALESCE(e.elevacion_m, 0.0) - 15.0) / 80.0))
              WHEN COALESCE(d.region, 'Pampeana') IN ('NOA', 'NEA') AND COALESCE(e.elevacion_m, 0.0) BETWEEN 60.0 AND 550.0
              THEN 0.34 * LEAST(1.0, GREATEST(0.1, (COALESCE(cb.precip_anual_mm, 650.0) - 500.0) / 600.0))
              ELSE 0.10
            END
          )::FLOAT AS raw_agro
        FROM cells c
        LEFT JOIN dim d ON c.cell_id = d.cell_id
        LEFT JOIN elev e ON c.lat_clima = e.lat_clima AND c.lon_clima = e.lon_clima
        LEFT JOIN climate_base cb ON c.lat_clima = cb.lat_clima AND c.lon_clima = cb.lon_clima
        LEFT JOIN ind_match im ON c.cell_id = im.cell_id
        LEFT JOIN focos_raw_cnt frc ON c.cell_id = frc.cell_id
      ),
      fuel_norm AS (
        SELECT
          cell_id,
          grid_row,
          grid_col,
          frac_dentro,
          is_industrial,
          industrial_focos_share,
          raw_no_comb AS fuel_no_combustible,
          ((1.0 - raw_no_comb) * raw_bosque / GREATEST(raw_bosque + raw_pastizal + raw_agro, 1e-6))::FLOAT AS fuel_bosque,
          ((1.0 - raw_no_comb) * raw_pastizal / GREATEST(raw_bosque + raw_pastizal + raw_agro, 1e-6))::FLOAT AS fuel_pastizal,
          ((1.0 - raw_no_comb) * raw_agro / GREATEST(raw_bosque + raw_pastizal + raw_agro, 1e-6))::FLOAT AS fuel_agro
        FROM fuel_raw
      )
      SELECT
        c.cell_id, c.lat_centro, c.lon_centro, c.lat_clima, c.lon_clima,
        COALESCE(e.elevacion_m, 0.0)::FLOAT AS elevacion_m,
        m.focos_ultimos_7d, m.focos_ultimos_30d, m.frp_ultimos_7d, m.anios_desde_ultimo_fuego,
        COALESCE(cl.m01, 0)::FLOAT AS m01, COALESCE(cl.m02, 0)::FLOAT AS m02, COALESCE(cl.m03, 0)::FLOAT AS m03,
        COALESCE(cl.m04, 0)::FLOAT AS m04, COALESCE(cl.m05, 0)::FLOAT AS m05, COALESCE(cl.m06, 0)::FLOAT AS m06,
        COALESCE(cl.m07, 0)::FLOAT AS m07, COALESCE(cl.m08, 0)::FLOAT AS m08, COALESCE(cl.m09, 0)::FLOAT AS m09,
        COALESCE(cl.m10, 0)::FLOAT AS m10, COALESCE(cl.m11, 0)::FLOAT AS m11, COALESCE(cl.m12, 0)::FLOAT AS m12,
        fn.grid_row,
        fn.grid_col,
        fn.frac_dentro,
        fn.is_industrial,
        fn.industrial_focos_share,
        fn.fuel_bosque,
        fn.fuel_pastizal,
        fn.fuel_agro,
        fn.fuel_no_combustible,
        m.focos_ultimos_1d,
        m.focos_ultimos_2d,
        m.frp_ultimos_1d
      FROM cells c
      LEFT JOIN elev e ON c.lat_clima = e.lat_clima AND c.lon_clima = e.lon_clima
      JOIN mem m ON c.cell_id = m.cell_id
      LEFT JOIN clim cl ON c.cell_id = cl.cell_id
      JOIN fuel_norm fn ON c.cell_id = fn.cell_id
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    stats = con.execute(
        f"""
        SELECT
          COUNT(*),
          COUNT(DISTINCT (lat_clima, lon_clima)),
          SUM(CASE WHEN is_industrial > 0.5 THEN 1 ELSE 0 END),
          ROUND(AVG(fuel_bosque), 4),
          ROUND(AVG(fuel_pastizal), 4),
          ROUND(AVG(fuel_agro), 4),
          ROUND(AVG(fuel_no_combustible), 4)
        FROM read_parquet('{out.as_posix()}')
        """
    ).fetchall()[0]
    print(
        "rows, weather_pts, ind_cells, mean(bosque, pastizal, agro, no_comb):",
        stats,
        "size_bytes:",
        out.stat().st_size,
    )


if __name__ == "__main__":
    main()
