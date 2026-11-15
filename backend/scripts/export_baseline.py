"""Export self-contained cell_baseline.parquet from ModeloPredictivo datasets."""

from pathlib import Path
import duckdb


def main() -> None:
    modelo_dir = Path("C:/Users/Denis/Documents/codigo/ModeloPredictivo")
    firms = (modelo_dir / "firms_argentina_harmonized.parquet").as_posix()
    weather = (modelo_dir / "data/**/nasapower_2026.parquet").as_posix()
    out = Path("C:/Users/Denis/Documents/codigo/FHIPredict/backend/models/cell_baseline.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    ref_date = con.sql(f"SELECT MAX(fecha) FROM read_parquet('{firms}')").fetchone()[0]
    print("ref_date:", ref_date)

    con.execute(f"""
    COPY (
      WITH pos AS (
        SELECT cell_id::INT AS cell_id, fecha, n_harmonized::FLOAT AS n_harm, phi_harmonized::FLOAT AS phi_harm
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
      elev AS (
        SELECT lat_clima::FLOAT AS lat_clima, lon_clima::FLOAT AS lon_clima, ANY_VALUE(elevacion_m)::FLOAT AS elevacion_m
        FROM read_parquet('{weather}')
        GROUP BY lat_clima, lon_clima
      ),
      mem AS (
        SELECT
          c.cell_id,
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
      )
      SELECT
        c.cell_id, c.lat_centro, c.lon_centro, c.lat_clima, c.lon_clima,
        COALESCE(e.elevacion_m, 0.0)::FLOAT AS elevacion_m,
        m.focos_ultimos_7d, m.focos_ultimos_30d, m.frp_ultimos_7d, m.anios_desde_ultimo_fuego,
        COALESCE(cl.m01, 0)::FLOAT AS m01, COALESCE(cl.m02, 0)::FLOAT AS m02, COALESCE(cl.m03, 0)::FLOAT AS m03,
        COALESCE(cl.m04, 0)::FLOAT AS m04, COALESCE(cl.m05, 0)::FLOAT AS m05, COALESCE(cl.m06, 0)::FLOAT AS m06,
        COALESCE(cl.m07, 0)::FLOAT AS m07, COALESCE(cl.m08, 0)::FLOAT AS m08, COALESCE(cl.m09, 0)::FLOAT AS m09,
        COALESCE(cl.m10, 0)::FLOAT AS m10, COALESCE(cl.m11, 0)::FLOAT AS m11, COALESCE(cl.m12, 0)::FLOAT AS m12
      FROM cells c
      LEFT JOIN elev e ON c.lat_clima = e.lat_clima AND c.lon_clima = e.lon_clima
      JOIN mem m ON c.cell_id = m.cell_id
      LEFT JOIN clim cl ON c.cell_id = cl.cell_id
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    stats = con.sql(
        f"SELECT COUNT(*), COUNT(DISTINCT (lat_clima, lon_clima)) FROM read_parquet('{out.as_posix()}')"
    ).fetchone()
    print("rows, unique_weather_points:", stats, "size_bytes:", out.stat().st_size)


if __name__ == "__main__":
    main()
