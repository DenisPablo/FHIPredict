"""Download historical daily 10m wind direction (WD10M) from NASA POWER (2018-2026).

Saves a compact ZSTD Parquet table `backend/models/nasapower_wd10m.parquet` with
columns `(lat_clima, lon_clima, fecha, wind_direction_10m_dominant)` matching the
exact 0.5 deg lat x 0.625 deg lon grid used by `nasapower_YYYY.parquet`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

REGIONAL_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/regional"
PARAMETER = "WD10M"
FILL_VALUE = -990.0

ARGENTINA_TILES: tuple[tuple[str, float, float, float, float], ...] = (
    ("norte_oeste", -30.0, -21.5, -74.0, -67.0),
    ("norte_centro", -30.0, -21.5, -67.0, -60.0),
    ("norte_este", -30.0, -21.5, -60.0, -53.0),
    ("centro_norte_oeste", -38.5, -30.0, -74.0, -67.0),
    ("centro_norte_centro", -38.5, -30.0, -67.0, -60.0),
    ("centro_norte_este", -38.5, -30.0, -60.0, -53.0),
    ("centro_sur_oeste", -47.0, -38.5, -74.0, -67.0),
    ("centro_sur_centro", -47.0, -38.5, -67.0, -60.0),
    ("centro_sur_este", -47.0, -38.5, -60.0, -53.0),
    ("sur_oeste", -55.5, -47.0, -74.0, -67.0),
    ("sur_centro", -55.5, -47.0, -67.0, -60.0),
    ("sur_este", -55.5, -47.0, -60.0, -53.0),
)

WD10M_SCHEMA = pa.schema(
    [
        ("lat_clima", pa.float32()),
        ("lon_clima", pa.float32()),
        ("fecha", pa.date32()),
        ("wind_direction_10m_dominant", pa.float32()),
    ]
)


def build_url(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_ymd: str,
    end_ymd: str,
) -> str:
    params = {
        "parameters": PARAMETER,
        "community": "AG",
        "longitude-min": f"{lon_min}",
        "longitude-max": f"{lon_max}",
        "latitude-min": f"{lat_min}",
        "latitude-max": f"{lat_max}",
        "start": start_ymd,
        "end": end_ymd,
        "format": "JSON",
    }
    return f"{REGIONAL_ENDPOINT}?{urllib.parse.urlencode(params)}"


def fetch_json(url: str, max_retries: int = 5, timeout_s: int = 90) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "fhipredict-wd10m-ingest/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError):
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 1.8
                continue
            raise
    raise RuntimeError(f"Exhausted retries for {url}")


def main() -> None:
    out_path = Path("C:/Users/Denis/Documents/codigo/ModeloPredictivo/data/weather/nasapower/wd10m_2018_2026.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        meta = pq.read_metadata(out_path)
        print(f"Existing {out_path.name} found: {meta.num_rows:,} rows ({out_path.stat().st_size / 1e6:.2f} MB)")
        return

    # Use 2026-11-11 as safe end date matching nasapower_2026.parquet
    latest_safe = date(2026, 11, 11)
    years = list(range(2018, 2027))
    tasks: list[tuple[int, str, str]] = []
    for year in years:
        start_ymd = f"{year}0101"
        end_dt = min(date(year, 12, 31), latest_safe)
        end_ymd = end_dt.strftime("%Y%m%d")
        for tile_name, lat_min, lat_max, lon_min, lon_max in ARGENTINA_TILES:
            url = build_url(lat_min, lat_max, lon_min, lon_max, start_ymd, end_ymd)
            tasks.append((year, tile_name, url))

    t0 = time.perf_counter()
    print(f"Fetching {len(tasks)} WD10M tile-year payloads from NASA POWER...")

    records: dict[tuple[float, float, str], float] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_map = {pool.submit(fetch_json, url): (year, tile_name) for year, tile_name, url in tasks}
        for fut in as_completed(future_map):
            year, tile_name = future_map[fut]
            payload = fut.result()
            for feat in payload.get("features", []):
                coords = feat.get("geometry", {}).get("coordinates", [])
                if len(coords) < 2:
                    continue
                lon_val = float(coords[0])
                lat_val = float(coords[1])
                series = feat.get("properties", {}).get("parameter", {}).get(PARAMETER, {})
                for ymd, raw_val in series.items():
                    if raw_val is not None and float(raw_val) > FILL_VALUE:
                        records[(lat_val, lon_val, ymd)] = float(raw_val)
            done += 1
            if done % 12 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"  [{done:3d}/{len(tasks)}] last={year}:{tile_name} | records={len(records):,} | {elapsed:.1f}s")
                sys.stdout.flush()

    lat_col: list[float] = []
    lon_col: list[float] = []
    date_col: list[date] = []
    wd_col: list[float] = []

    for (lat_val, lon_val, ymd), wd_val in records.items():
        lat_col.append(lat_val)
        lon_col.append(lon_val)
        date_col.append(datetime.strptime(ymd, "%Y%m%d").date())
        wd_col.append(wd_val)

    table = pa.table(
        {
            "lat_clima": lat_col,
            "lon_clima": lon_col,
            "fecha": date_col,
            "wind_direction_10m_dominant": wd_col,
        },
        schema=WD10M_SCHEMA,
    )
    pq.write_table(table, out_path, compression="zstd")
    print(
        f"Saved {out_path} ({len(records):,} rows, {out_path.stat().st_size / 1e6:.2f} MB) "
        f"in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
