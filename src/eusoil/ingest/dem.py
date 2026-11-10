"""Elevation and slope at the survey points from the Copernicus DEM GLO-90.

The DEM is read directly from the public copy on AWS (bucket
``copernicus-dem-90m``, Cloud-Optimised GeoTIFFs in 1 x 1 degree tiles),
fetching only the 3 x 3 pixel window around each point by HTTP range
requests. Nothing is downloaded in bulk and no account is needed.

* Elevation is the value of the pixel containing the point.
* Slope uses Horn's (1981) third-order finite difference on the 3 x 3
  window, with pixel spacing converted from degrees to metres at the point's
  latitude.

Tile edges. The GLO-90 tiles are pixel-is-point grids whose footprint is
shifted half a pixel from the whole degree (the N39 tile spans about
39.0004-40.0004 N), so a point within a pixel of a tile edge can fall in
the neighbouring tile, and its 3 x 3 window can straddle two to four tiles.
Every point is first read from its nominal tile; points left with a missing
value are then read again pixel by pixel, each of the nine pixels from the
tile that actually contains it. Values still missing after that are genuine
no-data (sea) or unavailable tiles.

Attribution required when publishing results:
"Copernicus DEM GLO-90 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and
Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA."
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

LOG = logging.getLogger(__name__)

BUCKET_URL = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"
EARTH_RADIUS_M = 6_371_008.8
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def tile_name(lat: float, lon: float) -> str:
    """Name of the 1-degree GLO-90 tile containing (lat, lon).

    Tiles are named after their south-west corner, for example
    ``Copernicus_DSM_COG_30_N47_00_E016_00_DEM`` covers 47-48 N, 16-17 E.
    """
    lat0 = math.floor(lat)
    lon0 = math.floor(lon)
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"Copernicus_DSM_COG_30_{ns}{abs(lat0):02d}_00_{ew}{abs(lon0):03d}_00_DEM"


def tile_url(name: str) -> str:
    return f"/vsicurl/{BUCKET_URL}/{name}/{name}.tif"


def horn_slope_deg(window: np.ndarray, dx_m: float, dy_m: float) -> float:
    """Slope in degrees from a 3 x 3 elevation window (rows north to south).

    Horn (1981): dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 dx) and
    dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 dy). Returns NaN if any cell
    is missing.
    """
    if window.shape != (3, 3) or not np.isfinite(window).all():
        return float("nan")
    a, b, c = window[0]
    d, _, f = window[1]
    g, h, i = window[2]
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx_m)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy_m)
    return float(np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))))


def pixel_spacing_m(res_x_deg: float, res_y_deg: float, lat: float) -> tuple[float, float]:
    """Convert a pixel size in degrees to metres at ``lat`` (spherical Earth)."""
    metres_per_degree = math.pi / 180 * EARTH_RADIUS_M
    return (
        abs(res_x_deg) * metres_per_degree * math.cos(math.radians(lat)),
        abs(res_y_deg) * metres_per_degree,
    )


def _read_tile_points(url: str, points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    with rasterio.open(url) as src:
        res_x, res_y = src.res
        nodata = src.nodata
        for point in points.itertuples(index=False):
            row, col = src.index(point.lon, point.lat)
            data = src.read(
                1,
                window=Window(col - 1, row - 1, 3, 3),
                boundless=True,
                fill_value=np.nan if nodata is None else nodata,
            ).astype("float64")
            if nodata is not None:
                data[data == nodata] = np.nan
            dx, dy = pixel_spacing_m(res_x, res_y, point.lat)
            rows.append(
                {
                    "point_id": point.point_id,
                    "elevation_m": data[1, 1],
                    "slope_deg": horn_slope_deg(data, dx, dy),
                    "dem_tile": url.rsplit("/", 1)[-1],
                }
            )
    return pd.DataFrame(rows)


def _candidate_tiles(lon: float, lat: float) -> list[str]:
    """Nominal tile first, then its eight neighbours, nearest edges first.

    The half-pixel shift of the footprints means a point can lie in a
    neighbouring tile; neighbours are opened only when needed.
    """
    frac_lat, frac_lon = lat - math.floor(lat), lon - math.floor(lon)
    d_lats = [0, -1, 1] if frac_lat < 0.5 else [0, 1, -1]
    d_lons = [0, -1, 1] if frac_lon < 0.5 else [0, 1, -1]
    offsets = sorted(
        ((a, b) for a in d_lats for b in d_lons),
        key=lambda ab: (abs(ab[0]) + abs(ab[1]), d_lats.index(ab[0]), d_lons.index(ab[1])),
    )
    return [tile_name(lat + a, lon + b) for a, b in offsets]


class _TileReader:
    """Opens tiles on demand and reads single pixels by coordinate."""

    def __init__(self, url_for: Callable[[str], str]):
        self._url_for = url_for
        self._open: dict[str, rasterio.io.DatasetReader | None] = {}

    def _get(self, name: str):
        if name not in self._open:
            try:
                self._open[name] = rasterio.open(self._url_for(name))
            except RasterioIOError:
                self._open[name] = None  # missing tile (sea) or unavailable
        return self._open[name]

    def locate(self, lon: float, lat: float):
        """The open tile whose footprint contains (lon, lat), with its name."""
        for name in _candidate_tiles(lon, lat):
            src = self._get(name)
            if src is None:
                continue
            b = src.bounds
            if b.left <= lon < b.right and b.bottom < lat <= b.top:
                return name, src
        return None, None

    def value(self, lon: float, lat: float) -> float:
        _, src = self.locate(lon, lat)
        if src is None:
            return float("nan")
        row, col = src.index(lon, lat)
        v = float(src.read(1, window=Window(col, row, 1, 1))[0, 0])
        return float("nan") if src.nodata is not None and v == src.nodata else v

    def close(self) -> None:
        for src in self._open.values():
            if src is not None:
                src.close()
        self._open.clear()


def _read_across_tiles(points: pd.DataFrame, reader: _TileReader) -> pd.DataFrame:
    """Elevation and slope for points whose window crosses a tile edge.

    The nine pixel centres are laid out on the grid of the tile containing
    the point, and each is read from whichever tile contains it. Adjacent
    tiles share the same pixel grid within a latitude band, so this
    reproduces the window of a seamless mosaic.
    """
    rows = []
    for point in points.itertuples(index=False):
        name, src = reader.locate(point.lon, point.lat)
        if src is None:
            rows.append(
                {
                    "point_id": point.point_id,
                    "elevation_m": np.nan,
                    "slope_deg": np.nan,
                    "dem_tile": "none",
                }
            )
            continue
        row, col = src.index(point.lon, point.lat)
        data = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                x, y = src.xy(row - 1 + i, col - 1 + j)
                data[i, j] = reader.value(x, y)
        res_x, res_y = src.res
        dx, dy = pixel_spacing_m(res_x, res_y, point.lat)
        rows.append(
            {
                "point_id": point.point_id,
                "elevation_m": data[1, 1],
                "slope_deg": horn_slope_deg(data, dx, dy),
                "dem_tile": f"{name}.tif (edge)",
            }
        )
    return pd.DataFrame(rows)


def extract_dem(
    points: pd.DataFrame,
    cache: Path | None = None,
    max_attempts: int = 3,
    url_for: Callable[[str], str] = tile_url,
) -> pd.DataFrame:
    """Elevation and slope for every point, tile by tile.

    ``points`` needs ``point_id``, ``lon`` and ``lat``. If ``cache`` is given,
    each finished tile is written there (Parquet per tile), so an
    interrupted run resumes where it stopped. Points with a missing value
    after the tile pass are re-read across tile edges (not cached; a few
    dozen points). ``url_for`` maps a tile name to a GDAL path; tests pass
    local files.
    """
    work = points[["point_id", "lon", "lat"]].copy()
    work["tile"] = [tile_name(lat, lon) for lat, lon in zip(work["lat"], work["lon"], strict=True)]
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    tiles = sorted(work["tile"].unique())
    with rasterio.Env(**GDAL_ENV):
        for k, name in enumerate(tiles, start=1):
            cached = cache / f"{name}.parquet" if cache is not None else None
            if cached is not None and cached.exists():
                parts.append(pd.read_parquet(cached))
                continue
            subset = work[work["tile"].eq(name)]
            for attempt in range(1, max_attempts + 1):
                try:
                    result = _read_tile_points(url_for(name), subset)
                    break
                except RasterioIOError as error:
                    if attempt == max_attempts:
                        LOG.warning(
                            "Tile %s unavailable (%s); %d points left missing",
                            name,
                            error,
                            len(subset),
                        )
                        result = pd.DataFrame(
                            {
                                "point_id": subset["point_id"],
                                "elevation_m": np.nan,
                                "slope_deg": np.nan,
                                "dem_tile": f"{name}.tif (unavailable)",
                            }
                        )
                    else:
                        time.sleep(2 * attempt)
            if cached is not None:
                result.to_parquet(cached, index=False)
            parts.append(result)
            LOG.info("DEM tile %d/%d %s: %d points", k, len(tiles), name, len(subset))
        out = pd.concat(parts, ignore_index=True)

        incomplete = out["elevation_m"].isna() | out["slope_deg"].isna()
        if incomplete.any():
            redo = work[work["point_id"].isin(out.loc[incomplete, "point_id"])]
            reader = _TileReader(url_for)
            try:
                fixed = _read_across_tiles(redo, reader)
            finally:
                reader.close()
            out = pd.concat([out[~incomplete], fixed], ignore_index=True)
            LOG.info(
                "DEM edge pass: %d points re-read across tiles, %d still incomplete",
                len(redo),
                int((fixed["elevation_m"].isna() | fixed["slope_deg"].isna()).sum()),
            )
    order = pd.Series(range(len(points)), index=points["point_id"].to_numpy())
    return out.sort_values("point_id", key=lambda s: s.map(order)).reset_index(drop=True)
