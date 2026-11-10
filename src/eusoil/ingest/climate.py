"""Extract WorldClim 2.1 bioclimatic variables at the survey points.

WorldClim layers are global GeoTIFFs in EPSG:4326 named
``wc2.1_<res>_bio_<n>.tif``. Values are read at the pixel containing each
point. A coastal point whose pixel is sea (no data) takes the nearest pixel
with data within ``fill_radius_px`` pixels; how many points needed this, and
how far, is returned so the audit can report it.

WorldClim is an internal resolution check only (protocol A7): its licence
does not allow redistribution, so nothing derived from it is published. The
primary climate block comes from TerraClimate (``eusoil.ingest.terraclimate``).

Reference: Fick, S. E. & Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial
resolution climate surfaces for global land areas. International Journal of
Climatology, 37(12), 4302-4315. https://doi.org/10.1002/joc.5086
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

_BIO_FILE = re.compile(r"wc2\.1_(?P<res>[^_]+)_bio_(?P<n>\d{1,2})\.tif$")


def worldclim_files(directory: Path) -> dict[int, Path]:
    """Map bio-variable number to file; requires all 19 to be present."""
    files: dict[int, Path] = {}
    for path in directory.glob("*.tif"):
        match = _BIO_FILE.search(path.name)
        if match:
            files[int(match.group("n"))] = path
    missing = sorted(set(range(1, 20)) - set(files))
    if missing:
        raise FileNotFoundError(f"WorldClim bio variables missing in {directory}: {missing}")
    return dict(sorted(files.items()))


def _nearest_valid(
    src: rasterio.io.DatasetReader, lon: float, lat: float, radius: int
) -> tuple[float, float]:
    """Value of the nearest pixel with data within ``radius`` pixels, and its distance in pixels."""
    row, col = src.index(lon, lat)
    size = 2 * radius + 1
    window = src.read(
        1, window=Window(col - radius, row - radius, size, size), boundless=True, masked=True
    )
    if window.mask.all():
        return float("nan"), float("nan")
    rr, cc = np.nonzero(~np.ma.getmaskarray(window))
    distance = np.hypot(rr - radius, cc - radius)
    k = int(np.argmin(distance))
    return float(window[rr[k], cc[k]]), float(distance[k])


def sample_raster(
    path: Path, lon: np.ndarray, lat: np.ndarray, fill_radius_px: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Read the pixel value under each (lon, lat) point.

    Returns ``(values, fill_distance_px)``: ``fill_distance_px`` is 0 where
    the point's own pixel had data, the distance in pixels to the pixel used
    where a neighbour filled in, and NaN where nothing was found within
    ``fill_radius_px``. The raster must be in EPSG:4326.
    """
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f"{path.name}: expected EPSG:4326, found {src.crs}")
        # masked=True marks both nodata pixels and points outside the raster.
        sampled = [v[0] for v in src.sample(zip(lon, lat, strict=True), indexes=1, masked=True)]
        values = np.array([np.nan if np.ma.is_masked(v) else float(v) for v in sampled])
        distance = np.where(np.isnan(values), np.nan, 0.0)
        for i in np.flatnonzero(np.isnan(values)):
            values[i], distance[i] = _nearest_valid(
                src, float(lon[i]), float(lat[i]), fill_radius_px
            )
    return values, distance


def extract_worldclim(
    points: pd.DataFrame, directory: Path, prefix: str = "wc30s_bio", fill_radius_px: int = 3
) -> pd.DataFrame:
    """Return one row per point with ``<prefix>1`` ... ``<prefix>19``.

    ``points`` needs ``point_id``, ``lon`` and ``lat`` (EPSG:4326). The
    column ``<prefix>_fill_px`` holds the largest fill distance across the
    19 layers (0 = every value from the point's own pixel).
    """
    lon = points["lon"].to_numpy(dtype="float64")
    lat = points["lat"].to_numpy(dtype="float64")
    out = pd.DataFrame({"point_id": points["point_id"].to_numpy()})
    fill = np.zeros(len(points))
    for number, path in worldclim_files(directory).items():
        values, distance = sample_raster(path, lon, lat, fill_radius_px)
        out[f"{prefix}{number}"] = values
        fill = np.fmax(fill, np.nan_to_num(distance, nan=np.inf))
    out[f"{prefix}_fill_px"] = np.where(np.isinf(fill), np.nan, fill)
    return out
