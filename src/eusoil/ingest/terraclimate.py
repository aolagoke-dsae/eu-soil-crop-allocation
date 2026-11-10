"""TerraClimate bioclimatic variables at the survey points (protocol A7).

TerraClimate (Abatzoglou et al., 2018; monthly, about 4 km, public domain
under CC0) supplies the climate block C. It is read from Microsoft's
Planetary Computer as a Zarr store; only the European window and the
months of the requested period are loaded. Two periods are used:

* ``1991-2020``, the current WMO standard normal: the primary climate block,
  columns ``tc9120_bio*``.
* ``2015-2018``, the years just before the 2018 survey: the climate-period
  sensitivity run, columns ``tc1518_bio*``.

From each period's monthly climatology, the 19 bioclimatic variables are
computed with WorldClim's definitions (as in ``dismo::biovars``; quarters
are three consecutive months, wrapping the year). The models use the six in
``config['blocks']['C']`` (bio1, bio4, bio6, bio12, bio15, bio17); all 19 are
kept for the cluster-based climate subset check.

Points whose nearest cell is missing (at the coast) take the nearest cell
with data. The distance to the cell used is stored as ``<tag>_cell_km``, so
the analysis can check it.

Reference: Abatzoglou, J. T., Dobrowski, S. Z., Parks, S. A. & Hegewisch,
K. C. (2018). TerraClimate. Scientific Data, 5, 170191.
https://doi.org/10.1038/sdata.2017.191
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LOG = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "terraclimate"
VARIABLES = ("tmax", "tmin", "ppt")
PERIODS = {
    "1991-2020": {"years": ("1991-01-01", "2020-12-31"), "tag": "tc9120"},
    "2015-2018": {"years": ("2015-01-01", "2018-12-31"), "tag": "tc1518"},
}
PRIMARY_PERIOD = "1991-2020"
EUROPE_BOX = {"lat": (30.0, 75.0), "lon": (-15.0, 40.0)}


def zarr_open_kwargs(extra_fields: dict) -> dict:
    """Keyword arguments for ``xarray.open_dataset`` from a STAC asset's fields.

    The xarray-assets STAC extension has carried the storage options in two
    places: a separate ``xarray:storage_options`` field (older catalogue
    entries) and, as the Planetary Computer's TerraClimate entry now does,
    nested inside ``xarray:open_kwargs``. Both are accepted; a nested value
    wins over the legacy field. The engine defaults to ``zarr``.
    """
    kwargs = dict(extra_fields.get("xarray:open_kwargs", {}))
    storage = dict(extra_fields.get("xarray:storage_options", {}))
    storage.update(kwargs.pop("storage_options", {}) or {})
    if storage:
        kwargs["storage_options"] = storage
    kwargs.setdefault("engine", "zarr")
    return kwargs


def open_terraclimate():  # pragma: no cover - needs network access
    """Open the TerraClimate Zarr store on the Planetary Computer.

    Requires ``pystac-client``, ``planetary-computer``, ``xarray``, ``zarr``
    and ``adlfs``. No account is needed; the asset is signed anonymously.
    """
    import planetary_computer
    import pystac_client
    import xarray as xr

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    collection = catalog.get_collection(COLLECTION)
    asset = collection.assets["zarr-abfs"]
    kwargs = zarr_open_kwargs(asset.extra_fields)
    # The catalogue asks for dask chunks (``chunks: {}``). Dask is not a
    # project dependency: without it xarray indexes the store lazily and
    # reads only the Zarr chunks that the selection below touches.
    kwargs["chunks"] = None
    return xr.open_dataset(asset.href, **kwargs)


def monthly_climatology(ds, years, variables=VARIABLES, box=EUROPE_BOX):
    """Mean of each calendar month over ``years`` in the window ``box``.

    Reads one variable and one year at a time and keeps running sums in
    float64, so peak memory stays at about one year of one variable for the
    window (roughly 140 MB for Europe) rather than the whole 2015-2018 cube.
    Returns an ``xarray.Dataset`` with dimensions (month, lat, lon).
    """
    import xarray as xr

    lat = ds["lat"]
    ascending = bool(lat[0] < lat[-1])
    lat_slice = slice(*box["lat"]) if ascending else slice(*box["lat"][::-1])
    window = ds[list(variables)].sel(time=slice(*years), lat=lat_slice, lon=slice(*box["lon"]))
    if window.sizes["time"] == 0:
        raise ValueError(f"No TerraClimate months between {years[0]} and {years[1]}")

    month = window["time"].dt.month.to_numpy()
    year = window["time"].dt.year.to_numpy()
    first, last = int(years[0][:4]), int(years[1][:4])
    missing = sorted(set(range(first, last + 1)) - set(year.tolist()))
    if missing:
        raise ValueError(f"TerraClimate store lacks the years {missing} of {first}-{last}")
    shape = (12, window.sizes["lat"], window.sizes["lon"])
    out = {}
    for name in variables:
        total = np.zeros(shape)
        count = np.zeros(12)
        for k, yr in enumerate(np.unique(year), start=1):
            LOG.info("TerraClimate %s %d (%d/%d)", name, yr, k, last - first + 1)
            idx = np.flatnonzero(year == yr)
            block = window[name].isel(time=idx).transpose("time", "lat", "lon").to_numpy()
            for i, m in zip(idx, month[idx], strict=True):
                total[m - 1] += block[i - idx[0]]
                count[m - 1] += 1
        if (count == 0).any():
            raise ValueError(f"{name}: a calendar month has no data in {years}")
        out[name] = (("month", "lat", "lon"), (total / count[:, None, None]).astype("float32"))
    return xr.Dataset(
        out,
        coords={
            "month": np.arange(1, 13),
            "lat": window["lat"].to_numpy(),
            "lon": window["lon"].to_numpy(),
        },
    )


def _quarters(monthly: np.ndarray) -> np.ndarray:
    """Sums over the 12 wrapping three-month windows; shape (12, n)."""
    return np.stack([monthly[[m, (m + 1) % 12, (m + 2) % 12]].sum(axis=0) for m in range(12)])


def bioclim_from_monthly(
    tmax: np.ndarray, tmin: np.ndarray, ppt: np.ndarray
) -> dict[str, np.ndarray]:
    """The 19 WorldClim bioclimatic variables from 12-month arrays (axis 0 = month).

    Inputs have shape (12, n_points), temperatures in deg C and precipitation
    in mm. Returns ``bio1`` ... ``bio19``, each of length n_points, with the
    definitions of ``dismo::biovars`` (standard deviations with n - 1).
    """
    if tmax.shape[0] != 12 or tmin.shape != tmax.shape or ppt.shape != tmax.shape:
        raise ValueError("Expected three arrays of shape (12, n)")
    tavg = (tmax + tmin) / 2.0
    precip_plus_one = ppt + 1.0
    t_quarter = _quarters(tavg) / 3.0  # mean temperature of each quarter
    p_quarter = _quarters(ppt)  # precipitation of each quarter
    cols = np.arange(tmax.shape[1])

    def at(values: np.ndarray, quarter: np.ndarray) -> np.ndarray:
        return values[quarter, cols]

    bio5 = tmax.max(axis=0)
    bio6 = tmin.min(axis=0)
    bio2 = (tmax - tmin).mean(axis=0)
    return {
        "bio1": tavg.mean(axis=0),
        "bio2": bio2,
        "bio3": bio2 / (bio5 - bio6) * 100.0,
        "bio4": tavg.std(axis=0, ddof=1) * 100.0,
        "bio5": bio5,
        "bio6": bio6,
        "bio7": bio5 - bio6,
        "bio8": at(t_quarter, p_quarter.argmax(axis=0)),
        "bio9": at(t_quarter, p_quarter.argmin(axis=0)),
        "bio10": t_quarter.max(axis=0),
        "bio11": t_quarter.min(axis=0),
        "bio12": ppt.sum(axis=0),
        "bio13": ppt.max(axis=0),
        "bio14": ppt.min(axis=0),
        "bio15": precip_plus_one.std(axis=0, ddof=1) / precip_plus_one.mean(axis=0) * 100.0,
        "bio16": p_quarter.max(axis=0),
        "bio17": p_quarter.min(axis=0),
        "bio18": at(p_quarter, t_quarter.argmax(axis=0)),
        "bio19": at(p_quarter, t_quarter.argmin(axis=0)),
    }


def nearest_valid_cells(
    grid_lat: np.ndarray, grid_lon: np.ndarray, valid: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row, column and distance (km) of the nearest cell with data for each point.

    Cell centres and points are both projected to ETRS89-LAEA (EPSG:3035),
    so distances are in true kilometres across Europe.
    """
    from pyproj import Transformer

    to_laea = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    rows, cols = np.nonzero(valid)
    cell_x, cell_y = to_laea.transform(grid_lon[cols], grid_lat[rows])
    point_x, point_y = to_laea.transform(lon, lat)
    tree = cKDTree(np.column_stack([cell_x, cell_y]))
    distance_m, index = tree.query(np.column_stack([point_x, point_y]))
    return rows[index], cols[index], distance_m / 1000.0


def extract_terraclimate(
    points: pd.DataFrame, period: str = PRIMARY_PERIOD, climatology=None
) -> pd.DataFrame:
    """``<tag>_bio*`` for every point, plus ``<tag>_cell_km``, the distance to the cell used.

    ``period`` is a key of ``PERIODS``. ``climatology`` (month x lat x lon)
    may be passed directly, which the tests do; otherwise it is computed
    from the Planetary Computer store.
    """
    if period not in PERIODS:
        raise ValueError(f"Unknown period {period!r}; choose from {sorted(PERIODS)}")
    tag = PERIODS[period]["tag"]
    if climatology is None:
        climatology = monthly_climatology(open_terraclimate(), years=PERIODS[period]["years"])
    grid_lat = climatology["lat"].to_numpy()
    grid_lon = climatology["lon"].to_numpy()
    tmax = climatology["tmax"].transpose("month", "lat", "lon").to_numpy()
    tmin = climatology["tmin"].transpose("month", "lat", "lon").to_numpy()
    ppt = climatology["ppt"].transpose("month", "lat", "lon").to_numpy()
    valid = (
        np.isfinite(tmax).all(axis=0) & np.isfinite(tmin).all(axis=0) & np.isfinite(ppt).all(axis=0)
    )
    row, col, dist = nearest_valid_cells(
        grid_lat, grid_lon, valid, points["lat"].to_numpy(), points["lon"].to_numpy()
    )
    bio = bioclim_from_monthly(tmax[:, row, col], tmin[:, row, col], ppt[:, row, col])
    out = pd.DataFrame({"point_id": points["point_id"].to_numpy()})
    for name, values in bio.items():
        out[f"{tag}_{name}"] = values
    out[f"{tag}_cell_km"] = dist
    return out
