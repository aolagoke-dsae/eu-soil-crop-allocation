"""TerraClimate 2015-2018 at the survey points, for the climate-period sensitivity run.

The primary climate block uses WorldClim normals (1970-2000). The protocol's
sensitivity run asks whether the conclusions hold with the climate of the
years just before the 2018 survey. TerraClimate (monthly, about 4 km) is read
from Microsoft's Planetary Computer as a Zarr store; only the European
window and the months 2015-2018 are loaded.

From the 2015-2018 monthly climatology, the six bioclimatic variables of the
primary block are computed with WorldClim's definitions (as in
``dismo::biovars``):

* ``tc_bio1``  annual mean temperature, mean of (tmax + tmin) / 2
* ``tc_bio4``  temperature seasonality, 100 x standard deviation of monthly means
* ``tc_bio6``  minimum temperature of the coldest month
* ``tc_bio12`` annual precipitation
* ``tc_bio15`` precipitation seasonality, coefficient of variation of (monthly precipitation + 1)
* ``tc_bio17`` precipitation of the driest quarter (three consecutive months, wrapping the year)

Points whose nearest cell is missing (at the coast) take the nearest cell
with data. The distance to the cell used is stored, so the analysis can
check it.

Reference: Abatzoglou, J. T., Dobrowski, S. Z., Parks, S. A. & Hegewisch,
K. C. (2018). TerraClimate. Scientific Data, 5, 170191.
https://doi.org/10.1038/sdata.2017.191
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "terraclimate"
VARIABLES = ("tmax", "tmin", "ppt")
YEARS = ("2015-01-01", "2018-12-31")
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


def monthly_climatology(ds, variables=VARIABLES, years=YEARS, box=EUROPE_BOX):
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
    shape = (12, window.sizes["lat"], window.sizes["lon"])
    out = {}
    for name in variables:
        total = np.zeros(shape)
        count = np.zeros(12)
        for yr in np.unique(year):
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


def bioclim_from_monthly(
    tmax: np.ndarray, tmin: np.ndarray, ppt: np.ndarray
) -> dict[str, np.ndarray]:
    """WorldClim bioclimatic variables from 12-month arrays (axis 0 = month).

    Inputs have shape (12, n_points). Returns arrays of length n_points.
    """
    if tmax.shape[0] != 12 or tmin.shape != tmax.shape or ppt.shape != tmax.shape:
        raise ValueError("Expected three arrays of shape (12, n)")
    tavg = (tmax + tmin) / 2.0
    precip_plus_one = ppt + 1.0
    quarters = np.stack([ppt[[m, (m + 1) % 12, (m + 2) % 12]].sum(axis=0) for m in range(12)])
    return {
        "tc_bio1": tavg.mean(axis=0),
        "tc_bio4": tavg.std(axis=0, ddof=1) * 100.0,
        "tc_bio6": tmin.min(axis=0),
        "tc_bio12": ppt.sum(axis=0),
        "tc_bio15": precip_plus_one.std(axis=0, ddof=1) / precip_plus_one.mean(axis=0) * 100.0,
        "tc_bio17": quarters.min(axis=0),
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


def extract_terraclimate(points: pd.DataFrame, climatology=None) -> pd.DataFrame:
    """Return ``tc_bio*`` for every point, plus the distance to the cell used."""
    if climatology is None:
        climatology = monthly_climatology(open_terraclimate())
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
    out = pd.DataFrame({"point_id": points["point_id"].to_numpy(), **bio})
    out["tc_cell_distance_km"] = dist
    return out
