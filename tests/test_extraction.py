"""Tests for the raster and climate extraction, on synthetic rasters and arrays."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from eusoil.ingest.climate import extract_worldclim, sample_raster, worldclim_files
from eusoil.ingest.dem import extract_dem, horn_slope_deg, pixel_spacing_m, tile_name
from eusoil.ingest.terraclimate import (
    bioclim_from_monthly,
    extract_terraclimate,
    monthly_climatology,
    nearest_valid_cells,
    zarr_open_kwargs,
)


def _write_raster(path: Path, data: np.ndarray, nodata: float = -9999.0) -> None:
    # 0.1-degree pixels with the upper-left corner at (0 E, 50 N).
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 50.0, 0.1, 0.1),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_sample_raster_reads_pixels_and_fills_sea_points(tmp_path):
    data = np.arange(25, dtype=float).reshape(5, 5)
    data[0, 0] = -9999.0  # a "sea" pixel in the corner
    path = tmp_path / "r.tif"
    _write_raster(path, data)
    lon = np.array([0.25, 0.05, 5.0])  # pixel (2,2), sea pixel (0,0), outside
    lat = np.array([49.75, 49.95, 40.0])
    values, distance = sample_raster(path, lon, lat, fill_radius_px=1)
    assert values[0] == 12.0 and distance[0] == 0.0
    assert values[1] in (1.0, 5.0) and distance[1] == 1.0  # an adjacent land pixel
    assert np.isnan(values[2]) and np.isnan(distance[2])


def test_worldclim_requires_all_nineteen_layers(tmp_path):
    for n in range(1, 19):
        _write_raster(tmp_path / f"wc2.1_30s_bio_{n}.tif", np.ones((2, 2)))
    with pytest.raises(FileNotFoundError, match="19"):
        worldclim_files(tmp_path)
    _write_raster(tmp_path / "wc2.1_30s_bio_19.tif", np.full((2, 2), 7.0))
    points = pd.DataFrame({"point_id": [1], "lon": [0.05], "lat": [49.95]})
    out = extract_worldclim(points, tmp_path)
    assert out.loc[0, "wc30s_bio19"] == 7.0 and out.loc[0, "wc30s_bio_fill_px"] == 0.0


@pytest.mark.parametrize(
    "lat, lon, expected",
    [
        (47.5, 16.2, "Copernicus_DSM_COG_30_N47_00_E016_00_DEM"),
        (43.4, -5.6, "Copernicus_DSM_COG_30_N43_00_W006_00_DEM"),
        (50.0, -0.5, "Copernicus_DSM_COG_30_N50_00_W001_00_DEM"),
    ],
)
def test_dem_tile_names(lat, lon, expected):
    assert tile_name(lat, lon) == expected


def test_horn_slope_on_planes():
    dx = dy = 30.0
    flat = np.full((3, 3), 100.0)
    assert horn_slope_deg(flat, dx, dy) == 0.0
    # A plane rising 1 m per metre eastwards has a 45 degree slope.
    x = np.array([-dx, 0, dx])
    ramp = np.tile(x, (3, 1)) + 100.0
    assert horn_slope_deg(ramp, dx, dy) == pytest.approx(45.0)
    ramp[0, 0] = np.nan
    assert np.isnan(horn_slope_deg(ramp, dx, dy))


def test_pixel_spacing_shrinks_with_latitude():
    dx_eq, dy_eq = pixel_spacing_m(1 / 1200, 1 / 1200, 0.0)
    dx_60, _ = pixel_spacing_m(1 / 1200, 1 / 1200, 60.0)
    assert dy_eq == pytest.approx(92.66, rel=1e-3)
    assert dx_60 == pytest.approx(dx_eq / 2, rel=1e-6)


def test_bioclim_definitions_on_known_series():
    months = np.arange(12)
    tmax = np.tile((10 + months).reshape(12, 1), (1, 2)).astype(float)
    tmin = tmax - 10
    ppt = np.tile(np.array([5, 1, 2, 50, 50, 50, 50, 50, 50, 50, 50, 3.0]).reshape(12, 1), (1, 2))
    bio = bioclim_from_monthly(tmax, tmin, ppt)
    assert sorted(bio, key=lambda k: int(k[3:])) == [f"bio{i}" for i in range(1, 20)]
    tavg = (tmax[:, 0] + tmin[:, 0]) / 2  # 5, 6, ..., 16
    assert bio["bio1"][0] == pytest.approx(tavg.mean())
    assert bio["bio2"][0] == pytest.approx(10.0)
    assert bio["bio5"][0] == 21.0 and bio["bio6"][0] == 0.0 and bio["bio7"][0] == 21.0
    assert bio["bio3"][0] == pytest.approx(10.0 / 21.0 * 100)
    assert bio["bio12"][0] == pytest.approx(ppt[:, 0].sum())
    assert bio["bio13"][0] == 50.0 and bio["bio14"][0] == 1.0
    # Driest quarter: Jan + Feb + Mar = 5 + 1 + 2 = 8 (Dec + Jan + Feb = 9).
    assert bio["bio17"][0] == pytest.approx(8.0)
    assert bio["bio9"][0] == pytest.approx(tavg[[0, 1, 2]].mean())  # its temperature
    # Warmest quarter Oct-Dec (14, 15, 16 deg C): its precipitation is 50 + 50 + 3.
    assert bio["bio10"][0] == pytest.approx(15.0)
    assert bio["bio18"][0] == pytest.approx(103.0)
    # Coldest quarter Jan-Mar (5, 6, 7 deg C): the driest quarter as well.
    assert bio["bio11"][0] == pytest.approx(6.0)
    assert bio["bio19"][0] == pytest.approx(8.0)
    assert bio["bio16"][0] == pytest.approx(150.0)
    ppt_wrap = ppt.copy()
    ppt_wrap[2] = 40.0  # March wetter: the quarter wrapping the year end (9) is now driest
    assert bioclim_from_monthly(tmax, tmin, ppt_wrap)["bio17"][0] == pytest.approx(9.0)


def test_bioclim_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        bioclim_from_monthly(np.ones((11, 1)), np.ones((11, 1)), np.ones((11, 1)))


def test_nearest_valid_cell_skips_missing_cells():
    grid_lat = np.array([50.0, 49.0])
    grid_lon = np.array([8.0, 9.0])
    valid = np.array([[False, True], [True, True]])
    row, col, dist = nearest_valid_cells(
        grid_lat, grid_lon, valid, np.array([50.0]), np.array([8.0])
    )
    assert (row[0], col[0]) in {(0, 1), (1, 0)}
    assert 60 < dist[0] < 115  # about 72 km east or 111 km south


def test_zarr_open_kwargs_reads_nested_storage_options():
    # Layout of the Planetary Computer TerraClimate entry (November 2026).
    fields = {
        "xarray:open_kwargs": {
            "chunks": {},
            "engine": "zarr",
            "consolidated": True,
            "storage_options": {"account_name": "cpdataeuwest", "credential": "token"},
        }
    }
    kwargs = zarr_open_kwargs(fields)
    assert kwargs["storage_options"] == {"account_name": "cpdataeuwest", "credential": "token"}
    assert kwargs["consolidated"] is True and kwargs["engine"] == "zarr"
    assert "storage_options" in fields["xarray:open_kwargs"]  # input left unchanged


def test_zarr_open_kwargs_accepts_the_legacy_field_and_defaults_the_engine():
    kwargs = zarr_open_kwargs({"xarray:storage_options": {"account_name": "old"}})
    assert kwargs == {"storage_options": {"account_name": "old"}, "engine": "zarr"}


def _synthetic_terraclimate(years=(2015, 2016)):
    """A small cube shaped like the TerraClimate store: descending lat, monthly time."""
    import xarray as xr

    time = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-01", freq="MS")
    lat = np.array([52.0, 51.0, 50.0])  # north to south, as in the real store
    lon = np.array([8.0, 9.0, 10.0])
    month = time.month.to_numpy()[:, None, None]
    offset = (time.year.to_numpy() - years[0])[:, None, None]  # 0 or 1
    base = np.ones((len(time), lat.size, lon.size))
    tmax = base * (10.0 + month + offset)
    ppt = base * (month * 10.0)
    ppt[:, 0, 0] = np.nan  # a "sea" cell
    return xr.Dataset(
        {
            "tmax": (("time", "lat", "lon"), tmax),
            "tmin": (("time", "lat", "lon"), tmax - 10.0),
            "ppt": (("time", "lat", "lon"), ppt),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_monthly_climatology_averages_each_calendar_month_over_years():
    clim = monthly_climatology(
        _synthetic_terraclimate(),
        years=("2015-01-01", "2016-12-31"),
        box={"lat": (49.0, 53.0), "lon": (7.0, 11.0)},
    )
    assert clim["tmax"].dims == ("month", "lat", "lon")
    # Mean of (10 + m) and (11 + m) for each month m.
    assert np.allclose(clim["tmax"].isel(lat=1, lon=1).to_numpy(), 10.5 + np.arange(1, 13))
    assert np.isnan(clim["ppt"].isel(lat=0, lon=0)).all()


def test_monthly_climatology_rejects_an_empty_period():
    with pytest.raises(ValueError, match="No TerraClimate months"):
        monthly_climatology(_synthetic_terraclimate(), years=("2030-01-01", "2030-12-31"))


def test_monthly_climatology_rejects_a_period_the_store_does_not_cover():
    # The store holds 2015-2016; asking for 2015-2017 must not average two years quietly.
    with pytest.raises(ValueError, match=r"lacks the years \[2017\]"):
        monthly_climatology(_synthetic_terraclimate(), years=("2015-01-01", "2017-12-31"))


def test_extract_terraclimate_uses_the_nearest_cell_with_data():
    clim = monthly_climatology(
        _synthetic_terraclimate(),
        years=("2015-01-01", "2016-12-31"),
        box={"lat": (49.0, 53.0), "lon": (7.0, 11.0)},
    )
    points = pd.DataFrame({"point_id": [1, 2], "lat": [52.0, 50.0], "lon": [8.0, 10.0]})
    out = extract_terraclimate(points, period="2015-2018", climatology=clim)
    assert out["point_id"].tolist() == [1, 2]
    assert out["tc1518_cell_km"].iloc[0] > 50  # moved off the missing cell
    assert out["tc1518_cell_km"].iloc[1] < 1
    assert np.allclose(out["tc1518_bio12"], 780.0)  # 10 * (1 + ... + 12)
    assert np.allclose(out["tc1518_bio1"], 10.5 + 6.5 - 5.0)
    assert {f"tc1518_bio{i}" for i in range(1, 20)} <= set(out.columns)


def test_extract_terraclimate_names_columns_by_period_and_rejects_unknown_ones():
    clim = monthly_climatology(
        _synthetic_terraclimate(),
        years=("2015-01-01", "2016-12-31"),
        box={"lat": (49.0, 53.0), "lon": (7.0, 11.0)},
    )
    points = pd.DataFrame({"point_id": [1], "lat": [50.0], "lon": [10.0]})
    assert "tc9120_bio1" in extract_terraclimate(points, climatology=clim).columns
    with pytest.raises(ValueError, match="Unknown period"):
        extract_terraclimate(points, period="1970-2000", climatology=clim)


def _write_glo_like_tiles(directory: Path, res: float = 0.1) -> None:
    """2 x 2 tiles on a plane z = 100 lon + 1000 lat, with GLO-90's half-pixel shift.

    As in the real tiles, the footprint of tile N49 E008 is shifted up and
    left by half a pixel: about 49.05-50.05 N and 7.95-8.95 E.
    """
    n = round(1 / res)
    for lat0 in (49, 50):
        for lon0 in (8, 9):
            left, top = lon0 - res / 2, lat0 + 1 + res / 2
            lon = left + res * (np.arange(n) + 0.5)
            lat = top - res * (np.arange(n) + 0.5)
            z = (100.0 * lon[None, :] + 1000.0 * lat[:, None]).astype("float32")
            with rasterio.open(
                directory / f"{tile_name(lat0 + 0.5, lon0 + 0.5)}.tif",
                "w",
                driver="GTiff",
                height=n,
                width=n,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(left, top, res, res),
                nodata=-32767.0,
            ) as dst:
                dst.write(z, 1)


def test_extract_dem_reads_windows_across_tile_edges(tmp_path):
    _write_glo_like_tiles(tmp_path)
    points = pd.DataFrame(
        {
            "point_id": [1, 2, 3],
            # interior; just south of the N50 tile footprint and at an E-W edge;
            # off the four tiles (sea / missing tile)
            "lat": [49.50, 50.02, 47.50],
            "lon": [8.50, 9.01, 8.50],
        }
    )
    out = extract_dem(points, url_for=lambda name: str(tmp_path / f"{name}.tif"))
    assert out["point_id"].tolist() == [1, 2, 3]  # input order kept
    interior, edge, outside = out.itertuples(index=False)
    assert np.isfinite(edge.slope_deg) and edge.dem_tile.endswith("(edge)")
    # Same plane, so the slope at the edge point matches the interior one.
    assert edge.slope_deg == pytest.approx(interior.slope_deg, rel=0.02)
    # Pixel centres sit on whole tenths here; nearest to (9.01, 50.02) is (9.0, 50.0).
    assert edge.elevation_m == pytest.approx(900.0 + 50000.0)
    assert np.isnan(outside.elevation_m) and np.isnan(outside.slope_deg)
