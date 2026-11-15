"""Read GISCO NUTS 2016 boundaries from the nested archive Eurostat distributes.

``ref-nuts-2016-01m.shp.zip`` holds one zip per layer and level (for example
``NUTS_RG_01M_2016_3035_LEVL_0.shp.zip``), each holding a shapefile. The
inner archive is read into memory and handed to GDAL as bytes, which works
the same on Windows and Linux without extracting anything to disk.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import geopandas as gpd

ARCHIVE = "ref-nuts-2016-01m.shp.zip"


def read_nuts(directory: Path, level: int, crs_code: str = "3035") -> gpd.GeoDataFrame:
    """Region polygons (``RG``) at NUTS ``level`` in EPSG:``crs_code``."""
    outer = Path(directory) / ARCHIVE
    inner = f"NUTS_RG_01M_2016_{crs_code}_LEVL_{level}.shp.zip"
    with zipfile.ZipFile(outer) as archive:
        if inner not in archive.namelist():
            raise FileNotFoundError(f"{inner} not found in {outer}")
        data = archive.read(inner)
    return gpd.read_file(io.BytesIO(data))
