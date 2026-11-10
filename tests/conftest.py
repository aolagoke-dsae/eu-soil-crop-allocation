"""Shared fixtures: the real configuration and tiny synthetic source files.

No test reads the licensed LUCAS data; every source is written to a
temporary directory with the same columns and codes as the real files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eusoil.config import load_config, resolve_paths

SOIL_HEADER = (
    "Depth,POINTID,pH_CaCl2,pH_H2O,EC,OC,CaCO3,P,N,K,OC (20-30 cm),"
    "CaCO3 (20-30 cm),Ox_Al,Ox_Fe,NUTS_0,NUTS_1,NUTS_2,NUTS_3,TH_LAT,TH_LONG,"
    "SURVEY_DATE,Elev,LC,LU,LC0_Desc,LC1_Desc,LU1_Desc"
)

# point, depth, pH, P, CaCO3, OC, country, LC, LU, lat, lon
SOIL_ROWS = [
    (1, "0-20 cm", "6.1", "< LOD", "", "21.5", "DE", "B11", "U111", 50.10, 8.60),
    (2, "0-20 cm", "7.4", "35.2", "< LOD", "18.0", "DE", "B16", "U111", 50.20, 8.70),
    (3, "0-20 cm", "7.9", "4.0", "120", "15.2", "FR", "E20", "U111", 46.00, 2.00),
    (4, "0-10 cm", "5.5", "20.0", "", "30.0", "FR", "B11", "U111", 46.10, 2.10),
    (5, "0-20 cm", "6.0", "22.0", "", "<0.0", "MT", "B11", "U111", 35.90, 14.40),
    (6, "0-20 cm", "5.0", "15.0", "", "40.0", "DE", "C10", "U120", 51.00, 9.00),
    (7, "0-20 cm", "6.5", "12.0", "", "25.0", "DE", "E20", "U370", 51.10, 9.10),
    (8, "0-20 cm", "6.6", "13.0", "", "24.0", "FR", "E30", "U111", 45.00, 3.00),
    (9, "0-20 cm", "6.7", "11.0", "<  LOD", "23.0", "FR", "Bx1", "U111", 45.10, 3.10),
]


def _soil_line(r) -> str:
    point, depth, ph, p, caco3, oc, country, lc, lu, lat, lon = r
    return (
        f"{depth},{point},{ph},{float(ph) + 0.5},12.3,{oc},{caco3},{p},2.1,150.0,,,,,"
        f"{country},{country}1,{country}11,{country}111,{lat},{lon},06-07-18,10234,"
        f"{lc},{lu},Cover,Desc {lc},Use"
    )


@pytest.fixture
def config() -> dict:
    cfg = load_config()
    # Small synthetic data cannot meet the real minimum class size.
    cfg["target"]["min_class_size"] = 1
    return cfg


@pytest.fixture
def source_files(tmp_path: Path, config: dict):
    """Write synthetic soil, harmonised-survey and texture files; return Paths."""
    paths = resolve_paths(config, root=tmp_path)
    paths.soil.parent.mkdir(parents=True)
    paths.soil.write_text("\n".join([SOIL_HEADER] + [_soil_line(r) for r in SOIL_ROWS]) + "\n")

    harmo = pd.DataFrame(
        {
            "point_id": [r[0] for r in SOIL_ROWS],
            "lc1": [r[7].upper() for r in SOIL_ROWS],
            "lc1_label": "label",
            "lc1_perc": "> 75 %",
            "lu1": [r[8] for r in SOIL_ROWS],
            "obs_type": "In situ < 100 m",
            "obs_dist": "1",
            "th_gps_dist": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 500.0, 2.0],
            "gps_lat": [r[9] for r in SOIL_ROWS],
            "gps_long": [abs(r[10]) for r in SOIL_ROWS],
            "gps_ew": ["East"] * len(SOIL_ROWS),
            "gps_prec": 5,
            "wm": "",
            "grazing": "",
            "parcel_area_ha": 1.0,
        }
    )
    harmo.loc[1, "gps_lat"] = 88.88889  # placeholder value seen in the real file
    paths.landcover_2018.parent.mkdir(parents=True)
    harmo.to_csv(paths.landcover_2018, index=False)

    texture = pd.DataFrame(
        {
            "POINTID": [r[0] for r in SOIL_ROWS],
            "NUTS_0": "DE",
            "Lucas2009": "YES",
            "Lucas2012": "NO",
            "Lucas2015": "NO",
            "Lucas2018": "NO",
            "Coarse": 10,
            "Clay": 20,
            "Sand": 40,
            "Silt": 40,
            "PSDAvailable": "Yes",
            "TextureFromSurvey": 2009,
            "USDA": "loam",
            "ISSS": "loam",
        }
    )
    texture.loc[0, "PSDAvailable"] = "No"
    paths.texture.parent.mkdir(parents=True)
    texture.to_csv(paths.texture, index=False)
    return paths
