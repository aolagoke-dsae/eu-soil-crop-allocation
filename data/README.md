# Data

Nothing under `data/` is committed. The LUCAS 2018 soil data may not be passed
to third parties (ESDAC licence), and every other file is either an open
download or rebuilt by the pipeline. This file explains how to obtain each
source and where to put it.

## Layout

```
data/
├── raw/         sources as downloaded; never edited
├── interim/     tables written by `python -m eusoil points | dem | terraclimate | worldclim`
└── processed/   the modelling table written by `python -m eusoil table`
```

## Sources

| Source | How to obtain | Place in `data/raw/` | Licence |
|---|---|---|---|
| LUCAS 2018 topsoil | Request on the ESDAC website (LUCAS 2018 TOPSOIL data); free, approval by e-mail | `LUCAS-SOIL-2018-data/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv` | ESDAC: no redistribution to third parties; cite the source |
| LUCAS topsoil texture (all surveys, version 10.03.2025) | Same ESDAC request | `LUCAS_Texture_All_10032025/LUCAS_Text_All_10032025.csv` | As above |
| Harmonised LUCAS land cover and use, 2018 table | JRC open data, `LUCAS_harmonised/1_table/lucas_harmo_uf_2018` | `LUCAS/LUCAS_harmonised/1_table/lucas_harmo_uf_2018.csv` | CC BY 4.0 |
| LUCAS Copernicus 2018 polygons | JRC open data, `LUCAS_2018_Copernicus` | `LUCAS/LUCAS_2018_Copernicus_1/` | CC BY 4.0 |
| WorldClim 2.1 bioclimatic variables, 30 arc-seconds (optional internal check) | worldclim.org, "Bioclimatic variables", 30s | `wc2.1_30s_bio/wc2.1_30s_bio_1.tif` ... `_19.tif` | Non-commercial use; no redistribution without permission, so nothing derived from it is published |
| WorldClim 2.1, 2.5 minutes (optional internal check) | worldclim.org | `wc2.1_2.5m_bio/` | As above |
| GISCO NUTS 2016, 1:1 million | Eurostat GISCO, NUTS 2016, shapefile archive | `NUTS2016/ref-nuts-2016-01m.shp.zip` | © EuroGeographics; Eurostat GISCO terms |
| Eurostat utilised agricultural area by country | Eurostat farm structure tables | `Eurostat-utilised-agricultural-area-by-country-dataset.xlsx` | Eurostat reuse policy (CC BY 4.0) |
| Copernicus DEM GLO-90 | **Not downloaded.** `python -m eusoil dem` reads only the pixels under the points from the public AWS copy | (none; results in `data/interim/dem.parquet`) | Copernicus DEM licence; attribution in `src/eusoil/ingest/dem.py` |
| TerraClimate, 1991-2020 (climate block) and 2015-2018 (sensitivity) | **Not downloaded.** `python -m eusoil terraclimate --period 1991-2020` (and `--period 2015-2018`) reads the Planetary Computer Zarr store | (none; results in `data/interim/terraclimate_1991_2020.parquet` and `terraclimate_2015_2018.parquet`) | Public domain (CC0) |

The NUTS codes in the soil file follow the **2016** edition (for example
Croatia's HR03 and HR04), so the boundaries must be the 2016 edition too.

## Helper script

`scripts/downloads/lucas_cover_ingestion_v2.py` mirrors the JRC open-data
folder `LUCAS/LUCAS_COVER/LUCAS2018/` into `data/raw/LUCAS/LUCAS_COVER/`. The
pipeline does not use that archive (it reads the harmonised 2018 table above);
the script is kept for reference. It needs `requests` and `beautifulsoup4`,
which are not in `requirements.txt`.
