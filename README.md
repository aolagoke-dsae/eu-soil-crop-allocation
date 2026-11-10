# Soil, Climate and Crop Allocation in European Agriculture

**Which crop is grown where in European agriculture, and what do laboratory soil
measurements add to the predictive performance of a crop-allocation model, beyond
free climate and geographic information?**

Evidence from about 10,000 agricultural points of the EU's LUCAS 2018 topsoil
survey, with laboratory-measured N, P, K and pH, the crop recorded in the field,
and climate and terrain at each point.

> **Status: Phase 1 (data ingestion and audit) in progress.** The analysis
> protocol, with its hypotheses and decision rules fixed before modelling, is in
> [`docs/PROTOCOL.md`](docs/PROTOCOL.md). Results will appear here once the
> sealed test blocks have been scored.

## Reproducing the pipeline

Requires Python 3.12 or later (developed on 3.14).

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1   macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest            # unit tests on synthetic data; no licensed data needed
```

Then place the sources as described in [`data/README.md`](data/README.md) and run:

```bash
python -m eusoil points          # clean and join the tables; write the audit
python -m eusoil worldclim       # climate at the points (local GeoTIFFs)
python -m eusoil dem             # elevation and slope (internet access)
python -m eusoil terraclimate    # climate 2015-2018 (internet access)
python -m eusoil table           # the modelling table
python -m eusoil precision       # the H1 precision check
```

## Related project

[Soil Carbon Kriging](https://github.com/aolagoke-dsae/soil-carbon-kriging) works
at the opposite end of the scale: it maps soil organic carbon within single fields
by kriging, with the uncertainty of each prediction and of each field's mean
stock. This project classifies crop allocation across the EU and tests what each
block of predictors adds.

## Data and licences

LUCAS 2018 topsoil data: European Soil Data Centre (ESDAC), European Commission,
Joint Research Centre. Orgiazzi, A., Ballabio, C., Panagos, P., Jones, A. &
Fernández-Ugalde, O. (2018). LUCAS Soil, the largest expandable soil dataset for
Europe: a review. *European Journal of Soil Science*, 69(1), 140-153. The soil
data are not redistributed here; the repository contains code, configuration and
aggregated results only.

Land cover: d'Andrimont, R. et al. (2020). Harmonised LUCAS in-situ land cover and
use database for field surveys from 2006 to 2018 in the European Union.
*Scientific Data*, 7, 352. Climate: WorldClim 2.1 (Fick & Hijmans, 2017) and
TerraClimate (Abatzoglou et al., 2018). Elevation: Copernicus DEM GLO-90.

## Author

**Dr. Adewole Olagoke** · [aolagoke-dsae](https://github.com/aolagoke-dsae)

Code licensed under the MIT License (see `LICENSE`).
