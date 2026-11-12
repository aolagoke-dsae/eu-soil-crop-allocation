# Soil, Climate and Crop Allocation in European Agriculture

**Which crop is grown where in European agriculture, and what do laboratory soil
measurements add to the predictive performance of a crop-allocation model, beyond
free climate and geographic information?**

Evidence from about 10,000 agricultural points of the EU's LUCAS 2018 topsoil
survey, with laboratory-measured N, P, K and pH, the crop recorded in the field,
and climate and terrain at each point.

> **Status: In progress.** The analysis
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
python -m eusoil points                            # clean and join the tables; write the audit
python -m eusoil dem                               # elevation and slope (internet access)
python -m eusoil terraclimate --period 1991-2020   # climate block: 1991-2020 normals (internet access)
python -m eusoil terraclimate --period 2015-2018   # climate-period sensitivity run (internet access)
python -m eusoil worldclim                         # optional internal check (local GeoTIFFs; not published)
python -m eusoil table                             # the modelling table
python -m eusoil precision                         # the H1 precision check
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
*Scientific Data*, 7, 352. Climate: TerraClimate (Abatzoglou et al., 2018),
public domain under CC0. WorldClim 2.1 (Fick & Hijmans, 2017) serves only as an
internal resolution check: its licence does not allow redistribution, so no
value derived from it is published here. Elevation: Copernicus DEM GLO-90
© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided
under COPERNICUS by the European Union and ESA.

## Author

**Dr. Adewole Olagoke** · [aolagoke-dsae](https://github.com/aolagoke-dsae)

Code licensed under the MIT License (see `LICENSE`).
