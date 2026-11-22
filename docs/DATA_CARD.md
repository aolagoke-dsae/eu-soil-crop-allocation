# Data card: LUCAS 2018 soil, climate and crop-allocation table

The modelling table behind this project: one row per LUCAS 2018 topsoil point on
agricultural land, with the crop group recorded in the field and the soil,
climate and terrain predictors. Built by `python -m eusoil table`; not committed,
because the LUCAS soil values may not be passed to third parties. Every number
below comes from the aggregated tables in [`reports/tables/`](../reports/tables/).

## Purpose

To predict the crop group grown at unsampled agricultural points in the EU and
to test what laboratory soil measurements add to climate and location
([protocol](PROTOCOL.md)). The table is not suited to field-level advice,
yield questions or causal claims about farmers' choices.

## Sources

| Source | Version | Used for | Licence |
|---|---|---|---|
| LUCAS 2018 topsoil (ESDAC, JRC) | LUCAS-SOIL-2018 v2 | pH, organic carbon, carbonate, N, P, K, EC; sampling depth | ESDAC: no redistribution; cite Orgiazzi et al. (2018) |
| LUCAS topsoil texture (ESDAC) | All surveys, 10 March 2025 | Clay, silt, sand | As above |
| Harmonised LUCAS land cover and use (d'Andrimont et al., 2020) | 2018 table | Crop code (LC1), land use (LU1), coordinates, observation type; the prediction domain | CC BY 4.0 |
| TerraClimate (Abatzoglou et al., 2018) | Microsoft Planetary Computer Zarr store | Bioclimatic variables, 1991-2020 (primary) and 2015-2018 (sensitivity) | CC0 |
| WorldClim 2.1 | 30 arc-seconds and 2.5 minutes | Internal resolution check only; nothing derived is published | Non-commercial, no redistribution |
| Copernicus DEM GLO-90 | Public AWS copy | Elevation and slope (Horn's method) | Copernicus DEM licence, with attribution |
| GISCO NUTS 2016 | 1:1 million | Country and NUTS-2 regions for maps | © EuroGeographics |

## Sample flow

| Step | Rule | Points left |
|---|---|---|
| Soil samples | LUCAS 2018 topsoil table, one row per point | 18,984 |
| Standard depth | 0-20 cm samples only | 18,744 |
| Excluded countries | Malta (2 points): too few to model | 18,742 |
| Agricultural land cover | Cropland (B) or grassland (E) at the point | 11,289 |
| Agricultural land use | U111: agriculture, without fallow and kitchen gardens | 10,284 |
| Crop group | Land-cover code in one of the eight groups | **9,906** |

Source: [`sample_flow.csv`](../reports/tables/sample_flow.csv). 1,981 of the 9,906
points form the sealed test set; analyses use the other 7,925, of which 7,252
have every predictor of the main models.

## Target

Eight crop groups, each with at least 300 points: wheat 1,768; other cereals
1,551; maize and root crops 1,094; oilseeds and pulses 973; fodder and temporary
grassland 708; permanent grassland 2,802; vineyards and orchards 567; olive
groves 443 ([`audit_class_counts.csv`](../reports/tables/audit_class_counts.csv)).
Root crops (248 points) were merged with maize and vineyards (297) with orchards,
so that every group has at least 300 points (protocol, Target definition). The
excluded land-cover codes and their counts are in
[`audit_excluded_codes.csv`](../reports/tables/audit_excluded_codes.csv).

## Predictors and missing values

| Block | Variables | Missing |
|---|---|---|
| Geography and terrain | Easting and northing (EPSG:3035), elevation, slope | 0 |
| Climate | TerraClimate 1991-2020: BIO1, BIO4, BIO6, BIO12, BIO15, BIO17 | 0 |
| Inherent soil | pH (CaCl₂), carbonate (log and presence), clay, silt | Clay and silt 8.6% |
| Nutrient soil | Total N, extractable P and K (log), P-below-detection flag | P 0.04% |
| Country | NUTS 0 code | 0 |

Points with a missing predictor are dropped from the models (complete cases);
the counts are reported with every result. Missing texture clusters by country:
Luxembourg 83%, Croatia 43%, Ireland 32%, the UK 31%, Finland 23%, Germany 14%
of points ([`audit_missing_by_country.csv`](../reports/tables/audit_missing_by_country.csv)),
so these countries are under-represented in the models.

## Data quality and known biases

- **Detection limits.** Values below the limit of detection are set to half the
  limit and flagged (A1). This matters for P: 10-40% of points per crop group are
  below its limit, most in olive groves (40%) and least in maize and root crops
  (10%) ([`audit_below_lod_by_group.csv`](../reports/tables/audit_below_lod_by_group.csv)).
  Three other treatments, including multiple imputation, leave every decision
  unchanged.
- **Texture from earlier surveys.** Clay and silt were measured in 2009 (7,192
  points), 2012 (713), 2015 (1,150) or, for one point, 2018 (A4).
- **Coordinates.** 9,887 points use the field GPS position, 19 the planned
  position (A6); 9,904 were observed in the field within 100 m.
- **Label certainty.** 7,564 points have the crop covering at least half of the
  observation window; the rest are mixed cover. Removing them changes no H1, H3a
  or H3b decision.
- **Organic soils.** 160 points have more than 120 g/kg organic carbon; excluding
  them changes no decision.
- **Uneven sampling.** Relative to its share of all 2018 LUCAS farmland points
  (the 2 km grid, sampled or not), Spain is sampled about 2.2 times and the Netherlands about 0.35
  times as densely as the EU average. Reweighting each country to its farmland
  share changes no decision.
- **One season.** Each point records the 2018 crop only. Rotations mean the same
  soil carries different arable crops in other years.
- **Soil reflects use.** Nutrients and organic matter are shaped by past
  management, so soil and crop are associated in both directions.

## Disclosure control

No point-level soil value leaves `data/`. Published outputs are counts, shares,
inner quantiles, model scores, and maps of soil variables as medians of 50 km
cells with at least 5 points (A15), or of regions with at least 10 points (A18).
Minima and maxima are not published, since each is an identifiable point's
value. Until 22 November 2026 the ALE tables published them for pH, P and K,
and the EDA histograms and violins drew them; all now stop at the 1st and 99th
percentiles (A21). Model scores by country are published for every country,
including four with fewer than 10 test points; they are not soil values, and
the model card notes that countries under 30 points are not interpretable.
`interpret_macro_f1_by_country.csv` omits countries under 10 points (A22).

## References

- Abatzoglou, J. T., Dobrowski, S. Z., Parks, S. A. & Hegewisch, K. C. (2018). TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958-2015. *Scientific Data*, 5, 170191.
- d'Andrimont, R. et al. (2020). Harmonised LUCAS in-situ land cover and use database for field surveys from 2006 to 2018 in the European Union. *Scientific Data*, 7, 352.
- Orgiazzi, A., Ballabio, C., Panagos, P., Jones, A. & Fernández-Ugalde, O. (2018). LUCAS Soil, the largest expandable soil dataset for Europe: a review. *European Journal of Soil Science*, 69(1), 140-153.
