# Model card: crop-group classifier for EU farmland (LUCAS 2018)

Dr. Adewole Olagoke · 22 November 2026 · version 1.0

The final model of this project: a random forest that predicts which of eight
crop groups is grown at an agricultural point in the EU from its location,
terrain, climate and laboratory-measured topsoil. How it was chosen and tested
is in the [validation report](VALIDATION_REPORT.md); the data are described in
the [data card](DATA_CARD.md); every choice was fixed in advance in the
[protocol](PROTOCOL.md).

## At a glance

| | |
|---|---|
| Task | Eight-class classification of the crop group at a point |
| Model | Random forest (scikit-learn 1.9.1), inverse-frequency class weights |
| Training data | 7,252 LUCAS 2018 farmland points with every predictor, in 27 countries: 26 EU member states (all but Malta) and the UK |
| Score on the sealed test set | **Macro F1 0.433** (95% interval 0.399 to 0.459), 1,800 points |
| Cross-validated estimate | Macro F1 0.437 (kNNDM folds, which here are random folds) |
| Chance level | About 0.12 for random guessing (stratified dummy 0.123); 0.216 for a model that knows only the country |
| Model file | `data/processed/final_model.joblib`, not published (see Licence) |

## Intended use

- **Primary.** Estimating the crop group at unsampled farmland points inside the
  27 countries sampled, at the typical spacing of the LUCAS grid (median distance
  to the nearest training point about 10 km), and studying what soil
  measurements add to climate and location. It was built as a research and
  portfolio study.
- **Expect many errors at single points.** For a typical crop group, fewer
  than half of its points are recognised (balanced accuracy 0.45).
- **Not for regional shares as they stand.** The balanced class weights shift
  predicted counts: on the test set olive groves are predicted 35% more often
  than they occur, vineyards and orchards 27% and maize 26%, and wheat 15% less
  often. Shares would need the class priors corrected first; this has not been
  done or validated.

## Out of scope

- **Field-level advice.** It predicts what is grown, not what should be grown or
  what would yield well. Soil nutrients partly reflect past management, so the
  model learns associations in both directions.
- **A new country.** Held out one country at a time, macro F1 falls to 0.310
  (hypothesis H3b, a loss of 0.126). Use outside the 27 countries, or in a
  country absent from training, is not supported.
- **Other years.** Trained on the 2018 crop only. Rotations mean the arable crop
  at a point changes from year to year; no other year has been tested.
- **Points without laboratory soil data.** All 19 predictors are required. The
  model has no missing-value handling; 9% of points lack clay and silt.
- **Probabilities taken at face value for permanent grassland.** See
  Calibration.

## Inputs

19 predictors in four blocks; the country itself is not an input (adding it
changes macro F1 by +0.001, hypothesis H3a).

| Block | Predictors | Source |
|---|---|---|
| Geography and terrain (G) | Easting, northing (EPSG:3035), elevation, slope | LUCAS coordinates; Copernicus DEM GLO-90 |
| Climate (C) | BIO1, BIO4, BIO6, BIO12, BIO15, BIO17, 1991-2020 normals | TerraClimate |
| Inherent soil (S_inh) | pH (CaCl₂), carbonate (log and presence), clay, silt | LUCAS 2018 topsoil; texture from 2009-2015 surveys |
| Nutrient soil (S_nut) | Total N, extractable P and K (log), P below detection | LUCAS 2018 topsoil |

## Output

A probability for each crop group, and the most probable group:
wheat; other cereals; maize and root crops; oilseeds and pulses; fodder and
temporary grassland; permanent grassland; vineyards and orchards; olive groves.

## Training

- **Selection.** Random forest, gradient boosting, multinomial logistic
  regression with spline terms, k-nearest neighbours on coordinates and two
  dummy baselines were compared under nested, domain-matched cross-validation.
  The forest had the best mean macro F1 (0.440) and was kept by the one-SE rule.
- **Predictor set.** Ladder step 5 (G + C + S_inh + S_nut). Adding the country
  (step 6) changed nothing (H3a), so the simpler set was kept.
- **Hyperparameters**: the per-parameter median (or most frequent value) of
  step 5's values tuned in each Stage B outer fold (A17), not tuned again:
  400 trees, maximum depth 20, `max_features` 0.409, minimum 2 points per leaf,
  `class_weight="balanced"`, seed 20261110.
- **Final fit.** Refitted once on all 7,252 complete-case training points; the
  sealed test set was not used for any choice.
- **No post-hoc calibration.** The final model's probabilities are the forest's
  own. Temperature scaling, fitted on inner folds in Stage A as the protocol
  requires, lowered top-label ECE from 0.040 to 0.017 but left permanent
  grassland at 0.060, so it was not applied.

## Performance on the sealed test set

The test set is 1,981 points sealed before any exploration (SHA-256 checked
before the fit) and scored once; 1,800 have every predictor. It was drawn as one
kNNDM fold, so its distances to the training points match those of unsampled
EU farmland. Because that farmland lies as close to the sample as random folds
do, kNNDM chose random assignment: the test set is a random 20% of the points,
and its score describes prediction between sample points, not transfer to new
areas (for that, see Out of scope). Intervals are 95% spatial block-bootstrap
intervals (175 km blocks, 2,000 resamples; resamples lacking a crop group were
dropped, and their number was not recorded). The cross-validated column is the
final configuration under the same kNNDM folds.

| Metric | Test | 95% interval | Cross-validated |
|---|---|---|---|
| Macro F1 (primary) | 0.433 | 0.399 to 0.459 | 0.437 |
| Balanced accuracy | 0.454 | 0.420 to 0.481 | 0.457 |
| Top-2 accuracy | 0.681 | 0.653 to 0.708 | 0.685 |
| Log loss | 1.381 | 1.322 to 1.437 | 1.371 |
| Brier score | 0.651 | 0.623 to 0.676 | 0.643 |

| Crop group | F1, test | 95% interval | F1, cross-validated | Recall, test |
|---|---|---|---|---|
| Permanent grassland | 0.737 | 0.691 to 0.778 | 0.747 | 0.72 |
| Olive groves | 0.615 | 0.456 to 0.727 | 0.605 | 0.72 |
| Vineyards and orchards | 0.581 | 0.459 to 0.671 | 0.447 | 0.66 |
| Other cereals | 0.413 | 0.320 to 0.488 | 0.422 | 0.39 |
| Maize and root crops | 0.367 | 0.297 to 0.427 | 0.406 | 0.42 |
| Wheat | 0.333 | 0.270 to 0.389 | 0.371 | 0.31 |
| Oilseeds and pulses | 0.244 | 0.177 to 0.308 | 0.297 | 0.24 |
| Fodder and temporary grassland | 0.171 | 0.093 to 0.253 | 0.198 | 0.17 |

The model separates permanent crops and grassland well and the arable groups
poorly. Wheat, oilseeds, other cereals and maize are mistaken for one another:
27% of oilseed points are predicted as wheat, and 18% of wheat points as
oilseeds. Fodder and temporary grassland is predicted as maize, permanent
grassland or other cereals about as often as correctly. One season's label on
a rotated field limits how far these can be told apart.

## Calibration

Top-label expected calibration error on the test set is 0.028. Class-wise
errors are 0.014 to 0.043 except for **permanent grassland, 0.064**, above the
protocol's 0.05 threshold. The balanced class weights move probability away from
permanent grassland, the largest class, so its probabilities are too low. The predicted classes, and
every decision of the study, are unaffected; a map of probabilities should
carry this caveat or be recalibrated first.

## Performance by country

Among the 15 countries with at least 30 test points, macro F1 on the test set
ranges from 0.15 (Hungary, 43 points) to 0.46 (Spain, 380); the other 12
countries have too few points to interpret. In 14 of the 15, macro F1 is lower
than over the EU (0.433); Spain is the exception. Part of the model's skill
lies in telling regional crop mixes apart (olive groves and vineyards in the
south, grassland in the Atlantic west), which a single country does not test.
A country's macro F1 averages only the crop groups present among its test
points, which flatters countries with few points and few groups.
[`final_test_by_country.csv`](../reports/tables/final_test_by_country.csv) has every country.

## What drives the predictions

Shuffling a block's values together in the fitted model (grouped permutation
importance, 30 shuffles per fold; macro F1 lost): climate 0.157, geography and
terrain 0.103, nutrient soil 0.083, inherent soil 0.025. Nutrients matter most
for permanent grassland (0.245) and maize (0.116); climate dominates for olive
groves (0.443) and vineyards (0.243). Permutation measures what this model
relies on, not what a block adds: climate and location are largely
interchangeable, and refitted on the ladder, climate adds only 0.015 to
location and terrain while soil adds 0.049 to both. Total N and organic carbon
are almost interchangeable (correlation 0.94), so N largely stands for organic
matter. Rankings of
single predictors are moderately stable across folds (Kendall's τ 0.65).

## Known limitations

- **One season per point.** The arable classes cannot be separated much better
  from soil and climate alone.
- **Associations, not causes.** Nutrient status reflects management.
- **Uneven sampling.** Relative to its share of all 2018 LUCAS farmland points
  (sampled or not), Spain is sampled about 2.2 times and the Netherlands 0.35
  times as densely as the EU average; reweighting changes no decision,
  but the pooled score leans towards densely sampled countries.
- **Errors cluster in space.** Moran's I of the errors is 0.14, so neighbouring
  points tend to be wrong together; this is why intervals use spatial blocks.
- **Complete cases.** Missing texture clusters by country (83% of points in
  Luxembourg, 43% in Croatia, about a third in Ireland and the UK), so these
  countries are under-represented in training and test.
- **No reproducibility log for the final fit.** The run log was added after the
  test set was scored (amendment A20); the final run is tied to commit
  `cec5a48` (code and configuration) and `7c15b48` (results). The sensitivity
  tables were regenerated afterwards, under `7e664ac` and committed in
  `338ab72`, adding one not-run row; the negative-control baseline had been
  committed in `cec5a48`, before the test. No decision changed (A21).

## Licence and access

Code: MIT. The model was trained on LUCAS soil data, which ESDAC licenses
without redistribution; a forest stores split thresholds taken from training
values, so the fitted model is not published. Anyone with ESDAC access can
rebuild it with `python -m eusoil final-test` after the earlier phases.
