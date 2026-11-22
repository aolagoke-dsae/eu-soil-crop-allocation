# Validation report: what laboratory soil data add to crop-allocation models for EU farmland

Dr. Adewole Olagoke · 22 November 2026

Reported along the TRIPOD+AI checklist (Collins et al., 2024) for the
development and validation of a prediction model, adapted to points rather
than patients. Methods were fixed in a dated [protocol](PROTOCOL.md) before
any model was fitted; amendments A1-A20 record every later change. The final
model is described in the [model card](MODEL_CARD.md) and the data in the
[data card](DATA_CARD.md). Every number below is in
[`reports/tables/`](../reports/tables/).

## Summary

**Question.** Which crop group is grown at a point of EU farmland, and how much
do laboratory soil measurements improve that prediction beyond free climate and
location data?

**Data.** 9,906 LUCAS 2018 topsoil points on farmland in 27 countries, each with
its crop recorded in the field, grouped into eight classes.

**Methods.** Six model families compared under nested, spatially
domain-matched cross-validation (kNNDM). Four pre-registered hypotheses tested
on a predictor ladder with paired spatial block bootstrap, 98.75% intervals,
Holm adjustment and a smallest effect of interest of 0.02 macro F1. One score on
a sealed test set of 1,981 points.

**Results.** A random forest on location, terrain, climate and soil reaches
macro F1 **0.433 (95% interval 0.399 to 0.459)** on the sealed test set,
against 0.437 in cross-validation. Soil adds 0.049 macro F1 beyond climate and
location (H1, supported). Nutrient measurements add more than inherent soil
properties, the reverse of the prediction (H2), though this depends on how
organic matter is assigned. Knowing the country adds nothing (H3a). Accuracy
falls by 0.126 in a country the model has not seen (H3b).

**Conclusion.** Soil data improve EU-scale crop prediction by a modest,
consistent amount; the improvement comes mostly through nitrogen, which tracks
organic matter. The model is useful for regional shares of crop groups within
the sampled countries, not for single fields or new countries.

## Introduction

Climate and location data are free and complete; laboratory soil data are
costly and sparse. Whether the soil measurements of the LUCAS survey add to the
prediction of land use, and which ones, decides whether they are worth
collecting for that purpose. The study tests four hypotheses
([protocol](PROTOCOL.md), amendment A13):

- **H1 (primary).** Laboratory soil measurements add to predictive performance
  beyond geography and climate.
- **H2.** Inherent soil properties (pH, carbonate, texture) add more than
  nutrient measurements (N, P, K), each added on top of geography and climate.
- **H3a.** The country adds predictive performance beyond geography, climate
  and soil.
- **H3b.** A model trained on other countries predicts an unseen country less
  accurately than unsampled farmland within the surveyed countries.

Each is decided against a smallest effect of interest of 0.02 macro F1.

Estimate **E1** describes how macro F1 changes with distance to the nearest
training point (A11).

## Methods

### Data and outcome

- **Source.** LUCAS 2018 topsoil (0-20 cm) with the harmonised land-cover table;
  texture from the 2009-2015 surveys at the same points (A4).
- **Eligibility.** Standard sampling depth, cropland or grassland, agricultural
  land use, a land-cover code in one of the eight groups; Malta excluded
  (2 points). See the [data card](DATA_CARD.md) for the full sample flow.
- **Outcome.** The crop group observed in the field in 2018: wheat, other
  cereals, maize and root crops, oilseeds and pulses, fodder and temporary
  grassland, permanent grassland, vineyards and orchards, olive groves.
- **Predictors.** Four blocks: geography and terrain (G, 4 variables), climate
  1991-2020 (C, 6), inherent soil (S_inh, 5), nutrient soil (S_nut, 4); the
  country (K) only in ladder step 6.
- **Missing data.** Complete cases; 8.6% of points lack clay and silt. Values
  below the detection limit are set to half the limit and flagged (A1).

### Sample size

Before any model was fitted, a precision check simulated the bootstrap
interval for a macro F1 score at this sample size with 50 and 100 km blocks:
its median width, about 0.006, is far inside the protocol's limit of 0.04 ([`precision_check.csv`](../reports/tables/precision_check.csv)).

### Splits

- **Sealed test set.** One of five kNNDM folds (1,981 points, 20%), chosen so
  that its distances to the training points match those from unsampled EU
  farmland to the sample (median 10.2 km). Its point IDs were hashed (SHA-256)
  and committed before exploration; the hash was checked before the final fit.
- **Cross-validation.** Five-fold kNNDM, repeated three times, on the 7,925
  training points; four inner folds for tuning.
- **Sensitivity schemes.** Random folds, square blocks of 25, 50, 100 and
  200 km, and leave-one-country-out.

### Models

Multinomial logistic regression with natural splines, random forest, gradient
boosting (histogram), k-nearest neighbours on coordinates, and majority and
stratified dummies. Tuned models had the same budget (60 random-search trials,
log loss). Classes were weighted by inverse frequency. The simplest model
within one standard error of the best mean macro F1 was selected.

### Hypothesis tests

The predictor ladder adds one block at a time (G; + C; + S_inh or S_nut;
both; + K). Each hypothesis compares two ladder steps fitted on identical
folds. Differences are tested with a paired spatial block bootstrap (175 km
blocks, set from the variogram range; 2,000 resamples). A hypothesis is
**supported**, **negligible**, **inconclusive** or **reversed** by where its
98.75% interval lies relative to ±0.02; p-values are Holm-adjusted over the
four tests.

### Performance measures

Macro F1 (primary), balanced accuracy, log loss, Brier score, top-2 accuracy,
per-class F1, top-label and class-wise expected calibration error (15 bins),
macro F1 by country, and Moran's I of the errors.

### Robustness

61 sensitivity runs were registered (59 informative for H1, H2 and H3a; 26
for H3b): label purity and in-situ observation only; three other treatments of
values below the detection limit; pH in water; another climate period and two
WorldClim resolutions; organic carbon in either soil block, and organic soils
excluded; each country reweighted to its share of farmland points; block sizes
and climate-cluster folds; other class-size thresholds and the coarser Level-1
classes; Germany only, the five largest countries, and each country left out;
no class weights; other algorithms; five other seeds; thinning to 10 km. A
hypothesis is **robust** if no run changes its decision, **mostly robust** if
one does, and **fragile** otherwise (A18). Two negative controls: labels
permuted within country (should fall to the country-only baseline, A19), and an
added noise predictor (should carry no importance).

## Results

### Participants and flow

| Set | Points | With every predictor |
|---|---|---|
| Eligible | 9,906 | |
| Training | 7,925 | 7,252 (673 dropped) |
| Sealed test | 1,981 | 1,800 (181 dropped) |

The test set holds every crop group and every country; its class mix matches
the training set within 1.5 percentage points for every group
([`split_seal_by_crop_group.csv`](../reports/tables/split_seal_by_crop_group.csv)).

### Model selection

| Model | Macro F1 (cross-validated) |
|---|---|
| Random forest | **0.440** |
| Gradient boosting | 0.433 |
| Logistic regression with splines | 0.406 |
| k-nearest neighbours on coordinates | 0.371 |
| Stratified dummy | 0.123 |
| Majority dummy | 0.054 |

The forest was best by mean and selected (one-SE threshold 0.437).

### Hypotheses

| Ladder step | Blocks | Macro F1 | 95% interval |
|---|---|---|---|
| 1 | G | 0.372 | 0.346 to 0.390 |
| 2 | G + C | 0.388 | 0.363 to 0.406 |
| 3 | G + C + S_inh | 0.403 | 0.377 to 0.422 |
| 4 | G + C + S_nut | 0.427 | 0.402 to 0.445 |
| 5 | G + C + S_inh + S_nut | **0.437** | 0.411 to 0.455 |
| 6 | Step 5 + country | 0.437 | 0.412 to 0.456 |

| | Comparison | Difference | 98.75% interval | Decision | Robustness |
|---|---|---|---|---|---|
| H1 | Step 5 − step 2 | +0.049 | 0.037 to 0.061 | Supported | Robust (0 of 59 runs disagree) |
| H2 | Step 3 − step 4 | −0.023 | −0.035 to −0.013 | Reversed | Fragile (7 of 59) |
| H3a | Step 6 − step 5 | +0.001 | −0.004 to 0.005 | Negligible | Mostly robust (1 of 59) |
| H3b | kNNDM − leave-one-country-out | +0.126 | 0.096 to 0.152 | Supported | Robust (0 of 26) |

H2's reversal rests on total N: N and organic carbon correlate at 0.94, and
when organic carbon is counted as inherent soil the difference becomes
negligible (−0.004). The protocol required H2's conclusion to hold with organic
carbon in either block; it does not. Six other runs make H2 inconclusive by widening its
interval (pure labels only, 25 and 50 km blocks, Germany only, without Ireland,
pH in water); none reverses the sign. The fair reading is that nutrient
measurements predict crops at least as well as inherent properties, mainly
through organic matter.

### Distance (E1)

| Validation | Median distance to training | Macro F1 |
|---|---|---|
| Random folds | 10.2 km | 0.441 |
| kNNDM (matched to the prediction domain) | 10.2 km | 0.437 |
| 25 km blocks | 15.4 km | 0.426 |
| 50 km blocks | 20.6 km | 0.423 |
| 100 km blocks | 30.5 km | 0.406 |
| 200 km blocks | 49.1 km | 0.389 |
| Leave one country out | 154.9 km | 0.310 |

Macro F1 declines steadily with distance. The pre-registered plateau rule
returns 25 km because the 25-50 km change is small, but it does not describe
the curve; random folds are not biased here because the prediction domain sits
as close to the sample as random folds do.

### Final model on the sealed test set

| Metric | Test | 95% interval | Cross-validated | Test − CV |
|---|---|---|---|---|
| Macro F1 | **0.433** | 0.399 to 0.459 | 0.437 | −0.004 |
| Balanced accuracy | 0.454 | 0.420 to 0.481 | 0.457 | −0.003 |
| Top-2 accuracy | 0.681 | 0.653 to 0.708 | 0.685 | −0.004 |
| Log loss | 1.381 | 1.322 to 1.437 | 1.371 | +0.010 |
| Brier score | 0.651 | 0.623 to 0.676 | 0.643 | +0.009 |

Every cross-validated overall metric lies inside its test interval: the
cross-validation predicted the test macro F1 to within 0.004.
Six of eight groups score slightly lower on the test set than in
cross-validation, by 0.01 to 0.05; vineyards and orchards score higher
(0.581 against 0.447) on 109 test points with a wide interval (0.459 to 0.671).
Per-class results, the confusion matrix and results by country are in the
[model card](MODEL_CARD.md) and
[`final_test_*.csv`](../reports/tables/final_test_metrics.csv).

![Sealed test set](../reports/figures/final_test.png)

### Calibration

Top-label ECE 0.028 on the test set (0.040 in cross-validation). Permanent
grassland's class-wise ECE is 0.064, above the 0.05 threshold, as in
cross-validation (0.073): the balanced class weights pull its probabilities
down.

### Interpretation and controls

- **Block importance** (macro F1 lost when a block is removed and the model
  refitted): climate 0.157, geography and terrain 0.103, nutrient soil 0.083,
  inherent soil 0.025.
- **Noise predictor**: 0.002, against 0.025 for the least important block.
- **Labels permuted within country**: 0.226, against 0.216 for a model that
  knows only the country; the excess, 0.009, is below 0.02, so the control
  passes (A19).
- **Spatial errors**: Moran's I 0.14 (p = 0.005); errors cluster, which the
  block bootstrap accounts for.

## Discussion

### Interpretation

Soil measurements add a modest but robust 0.05 macro F1, about a third of what
climate contributes. Most of the gain comes from nitrogen, which here stands
for organic matter: permanent grassland stands on soils richer in organic
matter than arable land, and the model uses that. Because organic matter
reflects how land has been used, part of this is the crop leaving its mark on
the soil rather than the soil deciding the crop.

The country adds nothing once soil and climate are in the model, but a model
trained without a country loses 0.126 in it: the crop mix of a country is
learnable from its neighbours' soil and climate only in part.

### Limitations

- **One season per point.** Arable rotations cap how well the arable groups can
  be separated; their F1 is 0.17 to 0.41.
- **Observational associations.** No causal reading of soil effects.
- **Texture from earlier surveys**, measured 2009-2015 rather than 2018.
- **Complete cases.** 9% of points are dropped, mostly for missing texture.
- **Fragile H2.** Its decision depends on organic matter's assignment and on the
  validation scheme.
- **Post-hoc chance level.** Negative control (a)'s baseline was defined after
  its result was seen (A19).
- **Protocol practices not built at the time** (run log, an end-to-end
  synthetic run, an EM prior-shift sensitivity check), listed in A20.

### Usability

For regional shares of crop groups in the 27 countries, at LUCAS-grid spacing.
Not for single fields, years other than 2018, or countries not sampled. Any
probability map should note permanent grassland's under-confidence.

## Other information

- **Registration.** The protocol was committed to this repository on
  10 November 2026 before any model was fitted; amendments are dated in it.
- **Code and data.** Code: MIT licence, this repository. LUCAS soil data:
  ESDAC licence, not redistributed; only aggregates are published.
- **Funding and interests.** None; an independent portfolio study.
- **AI assistance.** The code and the first drafts of the documents were written
  by an AI coding agent (Claude, Anthropic) under the author's direction; see
  the README section "How this was built".

## References

- Collins, G. S. et al. (2024). TRIPOD+AI statement: updated guidance for
  reporting clinical prediction models that use regression or machine learning
  methods. *BMJ*, 385, e078378.
- Linnenbrink, J., Milà, C., Ludwig, M. & Meyer, H. (2024). kNNDM CV: k-fold
  nearest-neighbour distance matching cross-validation for map accuracy
  estimation. *Geoscientific Model Development*, 17, 5897-5912.
- Orgiazzi, A., Ballabio, C., Panagos, P., Jones, A. & Fernández-Ugalde, O.
  (2018). LUCAS Soil, the largest expandable soil dataset for Europe: a review.
  *European Journal of Soil Science*, 69(1), 140-153.
