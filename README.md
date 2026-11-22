# Soil, Climate and Crop Allocation in European Agriculture

**Which crop is grown where in European agriculture, and what do laboratory soil
measurements add to the predictive performance of a crop-allocation model, beyond
free climate and geographic information?**

A pre-registered predictive study on 9,906 agricultural points of the EU's LUCAS
2018 topsoil survey: the crop group recorded in the field, laboratory-measured
soil chemistry and texture, and climate and terrain at each point.

> **Status.** Phases 0 to 6 of the [analysis protocol](docs/PROTOCOL.md) are
> run: data audit, sealed split, exploratory analysis, model selection,
> hypothesis tests, interpretation, 59 sensitivity runs and the single score on
> the sealed test set. On its 1,800 points with complete predictors the final
> model reaches macro F1 0.433 (95% interval 0.399 to 0.459), against 0.437 in
> cross-validation
> ([`final_test_metrics.csv`](reports/tables/final_test_metrics.csv)). The
> results below are cross-validated on the training points. The
> [model card](docs/MODEL_CARD.md) and the
> [validation report](docs/VALIDATION_REPORT.md), written along TRIPOD+AI, give
> the test results in full.

## Key findings

| | Question | Estimate (macro F1) | Decision |
|---|---|---|---|
| **H1** | Do laboratory soil data add to climate and location? | **+0.049** (98.75% interval 0.037 to 0.061) | Supported; robust in all 59 sensitivity runs |
| **H2** | Do inherent soil properties add more than nutrients? | **−0.023** (−0.035 to −0.013) | Reversed on the primary analysis: nutrients add more. Not established: the protocol requires the conclusion to hold with organic carbon in either soil block, and with it in the inherent block the gap is negligible |
| **H3a** | Does country add anything beyond climate, location and soil? | +0.001 (−0.004 to 0.005) | Negligible |
| **H3b** | Does the model transfer to a country it has not seen? | **+0.126** (0.096 to 0.152; within-country minus unseen-country macro F1) | Supported: transfer is much harder than mapping within the surveyed countries |

- **Soil data help, modestly.** Macro F1 rises from 0.388 (climate and location)
  to 0.437 (plus soil), against 0.371 for k-nearest neighbours on coordinates
  alone and 0.054 for always predicting the largest class. The gain is about 2.4
  times the smallest effect of interest set in advance (0.02). Holm-adjusted
  p-values: 0.004 for H1, H2 and H3b, 0.74 for H3a.
- **Total N, which tracks organic carbon, is the soil variable the fitted
  model relies on most** (exploratory; permutation drop 0.056, ahead of K and
  P), and it moves almost one-for-one with organic carbon (Spearman
  ρ = 0.94; median organic carbon over median N is 8.6 to 10 in every crop group). The
  fitted model relies on the nutrient block most for permanent grassland. When organic carbon is added to the
  inherent block, the gap between the blocks disappears (−0.004, negligible).
  The protocol's pre-registered reading of a reversed H2 is that it "would point
  to management, not soil type, as the soil signal"; organic matter reflects
  both, so the data cannot separate them.
- **The fitted model leans most on climate**, especially for olive groves and
  vineyards, then location, then nutrient soil, then inherent soil (macro F1
  lost when a block is shuffled: 0.157, 0.103, 0.083, 0.025). Climate and
  location are largely interchangeable, though: added to location and terrain,
  climate raises macro F1 by only 0.015, less than a third of what soil adds on
  top of both (0.049).
- **Performance depends on distance to the training data.** Mapping EU farmland
  between the sample points (median 10 km to the nearest sample) gives macro F1
  0.437; with 200 km blocks held out it falls to 0.389, and for an unseen
  country to 0.310. Spain loses most (0.45 to 0.20), although the other countries
  cover 97.5% of its climate and soil conditions, so what fails to transfer may
  be country-specific practice rather than environment.
- **Eight crop groups are hard to separate.** Permanent grassland (F1 0.75) and
  olive groves (0.61) are recognised well; fodder (0.20) and oilseeds and pulses
  (0.30) are not (cross-validated, final configuration). Wheat is confused
  with oilseeds and maize, olive groves with vineyards, and errors cluster in
  space (Moran's I 0.14).

![Predictor-block ladder and hypothesis decisions](reports/figures/stage_b_ladder.png)

## Approach

**Data.** LUCAS 2018 topsoil samples (0-20 cm) on cropland or grassland under
agricultural use, joined to the harmonised LUCAS land-cover record and to
climate, terrain and texture at each point. The sample flow, from 18,984 soil
samples to 9,906 points, is in
[`reports/tables/sample_flow.csv`](reports/tables/sample_flow.csv).

**Target.** Eight crop groups, each with at least 300 points: wheat (1,768),
other cereals (1,551), maize and root crops (1,094), oilseeds and pulses (973),
fodder and temporary grassland (708), permanent grassland (2,802), vineyards and
orchards (567) and olive groves (443).

**Predictor blocks.**

| Block | Variables |
|---|---|
| G, geography and terrain | Projected coordinates, elevation and slope (Copernicus DEM GLO-90) |
| C, climate | Six bioclimatic variables from TerraClimate 1991-2020 normals |
| S-inh, inherent soil | pH (CaCl₂), carbonate, clay and silt |
| S-nut, nutrient soil | Total N, extractable P and K, a P-below-detection flag |
| K, country | Country |

**Validation designed for the use case.** The model is meant to map unsampled
farmland inside the EU. Folds are built with k-fold nearest-neighbour distance
matching (kNNDM), so the distances between held-out and training points match
those between EU farmland and the sample. Because unsampled farmland lies as
close to the sample as random folds do (median 8.5 km against 8.9 km), kNNDM
chose one point per cluster: the folds, and the sealed test set, are in effect
random 20% splits. Spatial transfer is measured separately, with blocks of up to
200 km and with each country held out (E1, H3b). A sealed test set of 1,981
points was fixed by kNNDM before any exploration; the SHA-256 hash of its point IDs is in
[`reports/tables/split_seal.csv`](reports/tables/split_seal.csv), committed
before the exploratory analysis.

**Inference.** Every comparison is paired on the same points and folds, with
uncertainty from a spatial block bootstrap (175 km blocks, set from the class
variograms). Decisions use 98.75% intervals (95% adjusted for four tests)
against a smallest effect of interest of 0.02 macro F1, and Holm-adjusted
p-values.

**Model selection.** Random forest, chosen by the one-standard-error rule over
logistic regression on spline bases and gradient boosting, with equal tuning
budgets in nested cross-validation (macro F1 0.440, against 0.433 for boosting).

**Pre-registration.** The protocol, with its hypotheses, estimands and decision
rules, was committed before modelling. Every later change is a dated amendment
(A1-A21) in the protocol, including the ones made after a pipeline check.

## Robustness

The decisions were repeated in 59 sensitivity runs, beside a reference run
under the same fixed settings: detection-limit handling, target definition,
block sizes instead of kNNDM, sample restrictions, each country left out in
turn, climate source and resolution, other algorithms, model seeds and survey
reweighting. Two negative controls were run beside them. Four registered runs
are not reported: buffered leave-one-out and focal loss by a decision made
before Phase 5 (A18), class-prior re-estimation and missing-value imputation
because they were never built (A20, A21). The five seed runs cover the ladder
only, not H3b. Results:
[`sensitivity_summary.csv`](reports/tables/sensitivity_summary.csv); runs not
made: [`sensitivity_not_run.csv`](reports/tables/sensitivity_not_run.csv).

| | Runs disagreeing with the primary decision | Verdict |
|---|---|---|
| H1 | 0 of 59 | Robust |
| H2 | 7 of 59 (6 inconclusive at the threshold or in Germany alone; 1 negligible: organic carbon in the inherent block) | Fragile on the pre-set rule; the sign holds in 59 of 59 |
| H3a | 1 of 59 (Germany only) | Mostly robust |
| H3b | 0 of 26 | Robust |

The negative controls behave as they should. A pure-noise variable has an
importance of 0.002. With labels shuffled within each country, the model scores
0.226, the level of a model that knows only each country's crop mix (0.216), and
soil adds nothing (H1 +0.004).

![Sensitivity runs](reports/figures/sensitivity_forest.png)

## Limitations

- **One year, one crop.** Each point records the crop of 2018. Rotations put
  different crops on the same soil, which caps how well arable groups can be told
  apart.
- **Associations, not causes.** Soil nutrients and organic matter reflect past
  management as well as soil type, so the results say what predicts crop
  allocation, not what drives farmers' choices.
- **Texture from earlier surveys.** Clay and silt come from the 2009-2015
  surveys at the same point, measured for 91% of the points. The rest are
  dropped as complete cases, and the gaps cluster: texture is missing for 83%
  of the points in Luxembourg, 43% in Croatia and about 30% in Ireland and the
  UK, so these countries are under-represented.
- **Calibration and shares.** Class weighting shifts probability away from
  permanent grassland, the largest class (class-wise calibration error 0.064 on
  the sealed test set), and biases predicted shares: olive groves are predicted
  35% more often than they occur, wheat 15% less often.
  Decisions use predicted classes and are not affected; any map of probabilities
  should carry this caveat.
- **The E1 plateau rule.** The pre-registered rule returns 25 km because macro F1
  barely changes from 25 to 50 km blocks, but it falls again by 0.017 at each
  doubling beyond 50 km. Performance keeps declining with distance up to 200 km.

## Repository layout

```
config/analysis.yaml   every fixed choice: filters, blocks, splits, budgets, seeds
docs/PROTOCOL.md       the pre-registered protocol and its amendments
docs/DATA_CARD.md      sources, sample flow, missing values and known biases
docs/MODEL_CARD.md     the final model: intended use, test scores, calibration, limits
docs/VALIDATION_REPORT.md  the full study along TRIPOD+AI, built around the test score
src/eusoil/            the pipeline (python -m eusoil <command>)
tests/                 unit tests on synthetic data; run in CI on Linux and Windows
reports/tables/        aggregated results (no point-level soil data)
reports/figures/       figures (soil maps as medians of 50 km cells; distributions within the 1st-99th percentile)
reports/run_log.csv    one row per command: commit, config hash, seed, package versions
data/README.md         how to obtain each source; data/ itself is not committed
```

## Reproducing the pipeline

Requires Python 3.12 or later (developed on 3.14). The LUCAS soil data must be
requested from ESDAC; [`data/README.md`](data/README.md) lists every source and
where to put it.

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1   macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest            # unit tests on synthetic data; no licensed data needed
```

Then run the phases in order (`python -m eusoil <command>`). Long steps save
each finished fold and resume if interrupted.

| Phase | Commands |
|---|---|
| 1. Data | `points`, `dem`, `terraclimate --period 1991-2020`, `terraclimate --period 2015-2018`, `worldclim` (optional internal check), `table`, `precision` |
| 2. Split and exploration | `domain`, `seal` (refuses to re-seal), `variogram`, `folds`, `eda` |
| 3. Model selection | `stage-a`, `stage-a-report` |
| 4. Hypothesis tests | `stage-b-ladder`, `stage-b-validation`, `stage-b-report` |
| 5. Interpretation and robustness | `interpret`, `sensitivity`, `sensitivity-report` |
| 6. Final evaluation | `final-test` (scores the sealed test set once; refuses a second run) |

## How this was built

The code was written by an AI coding agent, Claude (Anthropic), under my
direction. It drafted the pipeline (about 6,800 lines in `src/eusoil/`), the
tests (about 1,900 lines) and the configuration, and helped draft the protocol,
this README and the data card. I set the questions, the hypotheses and the
validation design; decided every protocol choice and amendment; reviewed the
code before each commit; ran every command on the real data on my own machine;
and made every commit.

The agent's drafts were not trusted to be right. Four checks sat outside them:

- **The protocol came first.** It was committed before any model was fitted, and
  every later change is a dated amendment with its reason (A1-A21). My review of
  the first draft produced A11-A14: it had counted the confirmatory tests wrongly
  (five instead of seven), framed random-versus-spatial validation as a hypothesis that
  assumed its own answer, and given H2 no decision threshold.
- **Real-data runs failed loudly.** Running each phase on the LUCAS data caught a
  crash and excessive memory use in the TerraClimate extraction and elevation
  errors at the edges of DEM tiles, none of which the synthetic unit tests reach.
- **Continuous integration** on Linux and Windows caught a formatting regression
  from a late edit; the linter is now pinned to one version.
- **Results were read, not accepted.** A README sentence that overstated H2 was
  rewritten before release, and the permuted-label control, which looked like a
  failure, was resolved only by defining its chance level, recorded as defined
  after the result was seen (A19).

Smaller errors were fixed in review: a stray index column in an audit table and
a wrong configuration key for the decision threshold.

An audit of the protocol against the code on 22 November 2026, just after the
sealed test set was scored, found practices the protocol described but the agent
had not built. A20 lists each one. Two were added: a run log in
[`reports/run_log.csv`](reports/run_log.csv), so runs before that date have no
log row, and a test that no point is on both sides of any split. The rest are
corrected in the record. None of them changes a result.

A read-through of the write-up the same evening, by a separate AI reviewer
given only the repository and asked to act as a sceptical referee, found what
the drafting had missed: H3b reported with the wrong sign, block importance
described as refitting when it is permutation, two comparisons that mixed
different measures, a post-hoc reading of H2 stated as a finding, the fact
that kNNDM had chosen random folds left unsaid, and ALE tables that published
the minimum and maximum of pH, P and K, each a single point's value. A21
records each correction.

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
under COPERNICUS by the European Union and ESA. Boundaries: © EuroGeographics
for the administrative boundaries (GISCO NUTS 2016).

## Author

**Dr. Adewole Olagoke** · [aolagoke-dsae](https://github.com/aolagoke-dsae)

Code licensed under the MIT License (see `LICENSE`).
