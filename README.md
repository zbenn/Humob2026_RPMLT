# Humob2026_RPMLT

Code for **Reconstructing Post-Disaster Mobility with Local Activity Levels and
Temporal Patterns**, a HuMob Challenge 2026 workshop paper.

The task is retrospective reconstruction of February-March 2024 grid-level OD
activity on the Noto Peninsula. Observations before **and after** the missing
interval are available. This is not future-only forecasting.

## Method

The final estimate equally averages:

1. **Local activity levels:** nearby observed levels, month-centered weekday
   corrections and a conservative municipal water-outage adjustment.
2. **Temporal patterns:** robust ridge regression on calendar, earthquake/recovery
   and festival features, with target features clipped to their training ranges.

Diagonal and off-diagonal flows have separate ridge penalties. Pairs never
observed in the current training fold receive zero predictions. All dates,
preprocessing and fitting are recomputed within each validation fold.

The final model does **not** use an LLM or time-series foundation model. No API
key, paid service, GPU or network access is needed for fitting or inference.

## Install

Python 3.10 or later:

```bash
git clone https://github.com/zbenn/Humob2026_RPMLT.git
cd Humob2026_RPMLT
python -m pip install -e ".[dev]"
python -m pytest -q
```

The tests use synthetic inputs and included public auxiliary features. They do
not download or require the restricted competition dataset.

The reference scores were reproduced with NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1
and Matplotlib 3.8.4. `requirements-repro.txt` records this environment. Other
supported versions can have small floating-point differences.

## Obtain the competition data

Apply for access through the [official dataset record](https://zenodo.org/records/20709796)
and follow the [challenge instructions](https://takayabe0505.github.io/humob-2026/).
Place your authorized copy at `data/humob2026-dataset.tsv`, or pass its absolute
path with `--data`. The data are not bundled and are not downloaded automatically.

The release contains normalized, anonymized OD values, **not raw person counts**.
Dates marked `NA` are excluded; absent OD entries on valid dates are zero in the
released representation. The official `-1_-1` sentinel is supported.

Do not commit the TSV, parsed caches, individual predictions or generated
data-bearing outputs. They are excluded by `.gitignore`. No additional mobility
dataset is used. The required public calendar and water-service feature snapshots
are included with provenance in [DATA_SOURCES.md](DATA_SOURCES.md).

## Reproduce the main results

```bash
python -m rpmlt.cli evaluate --data data/humob2026-dataset.tsv --output outputs/evaluation
python scripts/check_benchmarks.py --scores outputs/evaluation/scores.csv
```

This runs the four main-table models on the ten diagnostic and eight earlier
validation windows. Outputs include per-window scores, daily scores, grouped
scores and the selected-cell errors needed by the paper figure. Use
`--protocol diagnostic` or `--protocol legacy` for a subset.

Reference combined normalized RMSE (lower is better):

| Model | Recovery | Gap | Event | Stress | Season | Primary | Rolling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local-level | 0.2089 | 0.2087 | 0.2153 | 0.2252 | 0.2668 | 0.2240 | 0.2091 |
| Temporal | 0.2091 | 0.2082 | 0.2144 | 0.2264 | 0.2063 | 0.2221 | 0.2081 |
| Combined | 0.2043 | 0.2068 | 0.2125 | 0.2197 | 0.2268 | 0.2167 | 0.2066 |
| Boundary interpolation | 0.2051 | 0.2067 | 0.2126 | 0.2280 | 0.2274 | 0.2251 | 0.2062 |

The metric averages daily diagonal/off-diagonal normalized RMSEs. Its denominators
cover all 1,476 diagonal and 1,476 x 1,475 off-diagonal pairs, including implicit
zeros. It is not RMSE pooled over all days or calculated over active pairs only.

**Interpretation:** these windows overlap and were reused during development.
They are robustness checks, not independent test sets or hidden-target scores.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for release scope and conventions.

## Generate a submission

```bash
python -m rpmlt.cli predict --data data/humob2026-dataset.tsv --output outputs/submission.tsv
python -m rpmlt.cli validate outputs/submission.tsv
```

The file contains exactly 58 valid target dates, excluding February 2 and March 5.
Each row contains `YYYYMMDD`, a tab and a Python nested OD dictionary. Values remain
floating point; do not round to integers or normalize the daily output again.
The learned ensemble covers the scoring box. Outside-box pairs use the observed
April mean, matching the original full-area export. This includes any released
unknown-location sentinel edges.

Local validation checks dates, IDs, finite nonnegative values and exact
serialization of scoring-box predictions. **It does not replace the official
validator.** Obtain that script from the
[organizers' repository](https://github.com/takayabe0505/humob2026-validator).
For example, clone it outside this repository:

```bash
git clone https://github.com/takayabe0505/humob2026-validator.git ../humob2026-validator
python ../humob2026-validator/humob2026_validator.py outputs/submission.tsv
```

Expect `Validation passed!`. Alternatively pass the script to `predict` using
`--validator ../humob2026-validator/humob2026_validator.py`. Existing submission
files are not overwritten. Nothing is uploaded to Box or the challenge server.

## Reproduce figures

After installing the package and completing evaluation:

```bash
python scripts/plot_paper.py --data data/humob2026-dataset.tsv --evaluation outputs/evaluation
python scripts/plot_spatial.py --data data/humob2026-dataset.tsv --output outputs/spatial
```

The first command creates the mobility/water-context and model-error figures.
The second creates the three-panel spatial distribution/change figure added
during paper development. It downloads public MLIT boundary geometry on its
first run (no additional mobility data). Plots use local observations, not hidden
target labels. The spatial map does not interpolate February-March.
Generated PDF/PNG/SVG files and numerical plotting intermediates remain local
under `outputs/`. Paper prose, Overleaf files and manuscript edits are not part
of these commands.

## Repository layout

```text
rpmlt/
  data.py          # authorized TSV input and official metric
  features.py      # public calendar and municipal water features
  models.py        # final components and boundary comparator
  evaluation.py    # exact validation windows
  submission.py    # TSV export and local validation
  cli.py           # evaluate / predict / validate
  assets/          # public auxiliary features only
scripts/           # benchmark checking and figure generation
tests/             # synthetic, offline tests
benchmarks/        # aggregate reference scores, not OD observations
```

## Citation and license

Please cite the workshop paper and comply with the competition dataset's
attribution requirements. The dataset documentation also requests citation of
Yabe et al., *YJMob100K: City-scale and longitudinal dataset of anonymized human
mobility trajectories*, Scientific Data 11, 397 (2024),
[doi:10.1038/s41597-024-03237-9](https://doi.org/10.1038/s41597-024-03237-9).
No publication DOI is invented for the workshop paper.

Source code is released under the [MIT License](LICENSE). That license does not
relicense the competition dataset, third-party boundary geometry or government
source documents. Auxiliary data provenance and attribution are documented
separately. This repository is a curated release from the project code, not the
private development history.
