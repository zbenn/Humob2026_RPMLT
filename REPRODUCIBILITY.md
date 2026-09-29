# Reproducibility and scope

## Included

- The exact fixed 50/50 local-level / guarded temporal model used by the final
  submission-generation code.
- Standalone input parsing, fold-local fitting, official scoring denominators,
  ten diagnostic windows and eight earlier windows.
- Main-table local, temporal, combined and boundary-interpolation comparisons.
- Full-area submission construction with April outside-box means and 58 dates.
- Public feature snapshots, synthetic tests, original-figure and spatial-figure
  generation scripts, and aggregate benchmark checks.

## Not included or claimed

This is a focused release, not an archive of every development experiment.
Exploratory LLM prompting, weather/housing/fine-grained-water ablations, learned
ensemble weights and unrelated model searches are not packaged in this version.
Their negative findings are discussed in the paper, but `evaluate` reproduces
the main comparison table only. There are no hidden-target labels or claimed
official ranking results here. No model tuning is performed on the figure's
inspected cells. The spatial plot is a descriptive development figure, not an
additional evaluation experiment.

## Numerical conventions retained

1. The sparse pair matrix is stored in float32 to match the original loader;
   fitting uses the original NumPy/SciPy numerical operations. Fold-unseen
   columns are forced to zero, even though the union of column IDs is fixed.
2. Local means use the nearest 30 available observations, not calendar days.
   Fewer than 14 post-interval observations trigger the pre-interval fallback.
3. The local weekday adjustment centers each training month separately and
   uses ordinary weekdays. Holidays-as-Sundays applies to the temporal model.
4. The water table is held at its first/last available date outside its range;
   unknown municipalities map to zero. This reproduces the original implementation
   and is not an assertion about pre-earthquake observed outages. The coefficient
   0.15 is fixed, not fitted to unobserved target outcomes.
5. Temporal features are standardized using training rows. Columns with standard
   deviation at most 1e-8 are discarded. Target values are clipped to the
   standardized training range. The intercept is not penalized.
6. Robust fitting performs at most 25 iteratively reweighted solves, with
   diagonal/off-diagonal penalties of 10/100, fixed residual scale from the
   initial ridge fit and the original convergence tolerance.
7. Scores average daily norms, and category summaries weight windows equally.
   Different categories and the earlier protocol overlap. They must not be
   interpreted as independent replicated evidence.
8. Submission serialization retains full floating-point precision. Slight BLAS
   differences can change final decimal digits; byte-identical TSV files are not
   promised across environments. Numerical equivalence is the meaningful check.

## Checks

Synthetic checks run with `python -m pytest -q`. Full-data checks require an
authorized local dataset:

```bash
python -m rpmlt.cli evaluate --data data/humob2026-dataset.tsv
python scripts/check_benchmarks.py
python -m rpmlt.cli predict --data data/humob2026-dataset.tsv
```

The benchmark checker compares all 72 model/window results and both metric
components against archived aggregate outputs (absolute tolerance 1e-8,
relative tolerance 1e-7). The original official validator used during development
was at commit `2549643345163a0f61801fb837b93ca070d05b0a` of
https://github.com/takayabe0505/humob2026-validator.
For an actual submission, also check the organizers' latest validator.

The code was extracted and refactored from the authors' project implementation.
It does not depend on the private workspace, its directory names, manuscript
repository, local cached arrays or credentials. Downloaded reference geography
and any restricted-data-derived output are runtime artifacts, not release files.

## Release verification

The standalone release was checked locally against the original project:

- 17 synthetic tests passed without restricted inputs.
- All 72 main-table model/window comparisons and their diagonal/off-diagonal
  components matched the archived results within the stated tolerance.
- The regenerated submission retained exactly the same 58 dates and 645,705 OD
  keys as the original export. Maximum absolute value difference was
  7.9581e-13 (floating-point roundoff).
- The original official validator returned `Validation passed!`; the independent
  local validator and exact scoring-box serialization checks also passed.

This verifies implementation equivalence, not accuracy on undisclosed target
labels. Neither the original nor regenerated submission is distributed here.
