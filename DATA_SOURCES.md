# Data sources and attribution

## Restricted competition observations (not included)

- HuMob Challenge 2026: https://takayabe0505.github.io/humob-2026/
- Dataset access: https://zenodo.org/records/20709796
- Dataset-specified related publication: https://doi.org/10.1038/s41597-024-03237-9

Users must obtain their own authorized copy. This repository contains neither
the original TSV nor parsed OD matrices, per-cell observations, generated
submissions or private correspondence. An aggregate evaluation score is not an
additional source of mobility labels for fitting.

## Included public auxiliary snapshots

| File in `rpmlt/assets/` | Content and provenance | Use |
| --- | --- | --- |
| `cells.csv` | Scoring-box `(cell_y, cell_x)` to municipality mapping, derived from MLIT N03 administrative geometry by largest intersecting municipal area | Assign water indices to OD endpoints |
| `water_daily.csv` | Municipal approximate water-outage fractions from January 19 to April 9, 2024; fixed preprocessing snapshot of government situation reports | Local-level water adjustment |
| `holidays.csv` | Official Japanese holiday dates for the observation period | Encode holidays as Sundays |
| `festivals.json` | Two documented 2024 festival intervals and original source links | Temporal-model event indicators |
| `nanao_water.csv` | 23 reports of water-flow restoration and confirmed drinkable supply, January 24-April 4 | Context figure only; not an input to the final model |

These tables are small projections of the project's existing public-feature
snapshots, not copied competition observations. They are intentionally bundled
to avoid silent changes when historical webpages or preprocessing change.

### Municipal water indices

Cabinet Office disaster reports, reports 18-40:

- https://www.bousai.go.jp/updates/r60101notojishin/r60101notojishin/index.html
- Report URL pattern:
  `https://www.bousai.go.jp/updates/r60101notojishin/r60101notojishin/pdf/r60101notojishin_NN.pdf`

The daily table is a derived, approximate household-level outage index, not
metered water consumption or an exact count of active residents. The original
feature snapshot was assembled during July-September 2026. The released model
uses the supplied table as frozen input; it does not rerun PDF extraction.
See `features.py` for endpoint holding and unknown-municipality handling.

### Grid assignment and plotting boundaries

MLIT National Land Numerical Information, N03 administrative areas dated
January 1, 2024, CC BY 4.0:

- https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2024.html
- Archive pattern:
  `https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2024/N03-20240101_PP_GML.zip`

Original cell assignment used prefectures Ishikawa (17), Toyama (16), and
Niigata (15). The spatial figure needs Ishikawa and Toyama only. Boundaries
provide geographic context, not measurements of post-earthquake coastline
displacement. Grid coordinates use the challenge's documented reference mapping.

### Calendar and festivals

- Cabinet Office holidays: https://www8.cao.go.jp/chosei/shukujitsu/gaiyou.html
- Machine-readable dates: https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv
  (project snapshot retrieved September 10, 2026).
- Abare festival, July 5-6, 2024: https://abarematsuri.jp/archives/263
- Ishizaki Hoto festival, August 3, 2024:
  https://www.city.nanao.lg.jp/koho/shise/koho/machinokao/r6/08/060803.html

The festival events were researched after observing validation spikes. They
are an exploratory modeling choice, not independent confirmation. The indicators
represent event dates, not attendance or independently sourced mobility counts.

### Nanao restoration reports

Nanao City water-service report index:
https://www.city.nanao.lg.jp/kurashi/sumai/suido/josuido/index.html

Example report (March 1, 2024):
https://www.city.nanao.lg.jp/jougesuidou/kurashi/sumai/suido/ryokin/documents/eriabetuitiran20240301.pdf

The extracted snapshots use 21,202 households as their denominator. Flow restored
and drinkable supply are different measures. Individual report dates, counts
and chronology are preserved in `nanao_water.csv`. These are shown only in the
paper's data-context figure.

## Rights and exclusions

The MIT code license does not override original source terms or attribution
requirements. Government source documents and downloaded geography retain their
own terms; cite their origin when reusing derived features or figures. No full
government PDFs, news articles, third-party validator code, raw mobility data,
credentials or API responses are bundled. The official validator is obtained
separately from its authors.
