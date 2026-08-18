# Luxembourg nested map-size benchmark v1

This campaign extends the existing nine-map 300 m benchmark without modifying
its result directories.

## Design

The 200 m crops are concentric with their existing 300 m parents. The centered
100 m crops were rejected after mobility export because their median active
vehicle counts were 0, 1, and 6. For a meaningful small-map dissemination
stress test, each replacement is the 10 m-aligned 100 m window inside the
centered 200 m crop with the highest median active count over the same fixed
30-minute LuST trace.

| Building class | 100 m crop | Median active | Scene buildings | 200 m median active | 200 m scene buildings |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse | factor_b1_v2_hotspot_100m | 17 | 39 | 23 | 53 |
| medium | factor_b2_v2_hotspot_100m | 11 | 131 | 17 | 151 |
| dense | factor_b3_v2_hotspot_100m | 12 | 174 | 20 | 232 |

The 100 m crops therefore also provide the requested higher-density small-map
stress case. Realized active-vehicle density must be reported alongside map
size; it is not an independently controlled density factor.

## Evaluations

Each usable 100 m and 200 m dataset runs:

- learned-acquisition expert bank with communication penalties 2%, 10%, 50%;
- local-only support-gated MLP baseline; and
- ideal central support-gated MLP baseline.

All runs use the same 10,000-link held-out test size and infer the square map
side from matching trace/test-set metadata. The learned acquisition grid stays
fixed in normalized unit-square coordinates.

The initial centered 100 m downstream jobs were canceled before evaluation.
Their generated preparation artifacts remain available only for audit.

## Locations

- Results: `results/legacy_experiments/luxembourg_nested_size_benchmark_v1/`
- Maps and crop manifest: `input_data/maps/luxembourg_real_city/`
- Primary submissions: `submitted_jobs.tsv`
- Replacement 100 m submissions: `submitted_hotspot_100m_replacements.tsv`

The earlier protected working snapshot was a duplicate and is not part of this
curated archive; the final factorial paper roots are under `results/paper/`.

