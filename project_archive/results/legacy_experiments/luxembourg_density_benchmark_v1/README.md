# Luxembourg controlled traffic-density benchmark v1

This campaign varies active vehicle/contact density independently on two fixed
propagation environments: the 100 m corridor `factor_b2_v2_hotspot_100m` and
the balanced 200 m zone `factor_b2_v2_200m`.

## Density construction

The original observed trajectories are retained exactly. For factor F, every
trajectory is copied F times and cyclically phase-shifted by equal offsets over
the 1,800-second window. Each copy has its own persistent vehicle/model
identity. This changes simultaneous traffic and contacts while preserving:

- the selected map and building geometry;
- the set of routes and per-trajectory motion;
- the radio propagation configuration; and
- the same 10,000-link held-out test set for that map.

This open-loop construction is appropriate because dynamic vehicles are
disabled as radio blockers. It does not model congestion feedback.

| Map | Traffic | Nodes | Minimum active | Median active | Maximum active |
| --- | --- | ---: | ---: | ---: | ---: |
| 100 m | 1× source | 935 | 1 | 11 | 32 |
| 100 m | 2× | 1,870 | 9 | 22 | 52 |
| 100 m | 4× | 3,740 | 30 | 46 | 67 |
| 200 m | 1× source | 1,149 | 4 | 17 | 39 |
| 200 m | 2× | 2,298 | 11 | 34 | 67 |
| 200 m | 4× | 4,596 | 51 | 68 | 93 |

The 100 m test set is an all-feasible corridor stress case. The 200 m test set
is 48.15% feasible and 51.85% blocked, so it also measures false positives.
Augmented mobility JSON is compactly serialized.

## Evaluations

For each new density, the campaign runs expert-bank penalties 2%, 10%, and
50%, plus local-only and ideal-central baselines. The 1× counterparts are
available in the adjacent `luxembourg_nested_size_benchmark_v1/` directory.

- Results: `results/legacy_experiments/luxembourg_density_benchmark_v1/`
- 100 m submission manifest: `submitted_jobs.tsv`
- 200 m submission manifest: `submitted_200m_jobs.tsv`
- Launcher:
  `code/legacy/campaign_implementations/SUMO/luxembourg_real_city/submit_density_sweep.sh`
- Mobility augmenter:
  `code/legacy/campaign_implementations/SUMO/luxembourg_real_city/augment_mobility_density.py`

