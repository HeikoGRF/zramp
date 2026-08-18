# Place Wallis method-development benchmark

This is the fixed result location for all methods evaluated on the Place Wallis
300 × 300 m map. Every row uses the same 10,000-pair static Sionna test set,
opaque buildings, no vehicle blockers, and a −100 dBm censor/noise floor.
For trainable methods, the definitive score is the arithmetic mean of ten
evaluations spaced 25 seconds apart over the final 250 seconds (steps 1574,
1599, 1624, 1649, 1674, 1699, 1724, 1749, 1774, and 1799). The corresponding
population standard deviations are stored in each method’s `metrics.json`. RMSE is
also evaluated every 50 steps to form the convergence curve, but those regular
checkpoints are not averaged into the definitive score.

Values exactly at −100 dBm are censored non-feasible links. Therefore, the
definitive split is:

- feasible: `RSSI > -100 dBm`
- non-feasible: `RSSI <= -100 dBm`

## Results

| Method | Overall RMSE | Feasible RMSE | Non-feasible RMSE |
|---|---:|---:|---:|
| Noise-floor baseline | 25.679 dB | 41.984 dB | 0.000 dB |
| Equal-weight greedy sharing | 26.760 dB | 12.257 dB | 32.465 dB |
| Bounded pair-RBF greedy sharing | 24.297 dB | 39.652 dB | 1.864 dB |
| Support expert bank, K=3 | 15.378 dB | 23.304 dB | 7.296 dB |
| Support expert bank, K=6 | 14.490 dB | 20.203 dB | 9.567 dB |
| Support expert bank, K=9 | 14.476 dB | 19.174 dB | 10.728 dB |
| Bounded K=3 bank, 8 degrees | 16.242 dB | 23.955 dB | 8.858 dB |
| Bounded K=3 bank, 12 degrees | 14.879 dB | 21.972 dB | 8.073 dB |
| Bounded K=3 bank, 16 degrees | 16.012 dB | 23.627 dB | 8.715 dB |
| Dominance-pruned unbounded bank | 14.677 dB | 17.309 dB | 12.849 dB |
| Learned-acquisition unbounded bank | 14.884 dB | 18.716 dB | 11.989 dB |

## Interpretation

Support-aware expert banks were promising on this one map, but the ranking of
variants changed with support geometry and the result was not yet a robust
multi-map comparison. The campaign motivated the later factorial evaluation
and the simpler deterministic cell-grid method. It is retained as development
evidence, not as a paper result.

`results.csv` is the machine-readable table. Rows marked `incomplete` record
attempts for which no comparable final score was produced. Complete raw runs
are under `methods/`; smoke tests are under `smoke/`.

The corresponding source is in
`code/legacy/campaign_implementations/experiments/place_wallis_benchmark/` and its
input-generation launchers are under
`code/legacy/campaign_implementations/SUMO/luxembourg_real_city/`.
