# Place Wallis 300 m city dataset

The selected crop covers SUMO coordinates `[7350, 5900]` through
`[7650, 6200]`. The Sionna scene uses crop-local coordinates `[0, 0]` through
`[300, 300]` and includes a 200 m geometry buffer.

## Preparation

```bash
SUMO/luxembourg_real_city/prepare_place_wallis_inputs.sh
```

This extracts a 10 m terrain grid from Luxembourg's official 2024 DTM, builds
the buffered building/terrain scene, and exports all physical vehicle IDs seen
in the 1,800 one-second frames from 07:45:00 through 08:14:59.

## Parallel Sionna trace

```bash
SUMO/luxembourg_real_city/submit_place_wallis_opaque_no_vehicle_blockers.sh
```

Five Slurm arrays cover frame offsets 0 through 4, for exactly 1,800
single-frame shards. Every active vehicle in the crop acts as transmitter and
receiver. The physical trace floor is -120 dBm; the final merge retains only
directed measurements with RSSI greater than or equal to -100 dBm.

The default output root is:

```text
/usr/itetnas04/data-scratch-01/hgraef/data/luxembourg_real_city/place_wallis_300m_30min_opaque_buildings_no_vehicle_blockers
```

Propagation matches the retained Bonnevoie configuration: 3.5 GHz, 23 dBm,
20,000 rays, depth 3, exact type-specific roof antennas, opaque buildings,
line of sight and specular reflections enabled, and no vehicle blocker meshes.

The merged archive records the filter and both the unfiltered and retained row
counts in `meta_json["measurement_filter"]`.
