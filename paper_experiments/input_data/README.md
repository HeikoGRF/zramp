# Input data and regeneration

This directory preserves the compact, irreplaceable inputs and the exact
workflow needed to regenerate large simulation intermediates. Generated
mobility JSON and merged RSSI NPZ files are deliberately not bundled.

## Layout

- `maps/luxembourg_real_city/`: terrain, building meshes, Sionna scenes, radio
  bounds, and related spatial assets.
- `zone_metadata/`: crop manifests, factorial-zone definitions, building
  fractions, and vehicle-level survey metadata.
- `prepared_traces/luxembourg_real_city/`: fixed static evaluation sets retained
  at their original paths.
- `prepared_traces/luxembourg_ci_temporal5_paper_final_v1/`: the authoritative
  45-row temporal selection in JSON and TSV form.
- `generation/`: external-source checksums, omitted-file inventory, generation
  launcher, and complete instructions.

## Final paper inputs

The main experiment uses nine factorial maps and five temporal realizations.
For every map/realization, the generation workflow produces:

- a one-second mobility JSON covering frames 0–1799; and
- a merged Sionna-derived RSSI NPZ covering the same horizon.

All five realizations use the tracked map geometry and the same per-map fixed
10,000-pair test set. Replicate 1 is the 07:45–08:15 window; replicates 2–5 use
the other preserved entries in `selected_windows.tsv`.

Run the preflight and generation-only Slurm launcher as documented in
`generation/README.md`. The short entry point is:

```bash
bash input_data/generation/generate_main_inputs.sh --check SOURCE_ROOT
bash input_data/generation/generate_main_inputs.sh \
  --submit SOURCE_ROOT EXTERNAL_OUTPUT_ROOT
```

Generated data is written to `EXTERNAL_OUTPUT_ROOT`, not back into the repository.
This keeps the submission compact while leaving the exact expensive
preprocessing and radio-tracing computation reproducible.

## Retained evaluation and auxiliary material

Twenty-two available fixed evaluation sets are retained because together they
occupy only about 4 MiB and define exact held-out samples. Nine belong to the
factorial paper maps; the others support selected size, density, additional-map,
and matched-zone result records.

The map collection likewise supports both the final experiment and selected
historical real-city results. Historical configuration files may retain their
original cluster paths as provenance; translate those paths to the tracked
maps/test sets and newly generated trace root when rerunning.

## Omitted intermediates

`generation/omitted_generated_inputs.tsv` records the SHA-256 digest, original
size, role, and path of all 134 removed files: 65 mobility products, 68 merged
RSSI products, and one duplicate legacy mobility trace. Their original total
size was 11,111,359,863 bytes.

Raw FCD streams and per-frame ray shards were already treated as transient
intermediates. The new generation workflow creates them in external working
storage and merges them into the simulation-ready products.
