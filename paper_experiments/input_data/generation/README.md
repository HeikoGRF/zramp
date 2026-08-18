# Regenerating the final paper inputs

The large generated mobility JSON and merged RSSI NPZ files are intentionally
not bundled. This directory preserves their exact provenance and provides a
generation-only launcher for the nine factorial maps and five temporal
realizations used by the paper.

The tracked maps, spatial metadata, 45 selected windows, and nine principal
fixed 10,000-pair evaluation sets remain under `input_data/`. Raw final
experiment outputs also remain in `results/`.

The recommended high-level helpers are:

- `scripts/download_lust3d.sh` to download, verify, and extract LuST3D;
- `scripts/select_timeframes.sh` to repeat the full-day scan and window
  selection; and
- this directory's `generate_main_inputs.sh` to preflight, submit, and verify
  all 45 mobility/RSSI pairs.

## External source

The mobility generator requires the extracted **LuST3d 1.0.0** dataset:

- Record: <https://doi.org/10.5281/zenodo.20799415>
- File: `LuST3d.zip`
- SHA-256: `7d86b3ee4f5bbbfe3898ab3561d4ea13290eba1e33b0d55b29d4f2d50d897d3e`

`external_lust3d_archive.sha256` records the downloaded archive checksum and
`external_lust3d_files.sha256` records all 16 extracted files used by the
project. The launcher refuses a source tree whose files differ.

From the `paper_experiments/` directory, download and extract it outside the
repository:

```bash
bash scripts/download_lust3d.sh /path/to/zramp-input-work/source
```

Compare the printed archive hash with the value above. The extracted directory
passed to the launcher must directly contain `lust3d.net.xml`,
`lust3d.poly.xml`, the route files, and `vtypes.add.xml`.

## Environment and preflight

Recreate the environment described in `environment/`. Then run from the
`paper_experiments/` directory, replacing the executable and source paths as
needed:

```bash
PYTHON_BIN=/path/to/environment/bin/python \
SUMO_BIN=/path/to/environment/bin/sumo \
  bash input_data/generation/generate_main_inputs.sh \
  --check /path/to/zramp-input-work/source/lust3d_v1
```

The preflight verifies the external-source hashes, installed package versions,
SUMO, nine Sionna map scenes, 45 selected windows, and the nine fixed evaluation
sets. The ray-tracing workers themselves require a compute node with the
LLVM/FMA support described in `environment/README.md`.

## Generate all 45 mobility/RSSI pairs

Ray tracing 81,000 frames is a cluster workload. The launcher submits only
input-generation jobs; it does not submit any learning experiment:

```bash
PYTHON_BIN=/path/to/environment/bin/python \
SUMO_BIN=/path/to/environment/bin/sumo \
SLURM_CONFIG=/path/to/slurm.conf \
SBATCH_ACCOUNT=your-account \
SBATCH_PARTITION=your-partition \
MAX_CONCURRENT_PER_ARRAY=20 \
  bash input_data/generation/generate_main_inputs.sh \
  --submit /path/to/zramp-input-work/source/lust3d_v1 \
           /path/to/zramp-input-work/generated-main-inputs
```

`SLURM_CONFIG`, `SBATCH_ACCOUNT`, `SBATCH_PARTITION`, and `SBATCH_EXCLUDE`
are optional overrides. `WINDOWS_TSV` may point to a newly selected 45-row
table; when omitted, the exact paper table is used. Set scheduler variables for
the target cluster. The launcher creates 45 dependency chains, one for every
map/replicate pair:

1. SUMO generates the exact one-second FCD window and exports mobility JSON.
2. A 180-task array ray traces 10 frames per task with Sionna RT.
3. A dependent merge job validates frames 0–1799, filters measurements below
   -100 dBm, and writes the merged RSSI NPZ.

The radio configuration is 3.5 GHz, 23 dBm transmit power, 20,000 rays,
maximum depth 3, opaque buildings, roof antennas, dynamic vehicle blockers,
and no refraction. These values come from the archived paper launchers.

A submission manifest is written to
`generated-main-inputs/submitted_input_jobs.tsv`. Generated FCD files and
per-frame ray shards remain in the external working directory and may be
removed after the merged products have been verified.

## Verify generated inputs

After all merge jobs finish:

```bash
PYTHON_BIN=/path/to/environment/bin/python \
SUMO_BIN=/path/to/environment/bin/sumo \
  bash input_data/generation/generate_main_inputs.sh \
  --verify /path/to/zramp-input-work/generated-main-inputs
```

Verification parses every mobility file, opens every RSSI archive, and checks
all 45 pairs for the expected formats and 1,800-frame horizon.

Generated pairs follow this layout:

```text
generated-main-inputs/replicates/factor_b{1..3}_v{1..3}_300m/rep{1..5}/
├── mobility/<zone>_rep<r>_all_vehicles_1s_full1800.json
└── rssi/<zone>_rep<r>_vehicles_1s_opaque_no_vehicle_blockers_ge-100dbm_r20k_d3_llvm.npz
```

Use the generated RSSI path as `--trace` when rerunning a paper configuration.
The corresponding fixed test set and radio-bounds network remain in the
repository. Absolute source/output paths embedded as provenance may make
regenerated files differ byte-for-byte even when their scientific arrays and
settings agree.

## Omitted-file record

`omitted_generated_inputs.tsv` lists the original archival path, role,
byte count, and SHA-256 digest of every removed mobility/RSSI intermediate,
including the duplicate legacy mobility trace. It records 134 files; it is an
inventory, not a claim that portable regeneration produces identical metadata
paths.

The preserved `selected_windows.tsv` is the authoritative temporal design.
Scripts for rescanning a full day and selecting new windows are retained in
`code/final/SUMO/luxembourg_real_city/`, but reselecting windows would define a
new experiment rather than reproduce the reported one.
