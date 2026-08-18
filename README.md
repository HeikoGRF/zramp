# ZRAMP research repository

This repository puts the experiment reported in the paper first: the
deterministic intensity-count cell-grid method and its paper baselines on nine
Luxembourg maps and five temporal realizations. It also preserves the
substantial development and negative experiments that led to the final design.

The repository itself is the project archive. The final paper workflow is kept
once under `paper_experiments/`; `project_archive/` adds the historical
experiments and manuscript sources without duplicating the final implementation.

## Repository structure

| Path | Purpose |
|---|---|
| `paper_experiments/` | Authoritative code, compact inputs, exact configurations, raw paper results, figures, and an end-to-end reproduction guide. |
| `paper_experiments/code/final/` | Final preprocessing, simulation, and analysis entry points. |
| `paper_experiments/code/shared_runtime/` | Simulator machinery shared by the final and historical experiments. |
| `project_archive/` | Documented development studies, negative results, legacy source, and experimental checkpoints. |
| `MANUSCRIPT.zip` | Complete Overleaf work tree containing the paper, report, presentation, figures, bibliography, and style files. |
| `project_archive/EXPERIMENT_CATALOG.md` | Reader-facing distinction between paper evidence and exploratory work. |
| `scripts/check_repository.py` | Pre-push check for caches and files at GitHub's 50 MiB warning threshold. |

## Reproduce the paper

Start with [paper_experiments/README.md](paper_experiments/README.md). Its
workflow covers:

1. creating the Python/SUMO/Sionna environment;
2. downloading and verifying LuST3D;
3. using the exact stored temporal windows or selecting new ones;
4. generating 45 mobility and Sionna RSSI trace pairs;
5. running any subset or all 495 paper simulations; and
6. regenerating confidence intervals and plot-ready tables.

The large generated mobility and RSSI intermediates are deliberately written
outside the repository. Exact maps, temporal selections, fixed evaluation
sets, provenance, and the code that regenerates the omitted intermediates are
tracked.

## Review the broader project

Read [project_archive/README.md](project_archive/README.md), then
[project_archive/EXPERIMENT_CATALOG.md](project_archive/EXPERIMENT_CATALOG.md).
Historical campaigns retain their configurations, relevant outputs, code, and
conclusions. They reuse the maintained runtime and compact inputs under
`paper_experiments/`.

The complete Overleaf work tree is provided in `MANUSCRIPT.zip`.

## Before pushing

Run:

```bash
python3 scripts/check_repository.py
```

The repository intentionally tracks many small result artifacts, but no file
should reach 50 MiB. Generated traces, scheduler logs, caches, ZIP files, and
new reproduced result trees are ignored so normal reruns do not inflate Git
history.

