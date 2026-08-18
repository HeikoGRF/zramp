# Paper result tier

`paper/` contains only the four raw result roots consumed by the authoritative
five-replicate aggregation. Development and negative results are kept
separately under `../../project_archive/results/legacy_experiments/`.

Rebuild the compact tables from the `paper_experiments/` directory:

```bash
bash scripts/rebuild_paper_outputs.sh
```

New simulation outputs should be written outside the Git repository or under
`results/reproduced/`, which is ignored by Git.

