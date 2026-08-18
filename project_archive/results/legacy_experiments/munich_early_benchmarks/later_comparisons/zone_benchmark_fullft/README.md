# Zone benchmark (full fine-tuning, no adapters)

This folder is self-contained: run the benchmark from here and all outputs are written to `results/`.

## Run (locally)

```bash
python run_zone_benchmark_fullft.py
```

## Run (Slurm)

```bash
sbatch submit_zone_benchmark_fullft.sh
```

Outputs:
- `results/benchmark_fullft.png`
- `results/benchmark_fullft_analysis.png`
- `results/benchmark_fullft_summary.txt`

