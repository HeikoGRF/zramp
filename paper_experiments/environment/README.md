# Environment

The final trace-generation environment used Python 3.10.20, Eclipse SUMO
1.27.0, Sionna RT 1.2.2, Mitsuba 3.8.0, Dr.Jit 1.3.1, NumPy 2.2.6,
Pandas 2.3.3, SciPy 1.15.3, and PyTorch 2.12.0+cpu on Linux x86_64.

`requirements-lock.txt` is the sanitized output of `pip freeze`. The original
freeze represented `packaging` by a build-host file URL; it is recorded here
as the equivalent installed version, `packaging==26.2`, so the lock is
portable. `conda-history.txt` is copied verbatim from the final environment.

Create the environment from this directory:

```bash
conda env create -f environment.yml
conda activate zramp-archive
```

The paper simulations consume the JSON/NPZ products written by
`input_data/generation/generate_main_inputs.sh`; those large intermediates are
not bundled. Sionna RT is needed for that radio-trace generation step.
Sionna RT/Mitsuba/Dr.Jit may require a newer x86-64 CPU or a supported GPU than
an arbitrary login node. Run trace generation on a compatible compute node.
The archived Slurm scripts show the resource requests used during the project,
but cluster partition and account names must be adapted locally.
