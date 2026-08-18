# Paper runtime model

`paper_runtime/cell_grid_patch_acquisition_v1_c16_pq4x256/` contains the small
bundle loaded by the paper runner to initialize its grid-support
representation. The final ranking and merge weights use deterministic summed
grid intensity, and the runs report that learned acquisition scoring was not
used.

Experimental acquisition and cross-map policy checkpoints are kept under
`../../project_archive/trained_models/experimental/`.

