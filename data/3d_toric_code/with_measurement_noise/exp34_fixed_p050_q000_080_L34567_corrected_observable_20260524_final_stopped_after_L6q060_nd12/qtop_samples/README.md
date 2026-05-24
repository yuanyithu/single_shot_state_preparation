# exp34 q_top Samples

Local archive:

- `exp34_corrected_observable_qtop_samples_only.npz`

The archive is intentionally ignored by Git. It stores only the data needed to redraw and recheck the corrected-observable fixed-`p=0.0500` q scan:

- `point_label`, `lattice_size`, `data_error_probability`, `syndrome_error_probability`
- `q_top_mean`, `q_top_std`, `converged`
- `q_top_samples_by_point`: per-disorder `q_top` values for each `(L,p,q)` point
- `q_top_samples_by_point_and_chain`: per-disorder, per-start-chain `q_top` values for diagnostic checks

Shape at creation on 2026-05-24:

- points: `34`
- samples per point: `2048`
- chains per point: `8`
