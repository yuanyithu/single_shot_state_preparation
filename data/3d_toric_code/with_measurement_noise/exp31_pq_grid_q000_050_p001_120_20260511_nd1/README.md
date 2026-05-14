# exp31 pq grid q000 050 p001 120 nd1

Purpose: scan the 3D toric code with measurement noise on a rectangular
`(p,q)` grid and analyze the threshold boundary in the fixed-`p`, q-scan
direction.

## Run

- Remote run id: `3d_toric_pq_grid_q000_050_nd1_20260511_175054`
- Remote host: `nd-1`
- Local copy: `remote_run/`
- Lattice sizes: `L=3,4,5`
- Data error probabilities: `p=0.0100,0.0300,0.0500,0.0700,0.1000,0.1200`
- Measurement error probabilities: `q=0.0000,0.0050,...,0.0500`
- Disorder samples: `256` per `(L,p,q)`
- Chunk size: `16`
- q-positive sampler: `num_start_chains=8`, `num_replicas_per_start=2`,
  `pt_num_temperatures=9`, `pt_p_hot=0.44`
- Completion: `3168/3168` chunks, `0` failed
- Wall time on nd-1: about `59.37` hours

## Outputs

- [remote q_top overview](remote_run/measurement_noise_threshold_search_sem95_overview.png)
- [remote per-q gap summary](remote_run/measurement_noise_threshold_search_gap_summary.png)
- [p-q pairwise boundary plot](analysis/exp31_pq_grid_pq_boundary.png)
- [fixed-p q curves](analysis/exp31_pq_grid_fixed_p_q_curves_sem95.png)
- [fixed-p q scan q_top plot](analysis/exp31_pq_grid_q_scan_sem95.png)
- [fixed-p q scan gap plot](analysis/exp31_pq_grid_q_gap_scan.png)
- [diagnostic heatmaps](analysis/exp31_pq_grid_diagnostic_heatmaps.png)
- [boundary summary JSON](analysis/exp31_pq_grid_boundary_summary.json)
- [boundary point CSV](analysis/exp31_pq_grid_boundary_points.csv)

## Interpretation

The threshold direction is: below threshold, larger `L` has larger `q_top`,
so the adjacent-size gap `q_top(L_small)-q_top(L_large)` is negative; above
threshold the gap becomes positive.

This run does not produce a stable common boundary curve. No scanned `p`
has both adjacent-size pairs (`L3-L4` and `L4-L5`) giving a clean single
crossing in `q`. The usable pairwise signs are:

- `L3-L4`: single crossings at `p=0.0300` (`q≈0.0196`) and `p=0.0500`
  (`q≈0.0222`); no crossing up to `q=0.0500` at `p=0.1000` and `p=0.1200`.
- `L4-L5`: single crossing only at `p=0.1200` (`q≈0.0376`); most other
  `p` values are nonmonotonic or noise-sensitive.

The q-positive convergence gate is also poor: only `5/180` q-positive
lattice points pass the strict lattice-point diagnostics. Therefore the
boundary plot should be read as a finite-size/mixing diagnostic, not as a
final threshold curve.
