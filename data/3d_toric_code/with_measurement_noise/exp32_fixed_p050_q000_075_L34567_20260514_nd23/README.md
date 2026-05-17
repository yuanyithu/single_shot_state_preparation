# exp32 fixed p=0.0500 q scan, nd-2/nd-3

- Status: `complete`.
- Grid: fixed `p=0.0500`, `q=0.0000,0.0050,...,0.0750`, `L=3,4,5,6,7`.
- Pooling: independent nd-2 and nd-3 source runs, expected `2048` disorder per `(L,q)` after pooling.
- Manifest summary: [`manifest_summary.json`](manifest_summary.json).
- Diagnostics summary: [`diagnostics_summary.json`](diagnostics_summary.json).
- q_top plot: [`analysis/fixed_p050_q000_075_exp32_nd23_pooled_sem95.png`](analysis/fixed_p050_q000_075_exp32_nd23_pooled_sem95.png).
- gap plot: [`analysis/fixed_p050_q000_075_exp32_nd23_pooled_gap_ci95.png`](analysis/fixed_p050_q000_075_exp32_nd23_pooled_gap_ci95.png).
- fixed-p summary: [`analysis/fixed_p050_q000_075_exp32_nd23_pooled_summary.json`](analysis/fixed_p050_q000_075_exp32_nd23_pooled_summary.json).

## Summary

- Remote completion: nd-2 and nd-3 each completed `80/80` child runs, with `failed_chunks=0` and `pending_chunks=0`.
- Pooling: all `80` `(L,q)` points were pooled from the two independent nodes; every point has `2048` disorder samples.
- Main plots:
  - [`analysis/fixed_p050_q000_075_exp32_nd23_pooled_sem95.png`](analysis/fixed_p050_q000_075_exp32_nd23_pooled_sem95.png)
  - [`analysis/fixed_p050_q000_075_exp32_nd23_pooled_sem95_tight.pdf`](analysis/fixed_p050_q000_075_exp32_nd23_pooled_sem95_tight.pdf)
  - [`analysis/fixed_p050_q000_075_exp32_nd23_pooled_gap_ci95.png`](analysis/fixed_p050_q000_075_exp32_nd23_pooled_gap_ci95.png)
  - [`analysis/fixed_p050_q000_075_exp32_nd23_pooled_gap_ci95_tight.pdf`](analysis/fixed_p050_q000_075_exp32_nd23_pooled_gap_ci95_tight.pdf)
  - [`analysis/fixed_p050_q000_020_exp32_nd23_pooled_focus_qtop_gap_tight.pdf`](analysis/fixed_p050_q000_020_exp32_nd23_pooled_focus_qtop_gap_tight.pdf)
- Pairwise gap sign changes:
  - `L3-L4`: `q≈0.0182`
  - `L4-L5`: noisy double sign change near `q≈0.0036` and `q≈0.0064`
  - `L5-L6`: `q≈0.0074`
  - `L6-L7`: `q≈0.0004`
- Diagnostics: q=0 multi-start spread passed for all 5 lattice sizes; q>0 convergence gate passed only `2/75` lattice-points. Large-L/high-q points have low ESS and PT min swap acceptance near zero.

Interpretation: this run gives a high-statistics fixed-`p=0.0500` q scan, but it does not support a clean common q-threshold. The larger-size pair crossings sit much closer to `q=0` than `L3-L4`, and most q>0 diagnostics fail, so the result should be read as finite-size drift plus mixing limitations rather than a final threshold estimate.
