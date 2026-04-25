# exp23_q050_L3456_p018_022_combined_summary

- Sources: `exp21` pooled L=3,4,5 plus `exp22a/b/c` pooled L=6.
- Parameters: `q=0.0500`, `p=0.18,0.19,0.20,0.21,0.22`, `2048` measurements/disorder, `PT7`.
- Disorder samples: L=3,4,5 use `288` pooled disorder; L=6 uses `288` pooled disorder.
- Direction rule: below threshold larger L should have larger `q_top`; above threshold larger L should have smaller `q_top`.
- Crossings from linear interpolation: `{'L3_minus_L4': [0.19288108586182065], 'L4_minus_L5': [], 'L5_minus_L6': []}`.
- Main read: L=6 lies below L=5 across this whole window, while L4-L5 still has not crossed by p=0.22. This is not a clean four-size threshold crossing.
- Caveat: L=6 mixing diagnostics are weak (`mean_q_top_spread` up to about 0.108 and max R-hat about 1.35), so the L=6 curve should be treated as a diagnostic extension rather than a final threshold estimate.
- Key plots: `q050_L3456_pooled_sem95.png`, `q050_L3456_pooled_gap_ci95.png`.
