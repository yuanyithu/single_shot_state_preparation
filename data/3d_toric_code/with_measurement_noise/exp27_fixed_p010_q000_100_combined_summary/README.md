# exp27 fixed p=0.1000 q scan pooled summary

- Sources: `exp26a/b/c`, pooled across three independent nodes.
- Grid: fixed `p=0.1000`, `q=0.0000,0.0100,...,0.1000`, `L=3,4,5`.
- Sampling: `1536` disorder per q after pooling.
- Main result: `L4-L5` gap crosses near `q≈0.0247`; `L3-L4` gap crosses near `q≈0.0608`.
- Interpretation: the two pairwise crossings are separated, so this is not a clean three-size common q-threshold at fixed `p=0.1`.
- Caveat: strict q>0 convergence gates fail, mainly through ESS/R-hat; use this as window evidence and improve mixing before precision threshold claims.

Key files:
- `fixed_p010_q000_100_exp26abc_pooled_sem95.png`
- `fixed_p010_q000_100_exp26abc_pooled_gap_ci95.png`
- `fixed_p010_q000_100_exp26abc_pooled_summary.json`
