# exp25 q=0.0100/0.0500/0.1000 dense pooled summary

- Sources: `exp24a/b/c`, pooled across three independent nodes.
- Grid: `q=0.0100,0.0500,0.1000`, `L=3,4,5`, `p=0.180,0.185,...,0.220`.
- Sampling: `1152` disorder per q after pooling.
- Main result: `q=0.0500` gives a common crossing window `p≈0.2032~0.2121`, representative point `p≈0.2076`.
- Direction check: `q=0.0100` remains mostly below threshold through `p=0.220`; `q=0.1000` is already above threshold at `p=0.180`.
- Caveat: `q=0.0100/0.0500` still have poor L=4/5 mixing diagnostics, so this is strong window evidence rather than a final precision threshold.

Key files:
- `exp25_pooled_sem95_overview.png`
- `exp25_pooled_gap_overview.png`
- `q0p0500_exp21_exp24_shared_p_overlay.png`
- `exp25_pooled_summary.json`
