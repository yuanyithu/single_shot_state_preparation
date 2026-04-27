# exp29_fixed_p010_q000_100_L3456_combined_summary

- Purpose: fixed `p=0.1000` q-axis summary combining `exp27` L=3/4/5 with pooled `exp28a/b/c` L=6.
- Inputs: `exp27_fixed_p010_q000_100_combined_summary` and `exp28a/b/c_fixed_p010_q000_100_L6_20260427`.
- Disorder: each q has `1536` disorder for L=3/4/5 and `1536` independent disorder for L=6.
- Main plots:
  - `fixed_p010_q000_100_exp26abc_L345_exp28abc_L6_pooled_sem95.png`
  - `fixed_p010_q000_100_exp26abc_L345_exp28abc_L6_pooled_gap_ci95.png`
  - `fixed_p010_q000_100_exp26abc_L345_exp28abc_L6_pooled_summary.json`

Interpretation:

- `L3-L4` crosses near `q≈0.0608`.
- `L4-L5` crosses near `q≈0.0247`.
- `L5-L6` is positive over the full scanned window, but L6 q>0 convergence fails and PT min swap acceptance is near zero.
- The L6 extension is therefore a mixing diagnostic, not a final threshold update. A reliable L6 run needs a stronger PT ladder and longer chains.

Gap CI note: `L5-L6` uses independent-disorder error propagation because L6 was run as a later independent extension. `L3-L4` and `L4-L5` keep the paired-disorder CI from `exp27`.
