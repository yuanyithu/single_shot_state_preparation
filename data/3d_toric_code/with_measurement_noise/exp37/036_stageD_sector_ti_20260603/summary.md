# Stage D sector-resolved TI progress summary

Overall: PASS

Accepted artifacts are in `accepted_combined/`.  The Stage D estimator is
sector-resolved thermodynamic integration against the Stage B exact L=2
zero-disorder benchmark; no AIS, FEP, or flip-reweighting is used.

The accepted benchmark uses the Stage C fixed-sector sampler in
`linear_kernel` mode.  For this zero-disorder linear-section L=2 reference,
the `P_L x` sector labels are equivalent to the corrected decoder-sector
labels, while avoiding expensive per-proposal section/signature calls.

## Accepted Run Composition

- Records 0,2,3,4,5: `full_linear_m1024/`
- Record 1: longer targeted rerun `rerun_r1_linear_m2048/`
- Combined artifacts: `accepted_combined/stageD_results.json`,
  `accepted_combined/ti_results.npz`,
  `accepted_combined/ti_comparison.csv`, and
  `accepted_combined/summary.md`

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| D1 | TV(w_TI,w_exact) <= 0.020 | max TV=0.004957 | PASS |
| D2 | abs dq_top <= 0.020 and CI covers exact | max abs dq=0.00545, CI misses=0 | PASS |
| D3 | coarse/fine grid TV and abs dq <= 0.020 | max grid TV=0.004336, max grid dq=0.003939 | PASS |

## Notes

The initial full six-point run with `grid=65,burn=120,m=1024,stride=2`
passed D1/D3 globally but missed D2 for record 1 by a narrow CI miss.
The targeted record-1 rerun with `burn=160,m=2048,blocks=64,bootstrap=1000`
passed all gates and is used in the accepted combined result.
