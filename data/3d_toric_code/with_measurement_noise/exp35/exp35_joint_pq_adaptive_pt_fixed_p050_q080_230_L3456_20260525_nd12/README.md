# exp35 fixed p=0.0500, q=0.0800..0.2300 scan

- Status: `complete`.
- Audit status: `invalid for threshold interpretation`; see [`audit_exp35_20260527.md`](audit_exp35_20260527.md).
- Grid: fixed `p=0.0500`, `q=0.0800,0.0900,...,0.2300`, `L=3,4,5,6`.
- Pooling: independent nd-1 and nd-2 source runs, expected `2048` disorder per `(L,q)` after pooling.
- Observable: corrected `c + eta + r(H_Z c) + r(H_Z eta)` with BP-LSD section diagnostics.
- Manifest summary: [`manifest_summary.json`](manifest_summary.json).
- Diagnostics summary: [`diagnostics_summary.json`](diagnostics_summary.json).
- q_top plot: [`analysis/fixed_p050_q080_230_exp35_joint_pq_adaptive_pt_nd12_pooled_sem95.png`](analysis/fixed_p050_q080_230_exp35_joint_pq_adaptive_pt_nd12_pooled_sem95.png).
- gap plot: [`analysis/fixed_p050_q080_230_exp35_joint_pq_adaptive_pt_nd12_pooled_gap_ci95.png`](analysis/fixed_p050_q080_230_exp35_joint_pq_adaptive_pt_nd12_pooled_gap_ci95.png).
- fixed-p summary: [`analysis/fixed_p050_q080_230_exp35_joint_pq_adaptive_pt_nd12_pooled_summary.json`](analysis/fixed_p050_q080_230_exp35_joint_pq_adaptive_pt_nd12_pooled_summary.json).

Warning: the pooled files are complete, but the audit found severe logical-sector
mixing and convergence-gate failures. Do not use the plots in this directory as
evidence for a physical q-threshold.
