# exp102 validation index

Current status is pre-pilot, not scientific certification.

- `001_local_implementation_20260719/`: registry cardinality/dimensions, task-plan identity,
  exp102 unit tests, and selected exp101 regression tests (83 combined PASS after the Numba update).
- `002_numba_smoke_20260719/`: local nonzero-syndrome pilot-cell smoke; diagnostic only and not a
  pilot pass. The deliberately tiny round budget fails mixing gates as expected.
- `003_numba_engine_20260720/`: full-round Numba/reference bit-identity and performance benchmark,
  including the `m=8,k=64` boundary. Local speedup is about 177x--196x.
- `004_pilot_ladder_20260720/`: clean-SHA three-node preflight and the complete configured ladder
  search. The maximum candidate fails m=4..8, so the pilot is fail-closed before gamma/held-out.
- `005_pt_v2_discovery_20260720/`: isolated Q32-ladder/multi-swap discovery implementation,
  cross-node digest runner, immutable ownership launcher, and discovery-only raw validation. The
  three-node screen completed, but all 12 transport candidates produced zero certified round
  trips; the frozen route is `EXHAUSTED` before S128/confirmation. Report-v3 also binds every raw
  file to its source/control/ownership/marker evidence.
- `006_pa_discovery_20260721/`: reviewed successor search using fixed-schedule q=0 population
  annealing plus a no-extra-randomness replay of 16 old PT trajectories. The implementation,
  frozen Q32 schedules/panels, exact oracles, reference/Numba identity, raw analyzer, immutable
  node ownership, marker verification, and runtime tools are certified. The clean-source Linux
  runtime gate and all 4 autopsy plus 64 hard-screen tasks completed. Autopsy is four times
  `INCONCLUSIVE` because conditioned attempts are insufficient; all four PA methods fail both hard
  cells through catastrophic genealogy collapse. The frozen zero-pass branch is `EXHAUSTED`, so
  rescue and blinded confirmation were not run and this is not `READY_FOR_FORMAL`.
- `007_q0_global_discovery_20260721/`: isolated global-sampling successor implementation. It
  includes deterministic logical catalogs, hard-coset cluster/joint heatbath kernels, independent
  defect trace, exact HGP/WMC oracles, no-pickle raw replay, character/D2 gates, m3 full-sector TI
  anchors, three-node preflight/runtime/digest consensus, immutable stage ownership, 72-hour
  schedule, postselection/control freezes, and fail-closed readiness. Local implementation tests
  pass; three immutable preflight attempts are audited below. The final attempt stopped at the
  frozen runtime gate before sampler work. No discovery stage completed, so this directory
  contains no physics result.
- `008_q0_global_preflight_portability_20260721/`: immutable evidence from the failed first global
  preflight plus its repair. It fixes archive git provenance, deterministic legacy-section testing,
  source-tree writes by the spec example test, cold-JIT TI projection, and post-worker source
  reverification. A fresh commit/deployment/run/schedule is required; this is not a sampler result.
- `009_q0_global_runtime_gate_separation_20260721/`: immutable evidence from the failed second
  preflight. A live performance fixture fluctuated at the TI wall boundary and incorrectly failed
  the deterministic regression suite before the persisted three-node runtime gate. Tests now
  validate live-report self-consistency while the unchanged dedicated consensus alone decides
  machine eligibility. The same-node postmortem passed T3; no sampler task ran.
- `010_q0_global_runtime_exhausted_20260721/`: immutable evidence from the third clean preflight.
  All node workers and canonical digests passed and all hard/defect methods fit T3, but the
  required full-sector-TI contingency exceeded its frozen 79,200-second window on nd-2 and nd-3.
  The discovery therefore closes before screen as
  `UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`; no sampler raw or physics result exists. The audit
  repair persists this legal aggregate terminal status without weakening any downstream gate.
- `011_q0_global_screen_diagnostic_20260721/`: isolated HARD2+EASY3 screen authorized after the
  full discovery runtime stop. The first immutable run remains a metadata-only gamma-libm
  conflict. The repaired fresh run `exp102_q0_screen_diagnostic_20260721_342dd5b` passed the
  three-node digest/runtime preflight, then completed and replay-validated 15 bias tasks plus all
  1280 T3 trajectories. Its verified terminal status is `UNRESOLVED_NO_HARD_COSET_PASS`: all five
  hard-coset candidates and all three defect-trace candidates passed 0/5 cells, so no method pair
  was selected. Completed metadata evidence includes an independent archive replay; this remains
  diagnostic only and grants no formal, held-out, or production authority.
- `012_hgp_collapsed_power_pt_20260722/`: local conda-12 feasibility evidence for the exact
  collapsed-HGP likelihood-power sampler. The short HP32 HARD2 probe motivated a frozen server
  screen but is not reusable measurement, confirmation, held-out, or formal evidence.
- `013_q0_hgp_global_screen_20260722/`: isolated HP32/HP64 plus MAM-IMH8 HARD2 diagnostic with
  adversarial P/U starts, exact small-HGP oracles, canonical frame binding, explicit collapsed-B
  slow-variable gates, immutable MAP artifacts, three-node runtime/digest consensus, and a
  two-phase preflight/local-attestation/measurement workflow. Fresh v2 completed all 384 sampler
  and two IS raw with full remote replay plus local audit. Terminal status is
  `UNRESOLVED_MAP_MIXTURE_FAIL`: HP64 passed 5/5 controls, HP32 passed 3/5, MAM passed 1/2 hard
  cells, and both HP/MAM pairs passed 0/4 agreement comparisons. Post-run raw shows that m8 MAM
  state changes were overwhelmingly within one logical sector because both minimum-weight anchors
  had the same logical signature. This remains diagnostic only and grants no formal readiness.
- `020_q0_collapsed_smc_v0_20260723/`: local exact-base collapsed-SMC feasibility diagnostic on
  the m8 hard cell. Small HGP enumeration verifies the collapsed target and reference/Numba
  transcript identity; all eight frozen N=128 populations then pass full seed replay and a
  raw-only audit. The terminal status is `LOCAL_COLLAPSED_SMC_WEIGHT_OR_GENEALOGY_NOT_VIABLE`:
  forced systematic resampling at all 63 nonzero bridge levels collapses final roots to 1--5
  (ESS 1.00--2.74), even though many individual incremental CESS values are near .9N. This
  rejects only that always-resample configuration, produces no q_top, and grants no remote,
  formal, held-out, or production authority.
- `021_q0_collapsed_ais_v0_20260723/`: fresh no-resampling exact-base AIS diagnostic on the
  same m8 hard cell. Its eight frozen N=128 paths pass deterministic seed replay and a separate
  raw-only audit which independently rebuilds the iid base, HGP/coset algebra, full AIS weights,
  gates, and report identities without importing the AIS engine. All eight nevertheless end with
  full-path ESS/N only `.0078125--.0100431` and a dominant final weight `.872760--1.000000`, so
  the terminal status is `LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE`. This distinguishes late
  path-weight concentration from the prior SMC genealogy collapse, rejects only CAIS64-B8-S1-N128,
  produces no q_top, and grants no remote, formal, held-out, or production authority.
- `022_q0_full_row_gibbs_v0_20260724/`: local exact collapsed-HGP full-row Gibbs feasibility
  screen on the m8 hard cell. Exact n=10/n=13 oracle tests, detailed balance/stationarity,
  reference/Numba transcripts, and the k=64 bit-63 boundary pass (`42` focused tests); the 24
  frozen P/U/legal-low-energy-L trajectories also pass full seed replay and a raw-only audit that
  does not import the sampler or runner. The terminal status is nevertheless
  `LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE`: P/L both reach a legal weight-63 state, while every U
  measurement remains at weight at least 248. With hard-coset dimension 832, the target-support
  upper bound `2**832*(.04/.96)**(248-63)` is only `1.3118273337331353e-05`. Thus this is not
  merely an all-character leave-return gate false negative; it is adversarial-init nonconvergence
  inside a bounded negligible target-mass region. It rejects only FRG-VE1 V0 and grants no remote,
  formal, held-out, or production authority.
- `023_q0_uniform_anchor_pt_v0_20260724/`: local uniform-anchored full-energy collapsed-B
  replica-exchange screen on the m8 hard cell. All 48 P/exact-K0-U/legal-low-energy-L trajectories
  have immutable raw and a separate bit-exact replay plus raw-only audit. Both UARE32-R1 and
  UARE64-R1 leave U at minimum measurement weight 247--262 while P/L agree near legal weight 63;
  the loose hard-coset support bound makes that observed U region negligible. The terminal status
  is `LOCAL_UNRESOLVED_UNIFORM_ANCHOR_TRANSPORT`; it rejects only these configurations and grants
  no q_top, remote, formal, held-out, or production authority.
- `024_q0_aux_stabilizer_v0_20260724/`: local auxiliary-stabilizer replica-exchange screen on
  the same m8 hard cell. Two frozen configurations, 32 and 64 replicas, use 48 total P/exact-K0-U/
  legal-low-energy-L trajectories, full bit-exact replay, and an independent pickle-free raw audit.
  P/L agreement does not rescue either candidate: U differs from both in weight, complete score,
  all 128 logical characters, and most B masks, while fixed-clock stability also fails. The
  conservative target-support bound is inconclusive at the U weights, so the terminal
  `LOCAL_AUXILIARY_STABILIZER_TRANSPORT_UNRESOLVED` result is a finite-budget convergence failure,
  not a negligible-mass or impossibility claim. It grants no q_top, remote, formal, held-out, or
  production authority.
- `025_q0_sector_bridge_feasibility_20260724/`: exact fixed-sector free-energy bridge algebra
  plus local m8 overlap diagnostic. V1's reverse-side exponent error is retained and invalidated
  for audit; V2 fixes it and passes the bidirectional small-HGP oracle, but its twelve-clock P/S
  bridge traces are almost entirely pinned and P has a 20 percent forward/reverse product gap.
  The terminal `LOCAL_FIXED_SECTOR_BRIDGE_OVERLAP_UNRESOLVED` result has no sector-mass or q_top
  claim and cannot authorize remote work.
- `026_q0_purity_wmc_feasibility_20260724/`: direct purity weighted-model-count feasibility. It
  confirms the correct one-copy `Z` and equal-logical-label two-copy `C` target, but the existing
  exact ternary-XOR elimination reaches width 67 for Z and 66 for C by the width-64 cap. This
  rules out only that encoding at the tested cap, not another exact contraction, a certified
  collapsed-B bound, or the posterior.
- `027_q0_collapsed_tail_bound_feasibility_20260724/`: directed-rounding collapsed-B factor-max
  envelope on the m8 hard cell. Exact small-code tests verify the envelope contains the true
  rational mass. On m8 it remains about `10^311.34` above the truth-free B=0 lower anchor after
  one row and requires width 25 at two rows, above the cap 18. The terminal
  `LOCAL_COLLAPSED_B_FACTOR_ENVELOPE_NOT_VIABLE_WITHIN_V0_CAP` rejects only that loose envelope;
  it produces no posterior, purity, q_top, remote, formal, held-out, or production authority.
- `028_q0_full_column_gibbs_v0_20260724/`: exact full-B-column (`2^24` state) collapsed-HGP
  heatbath. Its n=10/n=13 exhaustive conditional, detailed-balance, stationarity, hard-coset,
  and replay checks pass, but an outcome-blind m8 timing gate measures `.278952` seconds per
  column conditional. A minimum T1 trajectory has 245,760 such updates, so the frozen
  factor-two projection is `137111.403` seconds (about 38.1 hours) versus the two-hour cap.
  Terminal status is `RUNTIME_EXHAUSTED`; it creates no P/U/L raw, q_top, remote, formal,
  held-out, or production authority.
- `029_q0_crossfit_is_feasibility_20260724/`: historical proposal-only cross-fit diagnostics for
  the old MAM and LSI artifacts. They generate no fresh samples and deliberately establish no
  posterior, coverage, or launch authority; their sole purpose was to justify a separately
  frozen fresh-IID schedule rather than selecting a favorable old estimate.
- `030_q0_iid_is_preflight_20260724/`: outcome-blind local algebra/runtime preflight for a
  prospective three-proposal IID-MIS schedule. It verifies legal hard-coset draws and cross-density
  cost without reading a weight or sector statistic; it has no posterior or remote authority.
- `031_q0_iid_is_local_v0_20260724/`: fresh 49,152-draw, three-proposal IID-MIS diagnostic on
  the m8 hard sentinel. Raw-only replay passes, eliminating MCMC initialization and transport as
  explanations, but all primary importance views fail their frozen block-weight gates (minimum
  ESS `22.09--28.91 < 50`, maximum normalized weight `.1522--.1629 > .10`). Its terminal status
  is `LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_UNRESOLVED`; its apparent `.98--.993` collision values
  are diagnostics only, not q_top or posterior results. Full support and inter-proposal agreement
  do not certify an unobserved target tail, so this result grants no remote, formal, held-out, or
  production authority.
- `032_q0_bp_systematic_preflight_20260724/`: outcome-blind preflight for two independently
  ordered BP-guided systematic hard-coset proposals. Both exact coordinate bijections and
  densities replay, with construction near `.8` seconds and direct draws near `1.1` ms; fixed
  loopy-BP messages remain oscillatory, which is a proposal-quality warning rather than a
  density error. It reads no target weight, label, or purity and has no posterior authority.
- `033_q0_bp_systematic_iid_local_v0_20260724/`: fresh 49,152-draw component-provenanced
  BP-systematic IID-MIS test on the same m8 hard sentinel. Runner replay and the independent
  pickle-free raw analysis pass. Both BP sources individually meet their frozen ESS/maximum-weight
  gates and agree under the predeclared diagnostic tolerances, but the required equal mixture of
  BP-SYS-F64, BP-SYS-R64, and MAM-IMH8 fails (`min ESS=23.48 < 50`, `max weight=.14695 > .10`)
  through MAM-source blocks. Its terminal status is
  `LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_UNRESOLVED`. This rejects the frozen three-source schedule;
  it does not permit retrospectively dropping MAM to claim a BP-only pass, and it supplies no
  tail/normalizer bound, remote, formal, held-out, or production authority.
- `034_q0_collapsed_tail_depth2_feasibility_20260724/`: local exact-rational strict-tail
  feasibility probe for the same m8 hard sentinel. A depth-two, width-25 directed-rounding
  collapsed-B partition upper envelope completes inside its frozen resource gates (`3.14` seconds,
  2.50 GB peak RSS), but two fixed non-planted MAP-derived B marginals retain only `3.30805e-96`
  scaled lower mass against a total upper bound `3.11016e-11`. The resulting rigorous
  tail/retained ratio `9.40179e84` misses the frozen `.01` goal by about 87 decimal orders, so its
  terminal status is `DEPTH2_ENVELOPE_NOT_TIGHT_ENOUGH`. This rejects only this factorized
  envelope, not q=0 or all certified normalizer routes; it yields no posterior, q_top, remote,
  formal, held-out, or production authority.
- `035_q0_character_wmc_structure_feasibility_20260724/`: one-copy signed-character WMC
  structural probe on the m8 hard sentinel. Both raw-H_Z and ternary-XOR encodings reach
  min-degree induced width `378`; min-fill exceeds observed width `102` before its frozen
  120-second cap. It rejects only these exact-elimination encodings/orders and calculates no
  `Z_u`, posterior, purity, q_top, or formal result.
- `036_q0_trellis_wmc_structure_feasibility_20260724/`: independent linear-code trellis
  structure probe. The best of seven deterministic HGP-aware/Tanner orders has state exponent
  `584`, far beyond the frozen exponent-24 actionability cap. It rejects only the tested trellis
  representations/orders, not exact WMC or the q=0 posterior.
- `037_q0_houdayer_structure_feasibility_20260724/`: coordinate-only generalized-Houdayer
  structure probe using H_X plus a tensor logical complement. It establishes the pair-target
  algebra but shows a generic exact-uniform coordinate pair is one connected component; it is not
  MCMC or a transport result.
- `038_q0_houdayer_real_logicals_feasibility_20260724/`: frozen real low-energy pair probe for
  the tensor-logical HCA coordinates. All 200 P/L and L/L evaluations yield only whole-replica
  exchanges and zero new unordered pair states. This rejects that coordinate choice as a useful
  global transport mechanism, not all Houdayer bases.
- `039_q0_houdayer_reduced_logicals_feasibility_20260724/`: the sole pre-registered canonical
  reduced-logical coordinate counterfactual. P/L remains whole-pair only, but 102 of 120 frozen
  low-energy L/L pairs split into two components and create genuine new unordered pairs; this
  structural signal justified, but did not certify, an exact local pair-kernel test.
- `040_q0_houdayer_pair_runtime_preflight_20260724/` and
  `042_q0_houdayer_pair_runtime_rebind_20260724/`: outcome-blind local timing preflights for
  HCA-RHB1 (832 random-scan coordinate heatbaths per replica plus one complete-component HCA
  move per clock). The source-rebound report projects about `22.0` seconds per 128+1024-clock
  pair trajectory with twofold safety, so it permits a local diagnostic only.
- `041_q0_houdayer_pair_local_v0_20260724/`: fresh 32-raw local HCA-RHB1 adversarial-pair
  screen. Complete small-code stationarity, deterministic replay, and a sampler-independent
  no-pickle raw audit pass, but U/U fails: its normalized pair weight is `.1486354` versus
  `.0388554` for PP, every U/U trajectory has zero genuine new unordered-pair events, and each
  of its 1024 measurement HCA operations is only a whole-pair exchange. LL has 1,091 genuine
  events and agrees with PP/PL, showing that visible low-energy recombination does not solve
  exact-uniform-to-low-energy transport. Terminal status is
  `LOCAL_HOUDAYER_PAIR_TRANSPORT_UNRESOLVED`; it rejects only this kernel/budget and has no
  posterior, q_top, remote, formal, held-out, or production authority.
- `043_q0_collapsed_houdayer_structure_feasibility_20260724/`: a frozen local structural
  counterfactual for exact Houdayer swaps in the actual collapsed-B marginal, not an MCMC
  measurement. Small-HGP exhaustive checks verify B conversion, factor-pair invariance,
  involution, detailed balance, and stationarity. On the m8 hard sentinel, all 16 P/low-energy-L
  records, all 120 L/L pairs, and all 64 P/rank-complete-L controls have identical B masks despite
  distinct physical logical labels; the frozen low-energy logical directions lie entirely in A.
  The independent exact-K0 U/U pair has 284 differing B bits but a single component, hence only a
  whole-pair exchange. Terminal `COLLAPSED_B_HCA_NO_LOW_ENERGY_RECOMBINATION` rejects direct
  HP64-plus-collapsed-B HCA as optimization of the wrong slow variable. It creates no q_top,
  convergence, remote, formal, held-out, or production authority.
- `044_q0_bp_dominance_witness_feasibility_20260724/`: a local, no-sample structural check of
  whether the frozen three-component BP-systematic mixtures could be made into strict rejection
  envelopes. It scores 1,691 predetermined legal planted/logical/systematic-coordinate witnesses
  with the exact mixture density and directed Decimal rounding. The only available universal
  normalizer inequality, `Z <= (.96)^(-1600)`, makes the resulting rigorous witness lower bounds
  on `sup pi/q` tiny (`5.53e-63` forward and `2.54e-53` reverse), so terminal
  `BP_MIXTURE_REJECTION_ENVELOPE_WITNESS_INCONCLUSIVE` is explicitly not a BP pass. It exposes
  that a useful dominance certificate first needs the same missing tight global normalizer/tail
  bound; it grants no BP-only IID, rejection-sampling, q_top, remote, formal, held-out, or
  production authority.
- `045_q0_bp_imh_local_v0_20260724/`: immutable infrastructure-failed BP-IMH attempt. It wrote
  a 24-task manifest and one raw, then mixed a relative output path with an absolute receipt root.
  Terminal status is `INFRASTRUCTURE_FAILED_RELATIVE_OUTPUT_PATH`; the lone raw is explicitly
  forbidden and there is no receipt, report, sampler conclusion, or reuse authority.
- `046_q0_bp_imh_local_v1_20260724/`: fresh-seed exact BP-systematic independence-MH diagnostic
  on the m8 hard sentinel. Eighteen pre-raw tests include complete small-code stationarity and a
  full-label D2 counterexample; 24/24 P/U/distinct-L raw then pass exact runner replay and an
  independent 55,296-step no-pickle MH audit. Terminal status is
  `LOCAL_BP_IMH_TRANSPORT_UNRESOLVED`: P and L make zero real moves, while U makes only 1--3 burn
  and 0--2 measurement moves before all chains collapse to the same weight-62/P-label state.
  High observed acceptance counts in U are overwhelmingly accepted self-proposals. P/L
  `D2_norm=1`, so this exact kernel is sticky rather than globally mixed and grants no HARD2,
  remote, posterior, formal, held-out, or production authority.
- `047_q0_center_preserving_structure_20260724/`: truth-free dressed logical-XOR structural
  probe. The 127-move catalog is algebraically valid and rank 64, but its accessible signature
  ranks are only 4 from BASE and 1 from P; all frozen L starts have downhill routes to the same
  label and the optimistic full-rank scheduler bottleneck is `1.70e-10` expected accepts per
  direction. Terminal `LOCAL_CENTER_PRESERVING_STRUCTURE_NOT_VIABLE` rejects this common-basin
  collapse route without claiming q=0 impossible.
- `048_q0_random_full_column_local_v0_20260724/`: infrastructure-aborted first random-full-column
  wrapper. It used the reference mass-table path, stopped after one forbidden P raw, and is never
  continued or reused.
- `049_q0_random_full_column_local_v1_20260724/`: fresh 64+256-update exact random-scan
  full-B-column screen with P/U/L starts. All 12 raw replay, but every transport/agreement gate
  fails. In particular, exact A redraw creates logical-label activity while P/L B blocks remain
  almost frozen; U remains separated in B weight and likelihood. Terminal
  `LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_UNRESOLVED` is a short-screen failure, not a T1 result.
- `050_q0_full_column_map_bridge_structure_20260724/`: truth-free two-MAP-anchor bridge probe.
  The legal weight-62 anchors share a logical label and differ in six B bits across columns 11/17.
  Correct column order gives about `.03846` first-step conditional probability and `16.4` expected
  first departures over T1. This only justifies a fresh B-distinct T1 screen.
- `051_q0_global_move_independent_audit_20260724/`: independent algebra/raw-only audit of
  047/049/050. It reproduces every relevant hash, transition, state, label, weight, likelihood,
  gate, and bridge probability, preserving both failures and 050's narrow structural permission.
  It also records the incomplete transitive source identity in historical 049.
- `052_q0_random_full_column_t1_m8_20260724/`: frozen fresh T1 m8 contract with P, independent
  exact-K0 U, two B-distinct truth-free MAP, and eight low-energy B/logical-distinct S starts.
  It freezes 40 tasks at 2048+8192 clocks, full logical/B diagnostics, clean-archive binding, and
  three-node exact/runtime preflight. The immutable run from source `6fa489f` reached exact mass
  and transcript consensus, but all three replay-inclusive trajectory projections exceeded the
  frozen two-hour cap (`24701/24812/29871` seconds), so aggregate status is
  `RUNTIME_EXHAUSTED` and measurement raw count is zero. Independent audit SHA
  `817425db...75c6d` verifies the full evidence and decision. This is an implementation/resource
  failure before convergence testing, not a physical failure or impossibility result, and grants
  no m6, HARD2, formal, held-out, or production authority.
- `053_q0_random_full_column_streaming_preflight_20260724/`: exact memory-streaming successor
  preflight. The proposed streaming CDF and all portable transcripts agree across nodes, but the
  frozen legacy-byte comparison differs once per Linux node and the remote speed/runtime gates
  also fail. Terminal evidence is `CONFLICT` plus independent runtime exhaustion; no measurement
  raw exists.
- `054_q0_random_full_column_direct_block_preflight_20260724/`: exact positive-weight,
  fixed-`2^12`-candidate-block implementation of the full-column conditional. Complete `2^24`
  numeric checks, macmini/three-node portable transcripts, and the replay-inclusive runtime gate
  pass. This authorizes only a fresh m8 T1 diagnostic, not convergence or physics.
- `055_q0_random_full_column_direct_block_t1_m8_20260724/`: first fresh direct-block T1 contract.
  Its final immutable run passes the complete 054 preflight but its own 10-update estimator
  repeats fixed startup overhead when extrapolating, so it terminates at preflight as
  `RUNTIME_EXHAUSTED` with zero measurement raw. The failure is resource-estimator-specific, not
  a sampler result.
- `056_q0_random_full_column_direct_block_t1_m8_v2_20260724/`: runtime-corrected immutable m8 T1
  diagnostic from source `6933e319...`. Both three-node preflights pass and fixed `14/13/13`
  ownership completes all 40 raw. Primary and independent raw-only analyses agree on terminal
  `UNRESOLVED_DIRECT_BLOCK_T1_M8`: every family fails Rhat/ESS, and exact-K0 U remains at
  normalized weight `.097775` versus about `.03888` for P/M0/M1/S. P/U `delta q_top=.90374`
  despite thousands of visible U label changes. Audit SHA is `ada30d3c...b08e`. This is a frozen
  T1 sampler failure, not physical q_top or impossibility, and grants no m6/HARD2/formal/held-out/
  production authority.
- `057_q0_hgp_physical_pt_oracle_20260724/`: exact collapsed physical-p PT oracle and frozen local
  CPPT32 P/U T1 pair. Small-HGP cold-target, local/swap stationarity, p=.5 and portable k=64 tests
  pass, and the m8 table/runtime gates are feasible, but both T1 trajectories have zero round trips
  and retain large logical/B distribution disagreement. Independent raw-only status is
  `INDEPENDENT_RAW_ONLY_AUDIT_PASS_LOCAL_T1_PAIR_UNRESOLVED`. No remote work was launched; this
  rejects only CPPT32 at the frozen pair/budget and cannot independently confirm HP64.
- `058_q0_full_row_elimination_feasibility_20260724/`: exact 24-bit full-B-row conditional using a
  deterministic width-12 variable-elimination plan. Complete small-HGP conditional, detailed
  balance, stationarity and replay tests pass, and local m8 runtime is inexpensive. Exact frozen
  P/U/M0/S0 statistics nevertheless show that P/M0/S0 have less than `9.9e-6` union-bound chance
  of even one row move over 10240 cyclic updates, while U changes aggressively. An independent
  target-only elimination confirms this. Terminal interpretation is local conditional feasibility
  but standalone low-energy transport non-viability; no T1/remote/q_top/formal authority exists.
- `059_q0_hybrid_row_column_local_pilot_20260724/`: frozen local composition of one exact
  direct-positive B-column and one exact full-B-row heatbath per macroclock. All 16 P/U/M0/S0
  256+1024-clock raw replay and a sampler-independent transition/raw audit passes, but the
  necessary B-distribution gates fail: U remains at normalized B weight `.10823` and likelihood
  per factor `-11.2326`, versus about `.04/-5.1` for all low-energy families. The row block changes
  21--25 rows during U burn and then only 1--3 in measurement, so the proposed U-collapse plus
  low-energy-column-transport division of labor is falsified. No remote/T1/q_top/formal authority
  exists.
- `060_q0_multirow_joint_block_structure_20260724/`: pre-run, one-shot local structural screen for
  MR2/MR3/MR4 and row-column-cross exact collapsed-B blocks. The committed-source preflight,
  direct-perturbation scope reconstruction, independent bitset min-fill audit and focused tests
  are implemented, but no width report exists yet. A survivor would be only an HP64 contingency
  if Stage 3/4 fails; it cannot be the required orthogonal confirmer.
- `061_q0_next_stage_governance_20260728/`: completed read-only Stage-0 reconciliation. It
  inventories 1,472 dirty-root files without changing them, records 20 overlapping paths and
  seven byte differences, and separates method-internal, cross-method, cell and formal authority
  for validations 001--060. It creates governance evidence only, not sampler or remote authority.
- `062_q0_character_gate_calibration_20260728/`: completed local exact-IID/synthetic operating-
  characteristic calibration from source `c8642a0`. All five pre-registered common resource
  points fail, so terminal status is `CHARACTER_GATE_REDESIGN_REQUIRED` with independent audit
  PASS. The largest point passes both exact catalogs and the 511-character stress, but its
  688/4160-character simultaneous-coverage Wilson lower bound is `.97790 < .98`; no point reaches
  fresh confirmation. The report also records that a `.04` observed-character tolerance permits
  a `.08` frozen-estimator purity difference and gives no unobserved-character coverage. This is
  a gate-design failure, not sampler evidence or remote/formal authority.
- `063_q0_nishimori_auxiliary_calibration_20260728/`: pre-run physics-v2 exact Nishimori and
  negative-control calibration. It independently checks n=10/n=13 hard-coset posteriors, preserves
  known blind controls, measures omnibus/basis/nonbasis power, and freezes a fail-closed future raw
  schema requiring every planned fresh-IID disorder. It remains an auxiliary audit with no
  universal q_top-bias bound or confirmer authority; no one-shot report exists yet.
- `064_q0_hp64_resource_calibration_20260728/`: pre-run read-only validation-013 discrepancy and
  resource calibration. The science path independently recomputes HP64/MAM character statistics;
  the isolated resource path emits unselected timing scenarios and leaves strict full-grid totals
  null because m7, most p values and multi-code/disorder timing coverage are absent. No generated
  report selects a resource tier or authorizes remote/formal work yet.
- Held-out and production evidence do not exist. Their absence is an active production blocker.
