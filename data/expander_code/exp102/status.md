# exp102 status

**Q=0 HGP HARD-PAIR DIAGNOSTIC IMPLEMENTED / PRE-RUN -- formal work remains blocked**

On 2026-07-22 the user authorized fresh server testing of the isolated
`exp102.q0_hgp_global.screen.v1` diagnostic. The candidate HP32/HP64 sampler
analytically collapses one HGP block, performs likelihood-power replica
exchange on the exact marginal, and draws the eliminated block from its exact
conditional. The independent MAM-IMH8 mechanism uses a frozen full-support
multi-anchor independence proposal with the complete Hastings ratio. The
canonical diagnostic config SHA256 is
`163a5cc87486beabf453f3d4a57bc63f0c4e0b2f54619c60268ee7f0c9b2a341`.

MAM transport gates now count actual state changes, not merely accepted MH
decisions: an accepted proposal identical to the pre-step state is a self-loop.
The replayable sampler transcript is v2 and the enclosing screen MAP raw is v3;
both retain acceptance fields for diagnostics and add burn/measurement
state-change masks and counts.

Auxiliary preflight/runtime trajectories and IS probes now have namespaces
disjoint from measurement trajectories and frozen screen IS streams; an
exhaustive fixture-identity trajectory/IS 63-bit seed regression permits only
same-case reference/Numba reuse, and the clean deployment identities must be
rechecked immediately before launch. P/U and
cross-method `Delta q_top` gates also use a shared-character paired
finite-population SE plus side-specific delete-one trajectory jackknives,
rather than adding two marginal character errors and accidentally weakening
the consistency gate.

The frozen server scope contains 384 trajectories: HP32 and HP64 each run
`HARD2+EASY3` with 16 planted and 16 exact-uniform starts per cell (320 total),
while MAM-IMH8 runs only the two `HARD2` cells (64 total). MAP artifacts and
50000-draw IS diagnostics are also limited to HARD2. Local probes showed that
adding deterministic m3 near-MAP shells did not address the relevant basin:
the planted start still accepted 0/8192 measurement proposals, the uniform
start accepted only 1.5%, both weight-24 solves hit 180 seconds, and the m5
primary solve also exceeded 180 seconds. Those known-failing EASY3 MAP tasks
and near-MAP shells are excluded rather than optimized further.

Small-code production-bound enumeration now directly exercises the reference
B categorical and conditional A reconstruction paths on the n=10,k=4 and
n=13,k=1 HGPs for zero/nonzero syndrome and p=.04,.10,.25. Complete fixed-order
sweeps satisfy row normalization and stationarity, while each individual B
heatbath block satisfies detailed balance, all within `1e-13`. Both public
samplers independently bind the observable frame to the model before using RNG.
Raw replay also reconstructs the collapsed B trace and gates every B bit,
row/column parity, 64 frozen dense characters, `|B|` and `L(B)`; both catalog-
average and per-character P/U and cross-method differences must pass. The
workflow binds one immutable MAP artifact per hard cell, exact task/IS sets,
source/archive/config/registry identity, structure-only staging validation,
final bit-exact replay, and the real execution topology: distributed nd-2/nd-3
generation followed by a frozen 91-worker nd-3 analyzer. Runtime projection
counts those stages separately before applying the independent factor-two
safety margin.

No 013 server preflight, measurement raw, terminal report or sampler pass
exists yet. The maximum possible outcome is only
`DIAGNOSTIC_HARD_PAIR_FOUND`; EASY3 has no independent-method confirmation,
and one sentinel disorder cannot certify any `(m,p)` parameter point. This
contract cannot create `READY_FOR_FORMAL`, `FROZEN_HELD_OUT_PASS`, a
publication result or any of the 6144 production tasks.

The first outer launch attempt from clean source
`7654bcced23688705f396695370661199b81648a` exited before creating a run root,
orchestrator log, stage marker or raw because `nd-0` has no `screen`
executable. It launched no preflight or sampler and is infrastructure-only,
not a scientific run. The successor source replaces only the `nd-0` outer
control persistence with guarded `nohup` + `setsid`; all `nd-1`/`nd-2`/`nd-3`
scientific stages remain isolated `screen` jobs. A fresh deployment/run is
required rather than retrying that source in place.

The fresh successor attempt from
`df97fb5a7d38543beb515444b5692427dd28cc41` proved that the guarded `nd-0`
orchestrator survives SSH detachment and launches the first `nd-1` screen, but
the stage bootstrap exited before creating a run root, immutable marker or raw.
The bootstrap log identified a Bash 4.2 portability bug: under `set -u`, the
zero-prerequisite `build-schedule` path expanded an empty array. This is also
infrastructure-only and produced no preflight or sampler evidence. Its guard,
log and deployment remain audit evidence and are not reused. The successor
source guards every possibly empty array expansion and adds an executable
zero-prerequisite wrapper regression; another fresh deployment/run is required.

Read-only follow-up also found that the Linux node epochs are unsynchronized
(approximately nd-1 `+204 s`, nd-2 `+298 s`, nd-3 `+2 s` relative to nd-0;
chrony on nd-0/nd-1/nd-2 reported `Not synchronised`). This invalidated the old
cross-node epoch deadline assumption but produced no preflight or measurement.
The fresh successor protocol is now
`exp102.q0_hgp_global.screen.schedule.v2`: only nd-0
`CLOCK_BOOTTIME + boot_id` authorizes the 6/8/22/24-hour boundaries, every stage
launch and SUCCESS acceptance is rechecked with `now >= deadline` failing, and
measurement must reuse the original preflight clock anchor. Compute-node and
macmini epochs are retained only as finite diagnostics; runtime tier selection
uses the fixed worst-case hour-8 control origin and cannot read them or any
physics outcome.

The successor evidence protocol also records each nd-0 SUCCESS acceptance and
publishes ordered preflight/measurement phase manifests. A terminal package is
valid only jointly with the measurement acceptance manifest and the exact
control-derived set of 384 trajectory raw, two IS raw and all matching claims;
online publication and local pullback both reject missing, extra, symlinked or
hash-mismatched evidence.

**Q=0 DIAGNOSTIC SCREEN UNRESOLVED / PRE-PILOT — formal pilot blocked, production not started**

The fresh immutable run `exp102_q0_screen_diagnostic_20260721_342dd5b` completed the isolated
`exp102.q0_global.screen_diagnostic.v1` HARD2+EASY3 screen. Source
`342dd5bc0fb2c7694dbc58a8d0f2d92689c24991`, archive SHA256
`4a54ba28f3ee2add94e93dd052e4bda567d5e008691f84a098c21768b4fe11f3`, manifest SHA256
`2b8ab6d238d6319ea73c4c5da0ecf815a3d2e2ea28932dddc30bd40afe158b01`, and schedule-file SHA256
`f9aeccd95640a56fabe813796d0e1ce388cffa1bcccf2405a6bafcd913520832` are frozen. The three-node
preflight passed at T3; canonical digest was identically
`080b3170ca168dc3f237d22a4d18403eb2c0b0b2455e6d1e3ca876aae39c86a9`, including the exact
4096-value gamma SHA `a2c459ec9438e23f863c44528ac093c5b93d891b6a8bec0278b873fe47f2459a`.
All 15 bias tasks and all 1280 fresh measurement trajectories completed with no reuse and passed
identity, SHA, algebra, and bitwise raw replay.

The verified terminal status is `UNRESOLVED_NO_HARD_COSET_PASS`, with `selected_pair=null` and
terminal package identity SHA256
`0e0fb2f950eb609c984b29f5647321694c82f8f7a6810609fd1742d1472a990a`. Each of
`RC8-QC1/QC4/J08/J12/J16` passed 0/5 cells: all 25 method/cell summaries exceeded the absolute
P-versus-U `q_top` difference limit, every U family had `Rhat>1.05` and bulk ESS<400, and
`|delta q_top|` ranged from 0.06695 to 0.991999 against the 0.04 gate. `DT16/DT32/DT64` also
passed 0/5: all 480 defect-trace chains had zero fixed-clock D=0 observations and zero complete
leave-return excursions, so no conditional estimator existed. Thus both mechanisms failed within
the frozen T3 budget `(burn 8192, measurement 32768)`; the terminal name follows the prescribed
hard-coset-first decision order. This is a sampler-convergence failure within the tested algorithm
and budget, not an infrastructure failure, mathematical impossibility, or formal physics result.

A local conda-12 replay from the same verified archive independently revalidated every raw and
reproduced the same gates and terminal status. The two reports differ only in 62 derived
`core_seconds` values and 18 derived ESS values by at most 4 ULP (maximum absolute difference
`1.82e-12`); raw replay remains exact. Completed metadata evidence and its fail-closed verifier are
in `validation/011_q0_global_screen_diagnostic_20260721/completed_run_evidence/`. The predecessor
`exp102_q0_screen_diagnostic_20260721_5e1f5aa` remains permanently archived as
`CONFLICT_CROSS_NODE_GAMMA_LIBM`; none of its 15 bias raws was reused. There is still no
`READY_FOR_FORMAL`, formal sampler, held-out campaign, `FROZEN_HELD_OUT_PASS`, or production
authorization. The five explicit blockers remain `NO_T_VS_2T`, `NO_FRESH_HARD2_CONFIRMATION`,
`NO_CONF17_RES6_GAP8_SMALL6`, `NO_TI_OR_REVIEWED_INDEPENDENT_ORACLE`, and `NO_HELD_OUT`.

The reviewed successor to the exhausted PT-v2 and PA routes is now implemented under
`exp102.q0_global.discovery.v1`; its frozen contract is `GLOBAL_DISCOVERY_CONTRACT.md` and its
config SHA256 is `1d0a453f2bf8445ad6587c612c2eabb3049e76e2d73b59c230b8b1358b06e565`.
It adds deterministic low-weight logical catalogs, rejection-free hard-coset cluster Gibbs,
joint stabilizer-logical block heatbath, an independent fixed-clock defect-trace mechanism,
full-sector TI anchors for m3, exact small-code oracles, three-node digest/runtime consensus,
immutable 72-hour controls, and an independently replayed character-U-statistic analyzer.

The third immutable clean-source attempt, `exp102_q0_global_20260721_204b37d`, is the terminal
execution of this discovery contract. Source `204b37d8e00e7d11ffa2b6766b90d947892e179d`, archive
SHA256 `1583dce6b8bb81ad7780f323d21300b158ad435d710f3c0226b7b3028b8eb7f7`, manifest SHA256
`b69290798a11a3bf548483c6e223f96a64e0d9c7be0e48b89fa6e54a28a57ea3`, and schedule-file SHA256
`7874a0d967ba866d8834cf380b408947af614bdf3bec7b50c0f30fb4a332465c` are frozen. The local clean
archive passed 590 combined exp102+exp101 tests and post-test source reverification. All three
Linux workers completed with exclusive SUCCESS markers; their canonical digest was identically
`a3730d7380575976f88e35f5490b24a9b6949e3817b2fb3880775736cf2ad364`. WMC returned six
`INCONCLUSIVE_WIDTH` diagnostics.

Every hard-coset and defect-trace candidate was T3-eligible, with factor-two complete-schedule
projections of 1.307/2.441/2.055 hours on nd-1/2/3. The mandatory m3 full-sector-TI contingency,
however, projected to 78,705/116,275/251,241 seconds against the frozen 79,200-second window.
nd-2 and nd-3 therefore correctly reported `RUNTIME_EXHAUSTED`; worst-node consensus stops the
workflow before bias tuning or screen. No sampler raw, method freeze, HARD2 fresh run,
confirmation, resolution, or TI anchor exists. Evidence and an independent verifier are in
`validation/010_q0_global_runtime_exhausted_20260721/`. The combiner audit now persists this legal
terminal status instead of throwing, while every downstream stage still requires aggregate PASS;
the run was not retried and no threshold, safety factor, tier, or deadline changed.

The full-range conclusion is `UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`, not `IMPOSSIBLE` and not
a parameter-specific physical failure. The two preceding infrastructure-only preflights remain
permanently archived in `validation/008_q0_global_preflight_portability_20260721/` and
`validation/009_q0_global_runtime_gate_separation_20260721/`. There is no `READY_FOR_FORMAL`, new
formal sampler, held-out campaign, `FROZEN_HELD_OUT_PASS`, or production authorization. PT/PA/global
discovery raw remains barred from formal merge/freezer/production. Any later reduced range or new
global campaign requires a separately reviewed scientific contract and fresh tuning/held-out; it
cannot be presented as completion of this discovery.

The reviewed successor to the exhausted PT-v2 route was executed under the isolated
`exp102.q0_pa.discovery.v1` contract. Worker source
`f0dff0f8d3e055227b75c999a73c751e2a576768` used archive SHA256
`57811c43662b379524fb4f5099346f042d5577cc1e2c69a31299a11fd9c01324`. The nd-1/2/3 canonical
digest was identical (`f4ed9fff7512f8995a4f70c60072c1bba054aaf75e0440a4d00545880305f478`),
and the authoritative nd-2 runtime report passed all four gates: slowest m8 kernel
`56.91 us/particle-sweep`, startup `1.80 s`, maximum population projection `0.373 min`, and
factor-two full-schedule projection `1.064 min`.

All four transport-autopsy tasks passed identity and bit-for-bit parent replay. All classified
`INCONCLUSIVE`, because the required outbound phase-conditioned attempts fell below 200 near the
hot end. D0/D4 on m6 observed 3/5 certified hot updates but zero returns; both m8 tasks observed
zero hot updates. Thus the autopsy confirms that high aggregate edge rates did not provide enough
conditioned transport evidence, but it cannot assign one of the three causal labels.

The complete 64-population PA hard screen produced zero passing methods. Every population failed
the frozen genealogy gate: median final family ESS was about 1 and median surviving initial
families was 1--2, versus required 8 and 16. Some B96 populations also failed CESS and one maximum
particle-weight gate. Therefore `C192-2`, `B96-1`, `B192-1`, and `B96-2` all failed both hard cells.
The zero-pass branch is final `EXHAUSTED`: `B384-2` rescue is forbidden, confirmation/resolution
manifests were not created, and neither `READY_FOR_FORMAL` nor `FROZEN_HELD_OUT_PASS` exists.
Discovery raw remains barred from every formal merge/freezer; the formal versions remain
`exp102.q0_pt.v1 / exp102.scan.v1`.

The post-run analyzer audit fixed two evidence-only portability defects without changing raw or
any numerical gate: NumPy 2.3.4 versus 2.4.1 differed by up to 2 ULP in stored `ladder_p` and up to
4096 ULP (`5.68e-14` absolute) in accumulated log-Z replay, and autopsy evidence paths were not
JSON serializable.
Discrete transcripts remain exact; derived float replay is bounded at 8 ULP for ladders, 64 ULP
for non-cumulative values, and `32*G` ULP for cumulative log-Z. Local and remote analyzers agree
on the zero-pass outcome.

Clean source `da69528b43f4a9d1635083c21d713ba63ccec4ab` passed the three-node PT-v2
preflight and completed the frozen screen plus transport stages. The 45-cell screen passed D0,
D2, D3, and D4 at 9/9; D1 passed 8/9 and was rejected by one sub-0.20 swap edge. The 24-cell
transport stage then tested those four ladders at `S=4,16,64` on both hard cells. All 12 candidate
groups passed their long-run swap/hot-logical/residual gates (group minima for swap rate were
0.156--0.392), but all failed transport: across 96 instance trajectories only 13 ever received a
hot-rung update, there were 27 such visits, and there were zero uncertified, certified, or
sector-changing round trips.

Every `S=64` candidate has at least one instance with zero hot-updated visits, so the frozen
conditional rule does not permit `S=128`. The PT-v2 route therefore stops before the 17-cell
confirmation panel. It produced no primary/backup pair, formal v2 config, formal pilot, held-out,
freezer, task plan, or production run. The formal contract remains `exp102.q0_pt.v1` for the
exhausted historical pilot; discovery raw remains design evidence and is rejected by the formal
pilot path. The hardened analyzer independently verifies the exact NPZ set against node raw
manifests, control and LPT ownership hashes, source archive identity, stage fingerprints, statuses,
and exclusive SUCCESS markers before recomputing every counter and gate.

The independent registry, bit-identical reference/Numba hard-coset q=0 PT, net-transport
diagnostics, task identity/resume, fail-closed aggregation, publication loader, and pilot cell
runner are implemented. The first fake-Numba `R=8` ladder completed but failed all 576 cells; its
partial `R=12` successor was stopped after the full-round Numba replacement made that source SHA
obsolete. That history is audit-only; the clean-SHA ladder search described below supersedes it.
Held-out certification and the 6144 production tasks have not run, so no threshold curve or
scientific result is claimed.

Production requires `engine=numba`; the reference engine is an oracle only. The full-round Numba
kernel is bit-identical through the `k=64` boundary and gives about 177x--196x speedup in local
benchmarks. The PT-v2 implementation plus hardened evidence analyzer passes 77 exp102 tests and all
365 exp101 regressions locally. The discovery source passed the then-current exp102 suite on all
three nodes; its Linux PT-v2 digest was
`38f29fe037bcce399883b6f6d20b4500f54ba11e94ea5e8b98b586e8e402f659` everywhere.

The clean full-round source `bbe72da` passed three-node preflight and produced 10,752/10,752
ordered ladder cells. A post-run audit found that ladder/gamma had incorrectly inherited the
rounds-stage character-trace gate. This is now fixed, but the raw counters independently confirm
that m=4..8 still fail the actual ladder requirement at the maximum `(p_hot=0.49,R=64)` candidate:
only `93/96,89/96,85/96,84/96,87/96` cells pass swap/hot/residual. Under the frozen policy, those m
values must stop rather than proceed to gamma/rounds/held-out. No `FROZEN_HELD_OUT_PASS` exists.
Resuming requires an explicitly reviewed pilot-contract change, such as expanding R or changing
the ladder family, followed by a clean-SHA pilot rerun.

On 2026-07-20 the user approved appending `(p_hot=0.49,R=96)` and then `R=128` after the original
21 ladder pairs. Clean source `2b01d9dcb463ec47a1b30202fc9105430b95e18c` passed three-node
preflight; all nodes produced Linux smoke digest
`b9a5c8b22d8b2421723705b1567b825a5a1775a8efd20748e884436f8bee959f`. Run
`exp102_pilot_20260720_2b01d9d` completed all 13,056 planned ladder cells through the conditional
R128 attempt, with hashes verified locally against 13,270 remote files. Fresh merge-select chose
m=3 at `(0.45,64)` and m=4 at `(0.49,96)`, but the maximum R128 candidate left m=5..8 at
`94/96,94/96,93/96,94/96`; every failure was the `p=0.04` minimum swap-rate gate. Those sizes are
therefore `EXHAUSTED` under the frozen 23-pair policy. Gamma, rounds, held-out, freezer, task plan,
and production were not started. Resuming requires an explicitly reviewed new pilot/config
contract; lowering gates, hand-writing a freezer, or launching a reduced production is forbidden.
