# exp102 status

**Q=0 HGP HARD-PAIR DIAGNOSTIC V2 UNRESOLVED -- formal work remains blocked**

The immutable v2 run `exp102_q0_hgp_screen_v2_20260722_4d134ee` completed the
authorized `exp102.q0_hgp_global.screen.v2` diagnostic. It used source
`4d134ee7ca25125d341eb11cbfa34d6856514101`, archive SHA256
`ad72d2c7039192be721b87ce7c96c5da577af05acd37cacd9167e26a773d9027`,
source-manifest SHA256
`5bafae76b06ff46557ae8315bb281a42256e7e4e50ed2e9dae868695114b8ff8`
and config SHA256
`38092ec030f6c283f163c0ddb3eed612aa850c76ce34f130520522646fa883dc`.
The verified terminal status is `UNRESOLVED_MAP_MIXTURE_FAIL`, not an
infrastructure conflict and not a mathematical impossibility.

All three Linux preflight workers reached exact full-payload consensus and
selected `T3=(burn 8192, measurement 32768)`. The full and portable aggregate
payload SHAs were
`dfae05939b89af92c9ba7f758933dd99230f233d4db17b0aaae06414b11f1bd4`
and `4d1704e2efdb0f105e31aed552ca83587a1d47e3d3d27de7f82b5a0e8bb44e8d`.
The mandatory macmini conda-12 preflight audit was `PORTABLE_PASS`: all
declared discrete fields and four MAM acceptance-decision transcripts agreed
exactly. The 12 full-payload mismatches were exactly the preregistered
nonportable `log_q`, `log_acceptance` and IS floating fields. Re-enumerating
the actual deployment identities produced 31682 unique formal RNG streams,
all disjoint from auxiliary streams.

The fixed ownership run completed 384/384 fresh sampler trajectories and 2/2
frozen IS diagnostics without migration, replacement, retry or resampling.
The nd-3 full analyzer and an independent local audit validated all 386 raw
files, their identities, hashes, algebra and declared transcript projections.
The measurement acceptance-manifest SHA is
`42e12338dac640b725728f25c46b4d853a23e02392b2c9f2471f519ffcf5bba1`;
the joint-terminal SHA is
`7e9bd8d7efb657649c4a0b4f0d146b72063d4584b291479a49c805e6834ab4f1`;
the terminal-package identity is
`233e31e599180153f979a30dc971e8e8128be64505fd0572d68bc1ae87a64041`;
and the local terminal attestation is
`386e8a0eeadb5c24b376014b522dec36322456abf3b0d636c1ad16cc7681c755`.

`HP64` passed all 5/5 frozen cells and is the clear promising candidate.
`HP32` passed 3/5: `m03_c00,p=.10` missed one B-character absolute threshold
by only `0.0004396` while its uncertainty gate and all other B diagnostics
passed, but `m05_c00,p=.10` clearly retained a slow B mode (`U` maximum Rhat
`1.1552`, minimum B ESS `327.0`, pooled Rhat `1.1172`). `HP64` removed both
failures. On the two hard cells HP32 and HP64 also agreed closely:
`q_top=0.14575/0.14587` for m6 and `0.91356/0.91317` for m8. These are two
ladder sizes of one mechanism, not independent confirmation.

`MAM-IMH8` passed only 1/2 hard cells. On `m08_c06,p=.04`, its P family had
Rhat `1.06088` and minimum bulk ESS `379.74`; the targeted B projections were
also slow in both families (maximum Rhat `1.08245/1.05662`, minimum ESS
`275.33/361.16`, pooled Rhat `1.05568`). Sixteen B characters differed between
P and U, with maximum mean difference `0.05157`. In contrast, its ordinary
transition counters looked healthy: every trajectory changed state at least
520 times during burn and 1947 times during measurement, with state-change
rate at least `0.0594`. This is direct evidence that generic acceptance and
state-change counts did not measure the relevant global transport.

The cross-mechanism gate failed all four HP64/MAM family-cell comparisons.
For m6, MAM estimated `q_top=0.16241` versus HP64 `0.14587`; the absolute
difference `0.01647--0.01660` was below `.04` but about 30 standard errors and
therefore failed the preregistered consistency inequality. For m8, MAM gave
`q_top=0.99273` versus HP64 `0.91317`; both P and U differences were about
`.0795`, failing both the absolute and uncertainty gates. Weight and aggregate
D2 gates alone would not have exposed these discrepancies.

Post-run raw forensics explains why a longer identical MAM run is not the
default remedy. The two distinct minimum-weight m8 MAP anchors have the same
all-zero 64-bit logical coordinate. Across the 16 P trajectories, 39899
measurement state changes contained only 330 logical-label changes (`0.827%`);
the 16 U trajectories had 40735 state changes but only 288 logical-label
changes (`0.707%`). A typical chain visited only three logical labels, and all
32 ended with the same label as their first measurement state. Proposal
components with logical flip probabilities `.08`, `.25` and `.5` had zero
accepted proposals within each family's 524288 aggregate attempts. The `.02`
component had 11 accepted proposals per family, but these produced only 4 P
and 7 U actual same-sector state changes and no logical-label change.
The apparent MAM acceptance was dominated by same-sector moves around anchors
that lacked logical-signature coverage.

The initialization red-team remains important. All five frozen syndromes have
nonzero weights (`83,160,39,58,125`), so the physical all-zero bit string lies
outside every target hard coset. If one shifts explicitly by the planted error,
`x=e xor epsilon_true`, then `x=0` is exactly the existing planted `P` start,
not a new initialization. The 16 P trajectories share that legal state but use
independent RNG streams; the 16 U trajectories use independent exact K=0
hard-coset states. Replacing P/U by one common zero start would therefore
either sample the wrong support or hide initialization memory; it would not
establish convergence. Future diagnostics should retain P/U and add legal,
deterministically signature-stratified starts rather than making every chain
identical.

The next meaningful development target is an independent confirmer with
preregistered logical-signature coverage: for example, deterministic
logical-sector-constrained anchors or a genuinely different reverse-collapse
oracle. Its viability gate must count accepted cross-signature moves and
logical-character mixing by proposal component, not total acceptance, total
state changes or global IS ESS. Merely extending MAM T3, changing its start to
zero, or treating HP32/HP64 as independent confirmation would optimize the
wrong quantity.

This run certifies no `(m,p)` point. `EASY3` still lacks an independent method,
each panel entry is only one sentinel disorder, and no fresh T/2T,
CONF17/RES6/GAP8/SMALL6, formal 96-cell tuning or 448-cell held-out campaign
exists. There is no `READY_FOR_FORMAL`, `FROZEN_HELD_OUT_PASS`, publication
result or authorization for any of the 6144 production tasks.

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
