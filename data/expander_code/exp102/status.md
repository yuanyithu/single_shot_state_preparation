# exp102 status

**2026-07-30 VALIDATION 066 TERMINAL -- LOCAL DELIVERY GATE CONFIRMED;
PROJECT STILL BLOCKED BEFORE REMOTE**

Fresh local-only validation 066 completes from clean source
`bc47ae26dd26203f2b9c902feca2a10ea797c798`.  Its five frozen selection
points have the validity/outcome sequence `INVALID / FAIL / FAIL /
INCONCLUSIVE / PASS`; the first point is recorded as `INCONCLUSIVE` with
`NO_FINITE_CALIBRATED_MULTIPLIER`.  The selected common point is 32 IID
multinomial trajectory groups x 16384 independent draws per group with calibrated multiplier
`4.809673164164152`; these are not MCMC chains, clocks or ESS.  A fresh
seed namespace then confirms that same point as `PASS`.  The terminal status
is `LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED`.  The report JSON
self-SHA is `d255c67ee0a91985e933ccea8a9616c63e832e37c19cd16dc7eb5e35f05e5a0a`;
its complete file SHA is
`f11a3eb137793ce2bbe43734db82240cde45bafbdd57a2a1e6f97d520dad6ed8`.

The independent auditor regenerates the multinomial histograms, full-label
collision `q_top`, `D2_norm`, delete-one arrays, group-wise uncertainties,
calibration multiplier, decisions, Wilson bounds, selection and confirmation
from frozen seeds.  It passes as
`INDEPENDENT_AUDIT_PASS_LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED`,
audit JSON self-SHA
`485b789c3a86893662241ab0e529358fedde18b695c3514233adc492236261b3`;
its complete file SHA is
`3975de5eb1d9cebcc467efdd67d956dcfa4b98e4c3b205011a549d6cf8d7822c`.
The compact 4,372,205-byte report stores SHA receipts rather than persistent
trial raw.  Replay is bit-exact only in the frozen same-environment contract
(NumPy 2.4.1, `default_rng`, PCG64); it makes no cross-version portable-RNG
claim.
The post-run conda-12 exp101+exp102+066 regression is `1090 passed, 4
warnings`; all four warnings are pre-existing deprecated-alias or macOS fork
warnings.

This result confirms only the operating characteristics of the local
full-label `q_top`/`D2_norm` comparison gate.  Both deliberately common-wrong
`EXPECTED_KNOWN_BLIND` controls have true `D2_norm=.0625` yet candidate PASS
rate `1.0`, as preregistered.  This is an empirical common-wrong blind spot,
not only a theoretical warning.  Therefore 066
does not prove MCMC mixing, transport, target-basin or unvisited-tail coverage,
or correctness of any posterior estimate.  Legal adversarial initializations,
transport/Rhat/ESS/burn-crossing vetoes and an orthogonal confirmation method
remain mandatory.

The independent project state is `BLOCKED_BEFORE_REMOTE`.  Four blockers
remain exactly: `LARGE_K_ORTHOGONAL_CONFIRMER_PORTFOLIO_UNFROZEN`,
`FUTURE_SCHEMA_RUNTIME_COVERAGE_INCOMPLETE`, `CAMPAIGN_BUDGET_UNAPPROVED`, and
`STAGE3_MULTI_COMPARISON_MULTIPLICITY_UNFROZEN`.  Validation 066 does not
authorize m3 anchors, any remote sampler run, formal tuning, held-out work or
production.  There is still no certified cell or `(m,p)`, `READY_FOR_FORMAL`
or `FROZEN_HELD_OUT_PASS`.

**2026-07-28 NEXT-STAGE STAGES 1--2 TERMINAL -- CHARACTER-GATE REDESIGN
REQUIRED; NO REMOTE OR FORMAL AUTHORITY**

The local-only Stage-0--2 work in validations 060--065 is complete.  It is not
waiting on a server: no new remote measurement was launched.  Stage 1 stops
fail-closed at validation 062, whose five pre-registered common operating
points all fail.  At the largest point (32 trajectories, 16384 draws each),
the exact logical and collapsed-B catalogs and the 511-character logical
stress pass, but the 688-character B and 4160-character logical stresses miss
simultaneous coverage: the minimum one-sided Wilson lower bound is
`.9779025636 < .98`.  The independent audit reproduces terminal status
`CHARACTER_GATE_REDESIGN_REQUIRED`.

This is the primary blocker.  A maximum observed-character difference of
`.04` supplies at most a `.08` bound on the frozen catalog's mean squared
characters, and it says nothing about unobserved characters.  It is therefore
not a general `.04` bound on the delivered `q_top`.  The successor gate must
make direct `q_top` equivalence primary, retain full-label distribution/D2
checks independently, and use maximum-character checks as slow-mode
diagnostics rather than claimed coverage of an unobserved large-k tail.
Adding trials merely to move `.9779` above `.98` would not repair this mismatch.

Validation 063 independently enumerates 30 exact controls and remains
`NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT`: 14 correct-posterior groups at
`N=2048` miss the frozen `.01` simultaneous-equivalence precision target.  Its
original v2 auditor failed first on the legacy English-prefix mismatch
`equivalence gate failed` versus `equivalence power failed`; that fail-fast
result did not establish complete numerical agreement.  Validation 065 then
recomputed the complete immutable payload and found 11 discrepancies above
`2e-13` (maximum `.03400704`), all caused by floating `argmax` choices among
mathematically tied MAP sectors:

- `p=.04`, syndrome `05`: report label `0`, oracle label `15`;
- `p=.04`, syndrome `06`: report label `0`, oracle label `5`;
- `p=.10`, syndrome `03`: report label `10`, oracle label `0`.

The required interpretation is exactly `full_payload_match=false` and
`terminal_gate_invariant=true`: all 14 structured terminal failures remain
identical, but invariant terminal status is not an audit pass.  Validation 065
therefore persists
`CONFLICT_INDEPENDENT_NUMERICAL_RECOMPUTATION_MAP_TIE_SEMANTICS` (audit SHA
`5d49532e...13ab`), and its separate verifier passes only the recording of that
conflict as `INDEPENDENT_VERIFICATION_PASS_OF_RECORDED_MAP_TIE_CONFLICT`
(verification SHA `03cb4d1e...c5cc`).  Nishimori remains an auxiliary
diagnostic without a universal `q_top`-error or confirmation guarantee.

Validation 064 independently replays the old HP64 evidence and confirms that
m8 `.91317/.99273` compares HP64 with MAM, not HP64 P with U; the m6 P-family
HP64/MAM difference is `.0165964`, or `30.5903` paired SE.  Its 72-row resource
matrix remains scenario arithmetic only: empirical coverage lacks m7, most p
values and cross-code/disorder timing, so strict totals and resource selection
are `null`.  Validation 060 closes Stage 2 with MR2 as the sole structural
survivor (width 25, 512 MiB single-table lower bound).  MR2 is only a suspended
same-family contingency after a genuine HP64 Stage-3/4 failure; it cannot be
the missing large-k orthogonal confirmer.

Consequently Stage-3 m3 anchors, the easy 128-disorder block, m6/m8/HARD2 and
all formal tuning, held-out and production work remain unauthorized.  There is
still no certified cell or `(m,p)`, `READY_FOR_FORMAL`,
`FROZEN_HELD_OUT_PASS`, or production result.  Before any remote launch, a
fresh delivery-aligned gate contract, complete resource coverage and a
pre-frozen portfolio of no more than two large-k orthogonal-confirmer concepts
are required.  Old raw may
not be extended or reused, adversarial P/U starts may not be replaced by a
common P/physical-zero start, and no threshold may be relaxed post hoc.

**HYBRID ROW-COLUMN VALIDATION 059 TERMINAL -- LOCAL B-TRANSPORT NECESSARY
GATES FAIL; NO REMOTE OR FORMAL AUTHORITY**

Validation 059 freezes `HRC1-C24-DPB12-R24-VE12`: every macroclock applies one
uniform exact direct-positive B-column heatbath followed by one uniform exact
full-B-row elimination heatbath, then draws A|B exactly at observation clocks.
Complete small-HGP matrices verify the ordered clock preserves the strict
collapsed target, and focused/related tests pass.  Source
`1e9097644dbed0ccb6cd61da1dc80d57413ce4bb` then runs the pre-registered local
P/U/M0/S0 x4 panel at 256+1024 clocks; all 16 raw and full replays complete.
The post-run focused and complete exp101+exp102 regressions are `107 passed`
and `1033 passed, 4 existing warnings`, respectively.

The hypothesized division of labor is falsified.  P/M0/S0 late normalized B
weights are `.03922/.04065/.04159` with likelihood per factor
`-5.2297/-5.1555/-5.0977`; U remains at `.10823/-11.2326`.  U/P differences
are `.06901` in B weight, `6.0030` in likelihood and `.04992` in all-B-bit mean
square distance.  Each U chain changes 21--25 rows during burn but only 1--3
rows during measurement, showing it makes roughly one early row sweep and then
freezes in the wrong high-energy basin.  Zero of four U endpoints pass the
loose `.065/-6.5` collapse gate, and U still has first/last likelihood drift
`.5695`.

Primary status/report/raw-set SHA are
`LOCAL_HYBRID_B_NECESSARY_GATES_FAIL` / `2f25aa7c...873ba` /
`db6a303e...cd88`.  A sampler-independent raw auditor reconstructs every
column/row transition, cached syndrome, state/B block, label, weight,
likelihood, counter, seed, summary and gate as
`INDEPENDENT_RAW_AUDIT_PASS_LOCAL_HYBRID_B_NECESSARY_GATES_FAIL`, SHA
`443d461d...b7c`.  Runtime and replay pass, so this is a sampler transport
failure rather than infrastructure failure.

No nd-2/nd-3 task is launched.  The 059 raw cannot be extended, pooled or
reported as q_top.  The failure is not a physical parameter-point result or
`IMPOSSIBLE`; it rejects this fixed one-row/one-column composition and budget.
A meaningful successor must coordinate multiple rows/columns or otherwise
cross the collapsed-B basin barrier.  Common P/zero starts remain forbidden,
and this collapsed mechanism still cannot provide independent confirmation.

**FULL-B-ROW ELIMINATION VALIDATION 058 TERMINAL -- LOCAL CONDITIONAL
FEASIBLE, STANDALONE LOW-ENERGY TRANSPORT NOT VIABLE**

Validation 058 implements the exact collapsed full-B-row conditional with a
state-independent variable-elimination plan.  The frozen m8 graph has induced
width 12 and largest factor 8192.  Complete n=10/n=13 HGP enumeration verifies
the conditional, normalizer, detailed balance, complete-sweep stationarity and
PortablePrng/cache replay (`20 passed`).  The 128 MiB mass table builds in
`.316s`; measured row cost `.01291s`, incremental peak about 17 MiB and
factor-two T1 projection `264.39s` all pass the local resource gate.  Primary
feasibility report status/SHA is `LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE` /
`0f99bba4...172da`.  The final exp101+exp102 regression is `1020 passed` with
four existing warnings.  This status means only exact and computationally
feasible.

The scientific result is negative for the standalone kernel.  On the frozen
legal P/M0/S0 low-energy states, median conditional entropy is zero, median
expected Hamming change is about `1.2e-21--1.9e-21`, minimum row self
probability is at least `.9999999926`, and a sampled complete sweep changes
zero rows.  By contrast, exact-K0 U has median entropy `2.619` bits and median
expected change `11.645` bits; its first sweep changes all 24 rows/294 bits.
An independent target-only elimination reproduces the expected changes within
`7.8e-13` and bounds the probability of even one low-energy row change over
10240 cyclic updates below `9.9e-6`; audit status/SHA is
`INDEPENDENT_TARGET_AUDIT_PASS_LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE` /
`3845759b...bd1`.

Therefore no standalone T1 screen or remote job is launched.  The row block
may only be considered later as an exact U-collapse component in a fresh mixed
kernel whose other move demonstrably transports among low-energy B basins.
It is not a convergence result, q_top, parameter-point failure, `IMPOSSIBLE`,
or formal/held-out/production authority.  It also shares the collapsed-B
identity with HP/direct-column methods and cannot be the missing independent
confirmation.  Physical zero remains illegal for the nonzero syndrome;
shifted zero remains P, and P/U/MAP/S must remain adversarial starts.

**COLLAPSED PHYSICAL-P PT VALIDATION 057 TERMINAL --
LOCAL T1 PAIR UNRESOLVED; NO REMOTE OR FORMAL AUTHORITY**

Validation 057 first certified the CPPT target on exhaustive small HGPs: the
physical-p collapsed density matches the full hard-coset posterior, the local
and adjacent-swap kernels preserve the product target, the `p=.5` endpoint is
uniform, and reference/Numba outputs agree through `k=64` bit 63.  A shared
read-only m8 CPPT32 log-mass artifact occupies 4 GiB and builds locally in
`10.38s`; a 40-round P/U smoke projected T1 below the two-hour cap.  The full
exp102+exp101 regression was `1000 passed, 4 existing warnings`.

The frozen next step was exactly one P and one exact-K0 U trajectory at
`T1=(2048,8192)`, source `a90d3f01641f4ce1432f739d7a76cf6f9128885a`, with
fresh seed identities and no HP warm start.  It terminates as
`LOCAL_T1_PAIR_UNRESOLVED`: P/U plug-in q_top is `.900885/.144627`, normalized
weight `.038890/.061817`, B weight `.040009/.059124`, collapsed likelihood per
factor `-5.19372/-7.70041`, logical/B character D2 `.346827/.093028`.  Both
trajectories have zero cold-hot-cold round trips; minimum swap rates are
`.00547/.03945`, and cold-origin coverage is `.4375`.  Thus the temperature
path neither transports all replicas nor removes the adversarial initialization
memory.

Primary report SHA is `287d62b5...e1c1`; P/U raw SHAs are
`e771084a...6a27/dada68f8...7f`.  An independent raw-only auditor that does not
call the CPPT sampler/runner/analyzer recomputes the support, B/state/label,
weights, likelihood, characters, counters, gates and terminal status as
`INDEPENDENT_RAW_ONLY_AUDIT_PASS_LOCAL_T1_PAIR_UNRESOLVED`, SHA
`1dd1260d...bf0`.  The apparent `T1_PAIR_SUCCESS` marker means only that the
wrapper completed and wrote an auditable terminal report; it is not a sampler
PASS.

No nd-2/nd-3 job was launched.  This frozen CPPT32 route may not be extended,
replicated, pooled, warmed from HP64 or rescued by common P/zero starts.  It is
not a physical q_top, `IMPOSSIBLE`, CPPT64 theorem, or parameter-point result.
CPPT remains the same collapsed-B tempering family as HP64 and cannot supply
the missing independent confirmation even if a fresh successor later worked;
the next useful candidate must directly address B transport through an
orthogonal hard-coset mechanism or a rigorous oracle.

**DIRECT-BLOCK M8 T1 VALIDATION 056 TERMINAL --
UNRESOLVED SAMPLER CONVERGENCE; NO PHYSICAL RESULT OR M6 AUTHORITY**

The direct dressed-logical-XOR structure probe (validation 047) is terminally
`LOCAL_CENTER_PRESERVING_STRUCTURE_NOT_VIABLE`: its 127-move catalog is
algebraically valid and rank 64, but only rank 4 is accessible from its base
and rank 1 from P at the frozen T3 clock.  Every tested low-energy logical
start has a downhill route into the same base label, while the optimistic
state-independent full-rank scheduler has only `1.700868e-10` expected accepts
per direction.  It would manufacture convergence into one basin rather than
provide bidirectional transport.

The exact random-scan full-B-column Gibbs kernel has passed complete small-HGP
stationarity/detailed-balance and bit-replay tests.  Its fresh 64+256 local
screen (validation 049) remains `LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_UNRESOLVED`:
P/low-energy-L B blocks barely move, exact `A|B` redraws nevertheless make
logical labels move, and U remains far away in B weight/likelihood.  Thus
label changes alone are explicitly rejected as a convergence argument.  The
truth-free MAP bridge probe (validation 050) then found an ordered two-column
bridge with first-step conditional probability about `.03846` and about
`16.4` expected first departures over T1; the short run expected only about
`.5`, so it could not decide T1 viability.  Validation 051 independently
reconstructed 047/049/050 and preserved these exact permissions and failures
(`audit SHA c018e4af...18767`).  It also records that 049's old source identity
omitted transitive dependencies; the raw failure is audited, but its source
identity is not upgraded.

The fresh successor `exp102.q0_random_full_column.t1_m8.v0` was frozen in
validation 052.  It uses 2048 burn plus 8192 measurement full-column clocks
for 8 independent trajectories in each of P, exact-K0 U, two truth-free
B-distinct MAP families, and an 8-state truth-free low-energy S family whose
logical labels and B blocks are distinct.  This closes the blind spot that the
old L starts changed logical/A coordinates while sharing P's B block.  Primary
gates are character-U-statistic q_top/D2, full and B weights, B likelihood,
all B bit/row/column plus dense characters, logical characters, Rhat/ESS,
constant-character burn crossing, and bidirectional MAP-basin visits.  The
physical zero state remains illegal for this nonzero syndrome; shifted zero is
already P.

The immutable remote run `exp102_q0_rfcg_t1_m8_20260724_6fa489f` completed
three-node verified preflight from source
`6fa489f838dffea15b07e1ef3b3fbee3951dd3c0`.  All nodes agreed exactly on the
mass table and four portable transcripts (`exact_consensus=true`), but the
fixed four-worker replay-inclusive projections were `24701.47/24812.06/29871.42`
seconds per trajectory on nd-1/nd-2/nd-3, versus the frozen 7200-second cap.
Aggregate status is therefore `RUNTIME_EXHAUSTED`, and measurement raw count is
zero.  The conda-12 independent audit verifies the control/schedule identities,
40 tasks and 14/13/13 ownership, stage markers, all node/aggregate self-hashes,
runtime arithmetic, exact consensus, and raw absence; its status is
`INDEPENDENT_PREFLIGHT_AUDIT_PASS_RUNTIME_EXHAUSTED_CONFIRMED`, audit SHA
`817425dbaa6a9e5d90d03d34efe16f957beb7424eddd27dcde7cf12d60d75c6d`.

This does not test convergence and does not reject the physical cell or random
full-column mechanism mathematically.  It only says this frozen implementation,
clock, replay, concurrency, and two-hour resource contract cannot proceed.  No
m6 T1, HARD2, formal, held-out, `READY_FOR_FORMAL`, or production authority is
created.  A performance-changing successor requires a new scientific/runtime
contract and fresh source/seeds/raw; the failed gate must not be bypassed by
shortening the chain or weakening the cap after seeing the projection.

Validation 053 then tested a fresh memory-streaming implementation of the same
exact random-scan full-B-column heatbath.  The authoritative macmini preflight
from source `de68bbc06aa729063b24c1f40ba23cc404a44c9c` passed: all 12 complete
`2^24` legacy/streaming CDFs were byte-identical, speedup was `4.9391x`, and
the worst replay-inclusive T1 projection was `2432.39s`.  The immutable remote
run `exp102_q0_streaming_preflight_20260724_de68bbc` did not pass.  Every Linux
node had exactly one legacy/streaming byte mismatch (`U0,column=11`), so all
node reports and aggregate correctly terminated as `CONFLICT` under the
pre-registered any-mismatch rule.

This was not a disagreement among the proposed streaming kernels: the complete
streaming CDF SHA catalog and all four PortablePrng sampling/replay transcript
hashes agree across macmini and nd-1/2/3.  The mismatch is isolated to the
Linux legacy dense floating reference.  It is nevertheless terminal for this
frozen run.  Independently, the remote speedups were only `2.5911x/2.5372x/
1.3823x`, and worst projections were `8797.83/9144.89/17760.30s`, all beyond
the required `4.2x` and/or `7200s` gates.  Therefore fixing or reinterpreting
the float comparison would still not authorize T1.  No T1 raw was generated.
The independent audit preserves this combined outcome as
`INDEPENDENT_AUDIT_PASS_CONFLICT_AND_RUNTIME_EXHAUSTION_CONFIRMED`, audit SHA
`6426a1a01c01747f474d587a10cdb6db9e53db09112193499a8f9307adb7640f`.

A possible fresh successor may exploit the audited positive-mass range to
replace log/exp plus the 128 MiB full CDF with a fixed-order direct-weight block
sum, but it must first pass new exact small-HGP, underflow, portable replay,
and three-node runtime gates.  This is an implementation hypothesis, not an
authorization to reuse 053 evidence, remove adversarial starts, shorten T1,
or claim convergence.

Validation 054 implemented that successor as `RFCG-C24-DPB12-S1`.  It computes
the same exact random-scan full-B-column conditional using direct positive
weights, fixed `2^12` candidate blocks, 4096 block subtotals, and a second pass
only through the selected block.  The target remains
`pi(e|y) proportional to (.04/.96)^|e|` on `H_Ze=y`; the planted error is not
used in the kernel.  Complete `2^24` comparisons certify that the direct
weights remain normal, with worst scaled absolute error `2.020606e-14`, worst
relative error `7.290711e-14`, worst total variation `4.148991e-15`, and
log-weight lower bound `-221.658`.

The immutable run `exp102_q0_direct_block_preflight_20260724_61d605a` from
source `61d605a5e27db0970457736c72d1c45d72a12b10` passed on all three nodes.
The 12 ordered block-subtotal digests and four P/M0/S0/U0 PortablePrng
sampling/replay transcript digests agree exactly across macmini/nd-1/nd-2/nd-3.
The replay-inclusive T1 projections were `4144.85/4139.52/5454.14s`, below the
frozen `7200s` cap.  Aggregate status/SHA are `PASS` /
`27f6d276...10612bc`.  The independent conda-12 audit confirms source,
config/reference, all numeric/runtime gates, consensus, stages, and logs as
`INDEPENDENT_AUDIT_PASS_DIRECT_BLOCK_PREFLIGHT_CONFIRMED`, audit SHA
`9646c6f92070024680728bf377e802e647b48a2b66ca6210c89c436fbd70f539`.

This only authorizes a **fresh m8 T1 diagnostic**.  It is not evidence of
mixing or a posterior/q_top result.  The successor must retain 2048+8192
updates, full replay, and eight independent chains in each of P, exact-K0 U,
M0, M1, and truth-free low-energy S.  Starting all chains from physical zero
is illegal for the nonzero syndrome; shifted zero is already P and would hide
the central failure mode.  The full-label/B-character D2, weight/likelihood,
B-bit/row/column/dense-character, logical-character, Rhat/ESS, burn-crossing,
bidirectional MAP-basin, and B-column/label-change gates remain mandatory.
No m6, HARD2, formal, held-out, `READY_FOR_FORMAL`, or production authority
exists until the fresh T1 raw passes those gates.

Validation 055 freezes that fresh successor as
`exp102.q0_random_full_column_direct_block.t1_m8.v1`.  Its new control keeps the
same outcome-blind P/U/M0/M1/S geometry but refreshes every schedule seed and
logical/B character; all four seed fields have zero overlap with validation
052.  The implementation is bound byte-for-byte to the two validation-054
sampler files and portable artifact.  Before launch, a real miniature direct
raw and full replay exposed and fixed three dormant analyzer hazards: missing
direct `version/conditional_engine` binding, a missing `state_label` import,
and a one-ULP B-likelihood false conflict from using a different sum order.
The new analyzer independently rebuilds factor indices, preserves the frozen
kernel sum order, and rejects a tampered engine identity.

Validation 055 is terminally **`RUNTIME_EXHAUSTED` at preflight** and produced
zero measurement raw.  The first two schedule attempts failed before control
creation because the schedule's fresh run root had already been created, first
manually and then by a misplaced stage marker.  Both remain infrastructure
audit only.  The corrected immutable third run
`exp102_q0_direct_block_t1_m8_20260724_146ef55_r3` used source
`146ef550591a72435641c47baa8794c338f7a27e`, schedule SHA
`bbc2e268...ee731a`, and the frozen 40-task `P/U/M0/M1/S x8` ownership.

On that final source, the complete validation-054 three-node preflight passed
with exact consensus and replay-inclusive T1 projections
`4216.16/4149.15/4549.57s`; aggregate SHA is `ae356c9e...b35ac`.  The separate
055 runtime estimator, however, measured only `2+8` updates and linearly
scaled total elapsed time, including fixed initialization/runner overhead, to
10,240 updates before applying the safety factor.  It projected
`9272.13/8779.07/13638.99s`, so its own unchanged 7200-second gate correctly
stopped measurement with aggregate SHA `7fffcdda...f461`.  The independent
audit verifies both facts, schedule failures, all hashes/arithmetic, and raw
absence as
`INDEPENDENT_AUDIT_PASS_PORTABLE_PASS_T1_RUNTIME_EXHAUSTED_CONFIRMED` (SHA
`00622194...c665`).

Thus 055 did not test the direct-block sampler or the m8 posterior at all.  It
exposed a pre-registered runtime estimator that repeats fixed startup cost in
its extrapolation; this cannot be relabeled PASS or sampler failure.  A fresh
successor may use representative replay-inclusive probes (or a frozen
intercept/slope design) while retaining T1, the 7200-second cap, all five
adversarial initial families, full replay, and every statistical gate.  It must
use fresh source/contract/schedule/seeds/raw.  No m6, HARD2, formal, held-out,
`READY_FOR_FORMAL`, or production authority exists.

Validation 056 froze that fresh successor as
`exp102.q0_random_full_column_direct_block.t1_m8.v2`.  The immutable run
`exp102_q0_direct_block_t1_m8_v2_20260724_6933e31` used source
`6933e319b27840976f34e27c0d11313b6973cbe3`, archive
`b62d0e22...13eb5`, manifest `135eb089...17e48`, and schedule
`ca057fbc...58d3`.  The complete validation-054 portable preflight and fresh
two-length v2 preflight both passed exact three-node consensus; the latter's
worst replay-inclusive factor-two T1 projection was `6550.3213s < 7200s`.
Fixed `14/13/13` ownership then completed all 40 measurement raw with no reuse.

The frozen primary analyzer returned **`UNRESOLVED_DIRECT_BLOCK_T1_M8`**,
report SHA `e1bfb340...6015` and raw-set SHA `a267ded6...2259`.  An out-of-band
`allow_pickle=False` auditor independently reconstructed the PortablePrng K0
starts, hard-coset algebra, every B transcript/state/label/weight/likelihood,
q_top/D2, Rhat/ESS, all family/pair gates, MAP bridge, constant-character rule,
and terminal status without calling the sampler, replay runner, or primary
analyzer.  It returned `INDEPENDENT_RAW_ONLY_AUDIT_PASS`, audit SHA
`ada30d3c...b08e`.

All five families fail within-family Rhat/ESS.  P/M0/M1/S are close in q_top
(`.90378--.92260`) and mean weight (`.0388708--.0388953`), but their max Rhat
is `1.1335--1.3048` and min ESS only `66.86--87.61`; their pairwise pooled-B
Rhat and B-character means also fail.  U exposes the decisive global failure:
after 2048 burn updates its normalized state/B weights remain
`.097775/.101909`, versus about `.03888/.0400` for low-energy starts.  U has
`q_top=.0000405`, max Rhat `inf`, min ESS `39.75`, and every U/low-energy pair
fails all eight distribution gates.  For P/U, `delta q_top=.90374`, logical D2
upper `.93903`, B-character D2 upper `.20827`, and 466 B characters fail their
mean gate.  Yet every U chain records at least 580 B-column and 2406 label
changes, while the bidirectional MAP bridge passes in all M0/M1 chains.  Thus
visible motion and a known basin bridge do not establish global equilibration.

This is a fixed-budget sampler failure, not a physical q_top or proof that the
cell is impossible.  No m6, HARD2, formal, held-out, `READY_FOR_FORMAL`, or
production authority exists.  The raw may not be extended or merged with a
successor, and U/MAP/S may not be replaced by common P starts; physical zero is
outside the hard coset and shifted zero is already P.  The primary analyzer
also emitted a `uint8` underflow warning in its constant-character helper.  No
B character was globally constant, so the corrected independent computation
also has zero freeze failures and the terminal result is unchanged.  A fresh
successor must use signed arithmetic, a new contract/source/seeds/raw, and --
under the user's current resource rule -- only nd-2/nd-3.

**BP-SYSTEMATIC INDEPENDENCE-MH IS EXACT BUT STICKY ON ADVERSARIAL LOW-ENERGY
STARTS -- NO HARD2 OR REMOTE LAUNCH**

The fresh local `exp102.q0_bp_imh.local.v1` diagnostic on
`m08_c06,p=.04,d00,attempt022` is terminally
`LOCAL_BP_IMH_TRANSPORT_UNRESOLVED`.  The exact full-support proposal is
`.5` forward plus `.5` reverse BP-systematic, with each source itself
`.90 BP + .09 prior + .01 uniform`; its independence-MH ratio targets exactly
`(.04/.96)^|e|` on `H_Ze=y`.  P, eight independent exact-K0 U, and eight
distinct legal low-energy L starts use fresh seeds and a fixed `256+2048`
clock.  The planted error enters only P/L initialization, never the proposal,
energy, or acceptance ratio.

Before raw generation, the analyzer was red-teamed to add a full 64-bit label
collision `D2_norm` gate.  This closes a genuine blind spot where two logical
distributions can have equal purity and all 64 equal basis-character means
while occupying disjoint supports.  Eighteen focused tests cover the complete
small transition matrix, detailed balance/stationarity, transcript replay,
the D2 counterexample, complete source-tree binding, and relative CLI paths.
The v1 run then produced 24/24 raw and the independent `allow_pickle=False`
auditor reconstructed all 55,296 hard-coset MH decisions exactly.  Report
self-hash is `62a96e7f16cbbc020f8d4e893c413bd11ec54da928893ccf23abbf6c65983c58`;
raw-set SHA is `60ae69f3b829fd6037cf25979f0a55f3e74b52bc086fb988533f963ee70bc28c`;
audit self-hash is `d7af8f008c500b72df512a546a051b53e1c049de5fc29a92b428cb9a35fd2ce0`.

The failure is structural, not a questionable Rhat edge.  Every P and L
trajectory makes zero real burn or measurement moves.  U cools in only 1--3
real burn moves, but all eight chains land on the same weight-62 state and make
only 0--2 real measurement moves.  P's largest observed measurement log
acceptance is at most `-53.13`; L ranges at best from `-88.69` to `-47.79`.
Equivalently, the proposal undersupplies these high `pi/q` states by tens of
log units, so accepted self-proposals can be numerous while actual movement is
absent.  P versus L has diagnostic `delta q_top=1` and `D2_norm=1`; U versus L
has `.998413` for both.

The infrastructure-failed v0 attempt in validation 045 is separately frozen
as `INFRASTRUCTURE_FAILED_RELATIVE_OUTPUT_PATH`: it has one forbidden raw and
no receipt/report.  V1 has a fresh contract/config/seed namespace with zero
seed overlap; no v0 raw was reused.  Neither run creates a posterior result or
authorizes HARD2, remote, formal, held-out, or production work.  A naive
BP-cooling plus old full-row-Gibbs hybrid is also not justified: BP maps every
U chain into the P logical label, while the old full-row kernel maps P/L to the
same frozen B basin, so apparent family agreement could be common-mode
collapse rather than global sampling.  Any successor must demonstrate
result-independent high-`pi/q` basin/signature coverage and independent B/tail
evidence, not merely more accepted proposals or common initialization.

**BP-MIXTURE DOMINANCE WITNESS CANNOT CERTIFY GLOBAL COVERAGE -- NO BP-ONLY
IID OR REJECTION-SAMPLER LAUNCH**

The fresh local structural probe
`exp102.q0_bp_dominance_witness.feasibility.v0` is terminally
`BP_MIXTURE_REJECTION_ENVELOPE_WITNESS_INCONCLUSIVE` on
`m08_c06,p=.04,d00,attempt022`.  It deliberately does not reuse a BP-IID
estimate, run MCMC, or calculate a posterior statistic.  Instead, it tests a
necessary condition for converting the exact frozen three-component
BP-systematic proposal density into a strict rejection-envelope or
bounded-importance route.

For 1,691 pre-frozen legal witnesses (planted, 64 canonical rank-complete
reduced-logical states, and every planted-plus-one-coordinate state from the
two systematic bases), it evaluates the exact mixture density with upward
rounded Decimal arithmetic.  The only allowed normalizer inequality is

```text
Z = Pr_.04(H_Z e=y) / (.96)^1600 <= (.96)^(-1600).
```

It yields a rigorous *lower* witness bound on `sup pi/q`, but this universal
normalizer upper bound is so loose that even the largest witnesses are only
`5.53e-63` (forward) and `2.54e-53` (reverse), far below the frozen `1e6`
rejection-envelope cap.  This does not show BP is good; it shows that this
witness test cannot say anything useful without the missing tight global
normalizer/tail upper bound.

Accordingly, the result cannot authorize a fresh BP-only IID estimator,
rejection sampler, q_top, remote task, `READY_FOR_FORMAL`, held-out pass, or
production.  It also does not validate a common planted start or replace P/U
adversarial MCMC checks: this was an IID proposal-bound question, not a chain
initialization test.  The canonical config SHA is
`be78411d1459a6a33f835fc0780f70bd41cd4d0c2f45e9bb659dceb4f3faf180`; the
self-hashed report is
`d36815dce4662c922791409258cf1dbb43492f54465453cce116182e9862e20b` in
`validation/044_q0_bp_dominance_witness_feasibility_20260724/`.

**COLLAPSED-B HCA CANNOT TOUCH THE FROZEN LOW-ENERGY LOGICAL DIRECTIONS -- NO
HP64+B-HCA IMPLEMENTATION OR REMOTE LAUNCH**

The fresh local structural probe
`exp102.q0_collapsed_houdayer.structure.feasibility.v0` is terminally
`COLLAPSED_B_HCA_NO_LOW_ENERGY_RECOMBINATION` on
`m08_c06,p=.04,d00,attempt022`.  This was a deliberately narrower red-team
test of an exact generalized Houdayer move on the actual collapsed slow
variable,

```text
pi_B(B) proportional to (.04/.96)^|B| product_j M_p(Y[:,j] xor B H[:,j]).
```

It did not run MCMC or calculate a posterior statistic.  Small-HGP exhaustive
tests verify the B-mask conversion, factor-pair invariance, HCA involution,
row sums, detailed balance, and stationarity.  The real-code report is
self-hashed as
`e4e6b3cf5576a896d8c588e37224a260c24c410aeb4ac45216e536cd0319df9b`.

The decisive structural result is that all 16 frozen P/low-energy-L starts,
all 120 low-energy L/L pairs, and all 64 P/rank-complete-L controls have the
same collapsed B mask on both sides: their B disagreement count and component
count are zero, despite differing physical logical labels.  The logical
variation in this catalog lies entirely in A.  The independently exact-K=0
U/U pair has 284 differing B variables, but they form one complete component,
so its only B-HCA action is a whole-pair exchange and creates zero new
unordered B states.

This closes a subtle but important loophole in the apparent HP64+HCA idea:
an exact B-factor swap can be algebraically correct yet be irrelevant to the
logical direction that must be sampled.  Implementing a large hybrid and
measuring its acceptance or state changes would optimize the wrong variable.
The result rejects only direct collapsed-B HCA on this frozen catalog; it does
not prove HP64, physical-coordinate HCA, q=0, or the posterior impossible.
There is no `q_top`, convergence claim, remote task, `READY_FOR_FORMAL`,
held-out pass, or production authority.  Evidence and the pre-run red-team
review are in `validation/043_q0_collapsed_houdayer_structure_feasibility_20260724/`
and `COLLAPSED_B_HOUDAYER_REVIEW.md`.

**HCA-RHB1 PAIR KERNEL HAS REAL LOW-ENERGY RECOMBINATION BUT FAILS THE
EXACT-UNIFORM ADVERSARIAL FAMILY -- NO REMOTE LAUNCH**

The local Houdayer investigation has closed the frozen
`exp102.q0_houdayer_pair.local.v0` diagnostic on the nonzero-syndrome hard
sentinel `m08_c06,p=.04,d00,attempt022` as
`LOCAL_HOUDAYER_PAIR_TRANSPORT_UNRESOLVED`.  This is not a server wait or an
infrastructure stop: all 32 pair raws, deterministic replay, and the separate
`allow_pickle=False` raw-only audit completed.  The product target was exactly
`pi(e_left|y) pi(e_right|y)`, with each factor proportional to
`(.04/.96)^|e|` on `H_Z e=y`; the planted state was used only as a legal
adversarial initialization.

Two exact-estimation routes were first structurally ruled out for this
sentinel: the single-copy character-WMC probe `035` reaches induced width 378
under min-degree, while min-fill exceeds width 102 before its 120-second cap;
the best tested linear-code trellis order in `036` has state exponent 584,
far above the frozen exponent-24 actionability cap.  These are representation
and ordering failures, not claims that a partition function or q=0 is
impossible.

For Houdayer coordinates, `037/038` found that the tensor-logical basis yields
only whole-pair exchanges on the frozen real low-energy catalog.  The one
pre-registered alternative, canonical reduced logical coordinates, did have
real structure in `039`: 102 of 120 low-energy L/L pairs split into two
components and can create a new unordered pair (for example, weights 67 and
67 recombine to 63 and 71).  That signal justified the exact pair-kernel test,
but did not itself prove transport.  The pair kernel's small-HGP complete
transition matrix passes stationarity, and the source-rebound outcome-blind
preflight `042` projected only about 22.0 seconds per frozen pair trajectory
with a twofold safety factor.

The fresh HCA-RHB1 screen then used four legal, pre-frozen pair families:
planted/planted (`PP`), two independent exact-K=0 uniform hard-coset states
(`UU`), two deterministic low-energy distinct-label states (`LL`), and a
planted/low-energy whole-swap control (`PL`).  `PP`, `LL`, and `PL` agree near
normalized pair weight `.03886`; all eight `LL` pairs show real recombination
(1,091 aggregate new unordered-pair events).  In contrast, every `UU`
trajectory remains separated, with mean normalized pair weight `.1486354`, a
`.1097799` gap from `PP`, maximum basis-character gap `1.4603271`, failed
early/late stability, and zero new unordered-pair events.  All 1,024
measurement HCA operations in every U/U pair were only whole-pair exchanges.

Therefore this exact HCA kernel and its fixed local budget are rejected.  It
would be a scientific error to replace U/U by common planted or physical-zero
starts: physical zero is outside this nonzero-syndrome hard coset, and
shifted-coordinate zero is exactly the planted P state.  The result creates no
posterior, purity, `q_top`, remote, `READY_FOR_FORMAL`, held-out, or production
authorization, and does not imply that HCA, q=0, or the posterior is
mathematically impossible.  The immutable evidence is in
`validation/035_q0_character_wmc_structure_feasibility_20260724/` through
`validation/042_q0_houdayer_pair_runtime_rebind_20260724/`, especially
`validation/041_q0_houdayer_pair_local_v0_20260724/`.

**DEPTH-TWO COLLAPSED-B TAIL ENVELOPE NOT TIGHT ENOUGH -- NO REMOTE LAUNCH**

The fresh local-only, exact-rational `exp102.q0_collapsed_tail.depth2.feasibility.v0`
probe on `m08_c06,p=.04,d00,attempt022` is terminally
`DEPTH2_ENVELOPE_NOT_TIGHT_ENOUGH`. It is not MCMC: it uses two frozen,
non-planted deterministic MAP-derived B marginals only as retained lower mass,
and contracts a directed-rounding depth-two upper envelope for every collapsed
B configuration. The canonical config SHA256 is
`2d1c27b769e5011139265cecac9d8c794f694c31acc03d92a90829e706933bb0`; the
immutable self-hashed report is
`dffacc4ac340c33b49e8578432ce17a3f8b89a65698d08985677662f3d23f147`.

The width-25 contraction meets its resource gates (`3.14` seconds and
2,504,933,376-byte peak RSS; largest single table 512 MiB under the 6 GiB
cap), but its total scaled partition upper bound is `3.11016e-11` while the
two retained B marginals supply only `3.30805e-96` lower mass. Consequently
the rigorous tail/retained-mass upper ratio is `9.40179e84`, roughly 87 decimal
orders above the frozen `.01` target. This is a negative tightness result, not
a performance problem that more runtime, a different MCMC start, or a prettier
finite-sample diagnostic can solve. It rejects only this depth-two factorized
envelope; it does not prove q=0 or certified normalizer methods impossible.

The probe produces no posterior, purity, `q_top`, logical-sector decomposition,
remote task, `READY_FOR_FORMAL`, held-out pass, or production authorization.
In particular, a deeper or otherwise changed envelope would require a new
reviewed resource/tightness contract rather than treating this fast completion
as evidence to escalate it.

**FROZEN BP-SYSTEMATIC IID MIXTURE UNRESOLVED -- NO REMOTE LAUNCH**

The fresh local-only `exp102.q0_bp_systematic_iid.local.v0` hard-sentinel
diagnostic on `m08_c06,p=.04,d00,attempt022` is terminally
`LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_UNRESOLVED`. It used 49,152 direct,
component-provenanced hard-coset draws: 16 blocks times 1,024 draws from each
of the independently frozen BP-SYS-F64, BP-SYS-R64, and rebuilt MAM-IMH8
sources. There is no MCMC chain, P/U/L initialization, resampling, cloning,
or result-dependent extension. The target is exactly
`pi(e|y) proportional to (.04/.96)^|e|` on `H_Z e=y`; planted error never
enters the energy or proposal score. The runner's deterministic regeneration
and algebra replay pass, as does a separate `allow_pickle=False` raw-only
analysis. Raw SHA256 is
`fd662ae5a30ce0e0aa70ebf6253882da91c7cf479db9669400affe972a1625da`; the
immutable report's internal SHA256 is
`2a62ddf1d7bfc49b06e2a80e4d6d45f2d7558970bed4f7de28faedd0f25705fb`.

The two BP sources individually pass their frozen block-weight and provenance
coverage gates: F64 has minimum ESS `730.10` and maximum normalized weight
`.00750`; R64 has `457.00` and `.00219`. Their collision-derived diagnostic
values `.994531` and `1.000000` differ by `.0054695`, and their D2 diagnostic
also passes the deliberately empirical pairwise gate. This is not enough to
claim a posterior result: both can share an unobserved-tail failure mode.

The frozen equal three-source mixture fails its required weight-stability
gate, with minimum block ESS `23.48 < 50` and maximum normalized weight
`.14695 > .10`; MAM itself has `23.00` and `.14765`. Several mixture blocks
are consequently MAM-dominated. The test therefore rejects this exact
three-source estimator schedule. Although MAM was a stress source rather than
a pass-rescuing primary source, it is explicitly part of the frozen
equal-mixture estimator and cannot be stripped out after inspecting the
result. A BP-only successor would need a separately reviewed contract, fresh
seeds, and fresh raw, and still could not substitute for a global
tail/normalizer certificate or an independent confirmer.

The displayed collision diagnostics near one are not posterior purity,
`q_top`, or a physical result. The BP proposal density remains exact despite
the fixed BP iteration's oscillatory messages, but its defensive prior/uniform
components contributed no observed target mass, so finite local overlap gives
no rigorous control of a remote mode. There is no remote task,
`READY_FOR_FORMAL`, held-out pass, or production authorization.

**FRESH LOCAL IID-MIS WEIGHT-STABILITY FAILURE -- NO REMOTE LAUNCH**

The fresh local-only IID multiple-importance-sampling diagnostic on
`m08_c06,p=.04,d00,attempt022` is terminally
`LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_UNRESOLVED`.  It made 49,152 new direct
hard-coset draws (16 independent blocks, 1,024 draws from each of MAM-IMH8,
LSI-IMH-T05, and LSI-IMH-T10 per block), so it has no MCMC initialization,
P/U/L family, resampling, cloning, or chain-transport artifact.  The target
was exactly `pi(e|y) proportional to (.04/.96)^|e|` on `H_Z e=y`; planted
error never enters its energy.  The raw-only replay and a current-source
recomputation pass.  Raw SHA256 is
`6cc3c19710725ef5ab714e010d636c7a0a0e7928db71e68059b1029009382071` and the
immutable report SHA256 is
`d7dd5521b7292f68c01f0202d623da34968d060d3ae3422cd6f117e837a36e0a`.

The frozen importance-weight gates fail for every primary view: MAM has
minimum per-block ESS `22.09` and maximum normalized weight `.1522`,
LSI-T05 has `28.91` and `.1629`, and the equal three-proposal mixture has
`28.78` and `.1574`, against required ESS at least `50` and maximum weight at
most `.10`.  MAM/T05 agreement and the mixture's small jackknife SE do pass,
but those are not substitutes for stable weights.  The displayed diagnostic
values near `.98--.993` are therefore not a posterior purity, `q_top`, or a
physical result.

This test removes the question of whether a Markov-chain initial state was
chosen badly, but it reveals a different blind spot: full proposal support,
cross-proposal agreement, and an apparently precise finite estimate do not
bound unobserved target tails in an 832-dimensional hard coset.  Conversely,
a single low-temperature chain failing to traverse all 64 logical directions
is not by itself a mathematical proof that its `q_top` is wrong; what remains
unresolved is total mass outside the observed modes.  The frozen raw cannot be
extended or used to tune gates after seeing this outcome.  It stores source
proposal IDs but not each proposal's internal anchor/component ID, so it
cannot cleanly diagnose the weight tails by component.  Any successor needs a
new contract, fresh artifacts/seeds/raw with that provenance, and an
independent tail/normalizer certificate or confirmation route.  There is no
remote task, `READY_FOR_FORMAL`, held-out pass, or production authorization.

**FCG-C24 V0 FULL-COLUMN EXACT GIBBS RUNTIME EXHAUSTED -- DO NOT DEPLOY TO
HARD2**

The new collapsed-HGP kernel heatbaths all 24 bits in one B column jointly by
enumerating its exact `2^24` conditional.  Its small `n=10/n=13` exact-oracle
suite passes direct conditionals, detailed balance, full-sweep stationarity,
hard-coset preservation, and replay.  This establishes the local conditional
identity, not a posterior result.

Before generating a single P/U/L trajectory, an outcome-blind m8 runtime probe
froze one warm-up, two timed column updates, and the formal T1 resource
calculation: `(2048+8192)*24 = 245760` column updates, with a factor-two
safety margin and a two-hour per-trajectory cap.  The timing is `.278952`
seconds per exact column conditional, with `.442808` seconds setup and
1,197,178,880 bytes peak RSS.  The resulting projected T1 wall time is
`137111.403` seconds (about 38.1 hours), so the fixed runtime gate is
`RUNTIME_EXHAUSTED`.  The immutable report SHA256 is
`847b2abe1bfc1f91364a9a944d59ad30ca7ba84979e282ddce6f451506a63a80`.

No P/U/L raw, label, character, weight, `q_top`, or physics estimate was
generated; the probe constructed no full state at all.  Continuing with an
adversarial-start screen after this outcome would optimize a kernel that cannot
enter the required formal schedule.  This rejects only the complete `2^24`
full-column enumeration at the frozen runtime cap, not q=0, a differently
factorized exact block update, a certified tail calculation, or any physical
parameter point.  Full evidence is in
`validation/028_q0_full_column_gibbs_v0_20260724/`.

**LOCAL BRIDGE AND COLLAPSED-B CERTIFICATE FEASIBILITY UNRESOLVED -- DO NOT
DEPLOY THESE DIAGNOSTICS TO HARD2**

The local fixed-sector bridge V2 corrects a V1 reverse-ratio arithmetic error
and passes its small-HGP exact identity test, but it does not establish usable
overlap.  On `m08_c06,p=.04,d00,attempt022`, six bridge bits are almost fully
pinned in twelve fixed observations.  P has a 20 percent forward/reverse
product discrepancy (`5.232780885631004e-9` versus
`6.277157494182837e-9`), while S happens to agree.  This short-clock result is
neither a sector ratio nor evidence of within-sector stationarity, and it
never bounds the mass of unvisited logical sectors.

The separate exact-rational collapsed-B V0 tail-envelope probe also does not
open an estimator route.  Its directed interval classical factors enclose a
unit total probability, but the factorized normalizer upper envelope remains
about `10^311.34` times the truth-free `B=0` lower anchor after one B row; a
two-row contraction requires induced width 25, above the frozen width-18 cap.
Even the planted-B diagnostic anchor leaves a two-row prefix envelope about
`10^86.65` times its lower weight.  This rejects only the V0 factor-max
envelope at its bounded resource cap, not all branch-and-bound designs, q=0,
or the posterior.  Full evidence is in
`validation/025_q0_sector_bridge_feasibility_20260724/` and
`validation/027_q0_collapsed_tail_bound_feasibility_20260724/`.

Neither diagnostic produces `q_top`, a posterior estimate, `READY_FOR_FORMAL`,
or authority for remote, held-out, or production work.  In particular,
forcing every chain to P or a purported physical zero state would conceal the
previous adversarial-start failure; physical zero is not legal for this hard
cell's nonzero syndrome.

**UASRE32/64-R1-A1 V0 LOCAL AUXILIARY-STABILIZER TRANSPORT UNRESOLVED -- DO
NOT DEPLOY THESE CONFIGURATIONS TO HARD2**

The fresh local-only `exp102.q0_hgp_aux_stabilizer_pt.v0` adversarial-start
screen is terminally `LOCAL_AUXILIARY_STABILIZER_TRANSPORT_UNRESOLVED`. It
froze the nonzero-syndrome hard cell `m08_c06,p=.04,d00,attempt022`, two
auxiliary-stabilizer replica-exchange configurations (32 or 64 replicas),
P/exact-K0-U/legal-low-energy-L starts, eight independent trajectories per
family, and fixed `(burn, measurement)=(256,2048)`. Its immutable manifest
SHA256 is `1c5b931117a35b859c33a1a1abe348d0f8e547784395812e2ccb3884b2271c29`.

All 48 raw files completed (run SHA256
`c262bc5f9b6320d22fb066a3d70a61783fce5f1479fee437c50f1c4d23e9261f`). The
manifest-bound raw validator, separate six-worker bit-exact replay, and
pickle-free raw-only audit all pass. Their SHA256 values are respectively
`dd42401222d64ab22b01c361d14bab096eb8291f45254edc834be0f8e6bf7aba`,
`d99d0b27d8edb13c3b58bce4d05b15974befa281146b0ca19c71e02f5591b669`, and
`646c0ee7f40bac604adbd5c206c7bc25164b5fcc9c291d21e4baa8af5e09becf`.
The crosscheck confirms matching pre-registered gate summaries (SHA256
`485e1cd4f6f2bc01902bb3c8a2342c80a2a23d3b377ce16b57f9eb242f8d2966`).

Neither configuration passes. P and L agree pairwise, but the exact-uniform
U family disagrees with both in normalized weights, complete score, all 128
logical characters, and most B-mask means; U also fails fixed-clock
early/late stability. P/L agreement and nonconstant local B-mask motion are
not treated as global mixing. U minimum weights are 135--174 (32 replicas)
or 163--179 (64 replicas), versus a known legal P weight of 63. The loose
target-support gate is inconclusive at those U weights (upper bound is 1), so
this failure must not be recast as proof that U has negligible target mass.
It is the pre-registered distribution disagreement and time instability that
reject the configurations.

The frozen UASRE raw may not be extended, pooled, reweighted, used for
`q_top`, sent to HARD2, or used to authorize remote work, `READY_FOR_FORMAL`,
tuning, held-out, or production. Starting every chain at P or at a purported
zero state would hide rather than resolve the finding: physical zero is
illegal for this nonzero syndrome, and shifted-coordinate zero is P. This
rejects only the two fixed configurations and their local budget; it is not
an `IMPOSSIBLE` conclusion about q=0, the posterior, or all
auxiliary-stabilizer replica exchange. Full evidence is in
`validation/024_q0_aux_stabilizer_v0_20260724/RESULT.md`.

**UARE32/64-R1 V0 LOCAL UNIFORM-ANCHOR TRANSPORT FAILURE -- DO NOT DEPLOY
THESE CONFIGURATIONS TO HARD2**

The fresh local-only `exp102.q0_hgp_uniform_anchor_pt.v0` adversarial-start
screen is terminally `LOCAL_UNRESOLVED_UNIFORM_ANCHOR_TRANSPORT`. It froze the
nonzero-syndrome hard cell `m08_c06,p=.04,d00,attempt022`, complete-energy
uniform-anchored collapsed-B replica exchange, P/exact-K0-U/legal-low-energy-L
starts, eight independent trajectories per family, and fixed
`(burn, measurement)=(256,2048)`. Its immutable manifest SHA256 is
`9098102f1612cb70630d936fb86b949e9a19baa428c187238741d6dbd2f1b560`.

All 48 raw files completed (run SHA256
`322a23b72f1fb443e435f95ce64088f7a524437a3005b3e1979e7bb2ff507761`). A
raw-only V2 audit rebuilds the hard-coset algebra, exact scores, P/U/L starts,
packed states, labels, B/A traces, counter constraints, and frozen gates from
pickle-free NPZ data; it reports SHA256
`76e5233dba8a0a24618199f0f397552f9d8d01dd12bc5701016ea6f200d5290f`. A
separate V2 validator leaves the manifest-bound runner unmodified and repeats
all 48 trajectories with its raw validator and sampler replay; every replay
is bit-identical and the replay SHA256 is
`f2c84bb8334d7b1ac6c7c56799ca9e4296c07a24274066e2d5983df2e0d767d4`.
The V1 analyzer and first audit each had the same post-replay
time-half-dictionary indexing defect, so neither frozen source nor raw was
rewritten; the V2 artifacts are isolated and record their own source hashes.

Both UARE32-R1 and UARE64-R1 fail. P and L agree, but U disagrees with both
and remains at minimum measurement weights 247--255 or 247--262, respectively,
whereas a known legal P state has weight 63. With the deliberately loose full
hard-coset multiplicity bound (dimension 832), every U trajectory's observed
region satisfies `Pr_pi(|e|>=w) <= 2^832*(.04/.96)^(w-63) <=
3.148385600959564e-4`, below the frozen `.001` support threshold; U also fails
fixed-clock early/late stability. This is the intended evidence that common
P-like starts would hide, not fix, a global-transport failure.

The frozen UARE raw may not be extended, pooled, reweighted, used for `q_top`,
sent to HARD2, or used to authorize remote work, `READY_FOR_FORMAL`, tuning,
held-out, or production. The result rejects only these two configurations and
this budget; it is not an `IMPOSSIBLE` conclusion about q=0, the posterior, or
uniform-anchored replica exchange. Full evidence is in
`validation/023_q0_uniform_anchor_pt_v0_20260724/RESULT.md`.

**FRG-VE1 V0 LOCAL ADVERSARIAL-INITIALIZATION NONCONVERGENCE -- DO NOT DEPLOY
THIS CONFIGURATION TO HARD2**

The fresh local-only `exp102.q0_hgp_full_row_gibbs.v0` diagnostic is terminally
`LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE`. It froze the same nonzero-syndrome hard
cell `m08_c06,p=.04,d00,attempt022`, the exact collapsed-HGP full-row
variable-elimination heatbath, P/U/legal-low-energy-L starts, eight independent
trajectories per family, and fixed `(burn, measurement)=(64,512)`. Its
immutable manifest SHA256 is
`430659be5aac3b1fe099b2c15eadda194878beba663fb7b11874fc05b4bf69a7`.
The target remains exactly `pi(e|y) proportional to (p/(1-p))**|e|` on the
hard coset; planted error is used only to make the disorder and P start, never
as an energy reference.

All 24 raw files pass the runner's complete deterministic seed replay. The
terminal report SHA256 is
`53604ebf941bd514867baa83f1c47abc901803d10ae83923036d340da509550d`.
A separate raw-only audit never imports the full-row sampler or runner and
independently rebuilds HGP wiring, P/U/L starts, min-fill plan, labels,
packed states, B/A traces, counter invariants, raw hashes, and the frozen gate
from `allow_pickle=False` NPZ input; it passes with SHA256
`ca9556e01e0e7bdc0a26ddb69d067c1dd209f6439c90ad43ee3156ecb13cc561`.
The focused exact/oracle/reference/Numba/bit-63 suite passes `42` tests.

The all-character leave-return gate alone is deliberately not interpreted as
a stationarity proof: P/L's low variability could be physically correct for a
low-temperature posterior. The independently audited raw nevertheless gives a
stronger target-support failure. P and L both reach the same legal weight-63
state before measurement, while every U-family measurement remains at weight
at least 248. Here the hard coset has dimension 832. Even using its entire
cardinality as a worst-case multiplicity, the target probability of the
weight-248-or-higher region is at most
`2**832 * (.04/.96)**(248-63) = 1.3118273337331353e-05`, relative to the
known legal weight-63 state. Thus U's active label/B motion remains confined
to a demonstrably negligible target-support region, not an alternative
equilibrium mode or a reason to replace U with an illegal physical zero.
The raw-only convergence diagnostic SHA256 is
`b38501c0cab8183b53c2278bbed876e57542921c54bb435572a9fce8499746f5`.

This rejects only the frozen FRG-VE1 V0 configuration and fixed local budget.
Its raw may not be extended, pooled, used for `q_top`, sent to HARD2, or used
to authorize remote work, `READY_FOR_FORMAL`, tuning, held-out, or production.
It is not evidence that exact row conditionals, q=0, or the posterior are
mathematically impossible. Any successor must freeze a new mechanism and
retain both an adversarial support/convergence check and an independent
confirmation method.

**CAIS64-B8-S1-N128 V0 LOCAL FULL-PATH WEIGHT COLLAPSE -- DO NOT DEPLOY THIS
CONFIGURATION TO HARD2**

The fresh local-only `exp102.q0_hgp_collapsed_ais.v0` diagnostic is terminally
`LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE`. It froze the same hard
`m08_c06,p=.04,d00,attempt022` cell, HP64's 64-level quadratic collapsed
bridge, one reversible eight-bit B-block heatbath sweep at every nonzero
level, no resampling or cloning, and eight independently seeded N=128
populations (four column-major and four row-major exact-base constructions).
Its immutable manifest SHA256 is
`c3dc27a3e0d7a233ac66027c61f7e642e2cb343b5b01bc8120dd3e0211965ba6`.
The lambda=0 law is exactly iid Bernoulli(.04) B, so P/U/L and physical zero
are deliberately not substituted for this initializer; the physical zero
state is outside the cell's nonzero-syndrome hard coset.

All eight raw paths passed a full deterministic seed replay
(`5e6ae5e47ca67e17692f12051fd71a65a400664e9633ead4d20f15558e662ac7`) and a
separate raw-only audit which never imports the AIS engine or calls a sampler
(`c211911b2ceaaf6e2b033950b8eef32d6ac4c9623e68ae2b5f4cdd6ce5317321`). The
audit independently rebuilt the HGP syndrome, exact iid B base, classical
coset-mass table, every B-derived A syndrome and likelihood, all incremental
and cumulative weights, mutation-counter constraints, final target, gates,
and report identity. The report SHA256 is
`2f6c298324ce7f647cceec7ddd7f377a9dc2a2391ca77d0d8bf49dc2ab0f9324`.

Every population fails all frozen V0 path-weight gates: final importance
ESS/N is `.0078125--.0100431` instead of at least `.25`, final maximum weight
is `.872760--1.000000` instead of at most `.10`, and the largest single
incremental normalized weight is `.122396--.214436` instead of at most `.10`.
The median cumulative ESS falls from `85.93/128` at stage 15 to `1.22/128` at
stage 31 and `1.000002/128` at the cold endpoint. Thus removing resampling
correctly removes the prior genealogy failure, but it does not make the full
AIS path reliable: late bridge weights still concentrate on essentially one
particle. Per-stage movement, a valid exact base, or lack of cloning cannot be
substituted for full-path importance ESS.

This rejects only the frozen no-resampling CAIS64 configuration. Its raw may
not be extended, pooled, reweighted after the fact, used for `q_top`, sent to
HARD2, or used for `READY_FOR_FORMAL`; it is not evidence that the posterior,
all AIS schedules, or q=0 itself is mathematically impossible. Any successor
requires a separately reviewed target/bridge/mutation/weight contract, fresh
seeds and raw, and an independent confirmation mechanism before remote work.

**CSMC64-B8-S1-N128 V0 LOCAL ALWAYS-RESAMPLE GENEALOGY FAILURE -- DO NOT
DEPLOY THIS CONFIGURATION TO HARD2**

The fresh local-only `exp102.q0_hgp_collapsed_smc.v0` diagnostic is terminally
`LOCAL_COLLAPSED_SMC_WEIGHT_OR_GENEALOGY_NOT_VIABLE`. It froze the same hard
`m08_c06,p=.04,d00,attempt022` cell, the HP64 quadratic 64-level collapsed
bridge, exact iid Bernoulli(.04) B initialization, unconditional systematic
resampling at every nonzero level, one exact eight-bit B-block heatbath sweep,
and eight independent N=128 populations (four column-major and four row-major
exact-base constructions). The immutable manifest SHA256 is
`ee3496f1d08e3e78db306f91b921a96d402c80a225b8c7e214978590e615f979`.
This is an exact-base population diagnostic, so planted P/U/L starts are not
substituted for the lambda=0 prior; physical zero remains outside this
nonzero-syndrome hard coset.

All eight population raws passed a complete deterministic seed replay
(`4f59ea1766432dece1b4d5bac263d906ba426cdef42c196e6fac0b016650b0f8`) and a
separate raw-only audit that never invokes the sampler
(`73aff5e55eda314b8382813bd6a1feb3c64a25d3eda6bc11071f9161db224a23`). The
report SHA256 is
`4bea937e5b6ae60dc4971d516b1c068da8f6cc1602d75947afd9903549311b70`.

The error mode is precisely repeated resampling genealogy collapse, not a
claim about q_top or posterior impossibility. Final distinct roots are only
1--5, root-family ESS is 1.00--2.74, and the largest root holds .49--1.00 of
a population. The median root ESS falls from 57.49/128 at stage 15 to
1.22/128 at stage 31 and remains near one. In contrast, incremental CESS at
many of those stages is about .9N and the per-stage largest normalized weight
about .01. Thus an apparently benign individual reweight can still erase all
independent roots after 63 forced resamples. This eliminates only the frozen
always-resample CSMC64 configuration; its raw cannot be extended, pooled,
retuned, used for q_top, sent to HARD2, or used for `READY_FOR_FORMAL`. A
fresh non-resampling or sparse-resampling algorithm would need a new contract,
new seeds, and a direct ancestry/weight proof. It is not an `IMPOSSIBLE`
conclusion.

**DTC21-S1 V0 LOCAL D=0 TRANSPORT FAILURE -- DO NOT DEPLOY THIS KERNEL TO
HARD2**

The fresh local-only `exp102.q0_defect_tempered.v0` diagnostic is terminally
`LOCAL_D0_TRANSPORT_NOT_VIABLE`. It froze `m08_c06,p=.04,d00,attempt022`, a
21-rung finite-syndrome-penalty ladder (`Kq=4` through exact iid `Kq=0`), 256
burn plus 2048 fixed measurement rounds, and eight independent legal P/U/L
trajectories per family. The immutable manifest SHA256 is
`751f76bec3831fd8fad39ee96972bd2a5e54a3da4a2e87a90ba202554decb337`.
At any finite rung, a fixed-clock `D=0` state has exactly the desired
hard-coset posterior conditional distribution; this target identity is not a
finite-budget mixing claim. Physical all-zero remains illegal for this
nonzero-syndrome cell, while shifted-coordinate zero is already P.

All 24 raw files passed the runner's complete deterministic seed replay. The
terminal report SHA256 is
`58f1dbb227d748edeb266fe42fefd74768dc2384d3bcf2dfc850b6339000e49c`. A
separate raw-only audit rebuilt the code, syndrome, P/U/L starts, labels,
defects, D=0 masks, counter invariants, and transport gate from NPZ with
`allow_pickle=False`, without calling the sampler; it passed with SHA256
`6990ea671153446e65592b29f4d1a3ad08c954abb9767476e1ff193e4df8cb2f`.

Defect closure is active but does not become global logical transport: P/U/L
have 166/61/201 D=0 label changes, yet only 1/2/3 label-delta rank, and only
one chain per family has at least eight label changes (the gate requires six).
P/U/L basis leave-return coverage is 15/8/19 of 64. U additionally has one
trajectory with only 108 D=0 clocks and only 61 family label changes, below
the 256 and 64 gates. Thus many defect leave-return events, nonzero swaps, or
ordinary state changes cannot be misread as global hard-coset mixing.

This rejects only DTC21-S1 at this frozen local budget. Its raw must not be
extended, pooled, used for `q_top`, sent to a remote HARD2 screen, or used to
authorize `READY_FOR_FORMAL`, tuning, held-out, or production. It is not a
mathematical impossibility result.

**CTT64-S1 V0 LOCAL TRANSPORT FAILURE -- DO NOT DEPLOY THIS KERNEL TO
HARD2**

The fresh local-only diagnostic exp102.q0_hgp_ctt.v0 is terminally
LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE. Its immutable m08_c06/.04/d00
attempt022 manifest is
f77add0a8b1825b117ac49ed85b3a3a138045cb233bed43fb691cac9bd31ff85;
it froze CTT64-S1, 512 burn plus 4096 fixed measurement macrosteps, and
eight independent P/U/L trajectories per family. The initially proposed
all-reduced-directions L start was rejected before freezing because it had
weight 229 versus P's 63. The frozen deterministic 1-to-3 reduced-logical
selection instead gives a legal different-label L start of weight 67.

All 24 raw files completed and passed the runner's full deterministic
seed-replay. The terminal report SHA256 is
9361b4290111a06b8e029b2b692df591c0e4e692bc463a5a3ee5f2ae7f2200b2.
An independent raw-only audit separately rebuilt the code, syndrome, P/U/L
starts, characters, labels, hard-coset residuals, weights, raw schema,
trajectory digests, counters, and transport summary with allow_pickle=False;
it passed with SHA256
ce2acd3e9cbc38b8d1be270248ed8bb94c8f21c936cde90dd534b90eb0697c9e.
The actual p=.04, 64-level path also passed a reference/Numba transcript
regression (21 passed).

The fixed CTT path does not deliver global logical transport. P and L have
only 228/212 measurement label changes and rank 3; U has 2414 changes but
only rank 13. Their basis/nonbasis leave-return coverage is respectively
20/59, 30/64, and 21/57 of 64 (P/U/L ordering), so every family fails the
frozen rank-64 and all-character requirements. The CTT path diagnostics are
consistent with this rather than substituting for it: P and L accept zero of
32768 path proposals, while U accepts one. Conditional A redraw can still
create ordinary label changes inside a small subgroup, which is why neither
those changes nor acceptance is treated as a posterior or convergence result.

This eliminates only this exact local CTT64-S1 configuration and budget. Its
raw must not be extended, pooled, or used for q_top; it cannot authorize a
remote HARD2 screen, READY_FOR_FORMAL, tuning, held-out, or production. It
does not establish mathematical impossibility.

**Q=0 LOGICAL-STRATIFIED V0v2 TERMINAL TRANSPORT FAILURE -- NO FORMAL
SAMPLING AUTHORIZATION**

The fresh, diagnostic-only deployment
`exp102_q0_lsi_v0d_20260723_9f0c473` is complete and terminally
`UNRESOLVED_LSI_IMH_V0_TRANSPORT`.  It is bound to source
`9f0c47370bac65059ed50507c95582f594d66df3`, archive SHA256
`edc677d396b5a89588dba526e4f38ce1fbb0480a52f476fc3498630f6b232d48`,
and source-manifest SHA256
`6557c30e888ef59cd5ca61fdd7bb0fb305019f90dae5b934b5bb9be179554e0b`.
The single-producer artifact, 48-task manifest, Linux/macmini algebraic
audits, four preflight traces, all 48 raw files, and both terminal analyzers
passed their identity and replay checks.  The byte-identical nd-3/macmini
reports have file SHA256
`89e5d6c4aaf0792e35050a2dacff1e205e490d3a5250ed1c2f3734c46b3729c4`
and internal report SHA256
`64a05c06d07d0af4c0b27daded97687e5f830f227f03886c92d7f117aadd65a2`.

Neither frozen proposal temperature passes any initialization family.  At
both `tau=.5` and `tau=1.0`, legal planted `P` and legal decoded-tail `L`
families have zero measurement accepted cross-label changes, zero source
coverage, rank zero, and zero basis/nonbasis leave-returns.  Exact-K0 uniform
`U` shows some one-way collapse but still fails every gate: respectively
`57/44` cross-label changes, `3/3` chains with at least eight changes, four
sources, rank three, `10/64` basis and `54/64` nonbasis leave-returns.  The
gate requires `128`, `6/8`, `16`, rank `64`, and all `64/64` characters.
This is direct low-temperature global-transport failure, not a raw,
portability, acceptance-rate, or ordinary-state-change failure.  The frozen
V0 raw may not be extended, pooled, or used for a posterior estimate.

`formal_authorization=false`, `production_authorization=false`, and no
proposal temperature is eligible for a HARD2 screen.  This says only
`UNRESOLVED_WITHIN_LSI_IMH_V0_BUDGET`, not `IMPOSSIBLE`; it does not authorize
tuning, held-out, production, or a change to common/illegal all-zero starts.

**CATALOG-FREE MLB8-J16 V0 LOCAL PREFLIGHT FAILED -- DO NOT DEPLOY THIS
KERNEL TO HARD2**

The separate local diagnostic `exp102.q0_mlb8.catalogless.v0` froze
`m08_c06,p=.04,d00,attempt022`, `P/U/L` with eight independent trajectories
per family, and fixed `(burn, measurement)=(512,4096)`.  It deliberately
removed the older reduced logical catalog: every macrostep used only the
stabilizer heatbath plus one exact 8-logical/16-generator block heatbath.  Its
24 raw files all passed an independent deterministic replay, stayed in the
hard coset, and had zero burn and measurement catalog attempts/changes.

The terminal local report is
`LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE` (report SHA256
`50fc413aee4cb01d83234aa0ef94764114744fa81027a07f553a6557dbaffe71`).
P had only 20 measurement label changes, rank 3, and 4/64 basis
leave-returns; U had 453 changes/rank 44; L had 2938 changes/rank 54.  The
initialization separation is therefore real global-memory evidence, not an
acceptance or raw-integrity fault.  This fixed algorithm/budget is eliminated
before remote HARD2 deployment.  The result is neither a mathematical
impossibility nor a posterior/q_top/formal exp102 result; its raw may not be
extended, combined, or reused as an estimator.

**Q=0 LOGICAL-STRATIFIED V0 ARTIFACT PORTABILITY CONFLICT -- sampler stage did not start**

The clean-source run `exp102_q0_lsi_v0_20260723_b9a08a4` completed only its
`01_artifacts` stage with an exclusive `SUCCESS` marker. Its source was
`b9a08a4905e4c8e999e0c9e5b3408f20e83c4436`, archive SHA256
`a53515a6af914077303b040caa6d3b5046af0054cf8bc3683c10289e1548ae53`,
source-manifest SHA256
`f151754e619f233e8abd544ea4a5d1bb6ec58cfc6c7f999866bd08680e0712a0`,
registry SHA256
`883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b`,
and V0 config SHA256
`55d410248adb5975aa162b0cc0406ffe1a0bfa8199435a04d5862b999b803f8e`.
It is terminally `CONFLICT_CROSS_ENV_ARTIFACT_IDENTITY`, not a sampler
failure, physics result, or mathematical impossibility.

The mandatory artifact equality check failed before a portable
`control/V0_MANIFEST.json` could be created. macmini's full artifact manifest
was `f90fc8d23be45e7b5122424e96fe5d6769aa73cf20339dcc0e6da814db67e64f`
(file SHA256 `3b319c59161f9413a7113644622721745ee95daf3628ca2c45bd4824b3efe6ee`),
whereas nd-1 produced
`6171de3b81a6f84ba070ba62fb7c52620687284c860d0a0bc9513b8a51d74b98`
(file SHA256 `778e5f5e78bb1bb25d6f806f19bb5d9577781714f9ccd1f70e2b36a28196e1bd`).
The common classical matrix, codebook, syndrome, frame, registry, source and
all V0 scalar settings agree. The differing dependency stacks do not:
macmini used `ldpc/numpy/scipy=2.4.1/2.4.1/1.17.0`, while nd-1 used
`2.3.7/2.3.4/1.16.3`. Their MILP base-anchor SHAs differ, and 112866 of the
113566 BpLSD decoded candidate states (112093 recorded weights) consequently
differ; the rank-complete 128-anchor catalog, S-tail schedule and both
proposal SHAs therefore differ too. Both artifacts have `113566/113566`
valid candidates, and a local direct GF(2) audit confirmed the nd-1 artifact's
128 retained states, labels and transcript linkage. Internal algebraic
validity does not make two differently frozen proposals interchangeable.

No V0 manifest, cross-node preflight, sampler trajectory, sampler raw,
transport report, tuning, held-out or production task was created. The pulled
evidence is retained under
`validation/014_q0_logical_stratified_v0_20260723/remote_run/exp102_q0_lsi_v0_20260723_b9a08a4/`;
the run must not be retried or repurposed. A successor requires a newly
reviewed contract that either pins one identical decoder/solver stack for all
artifact builders or freezes one producer artifact and limits other platforms
to algebraic verification, while preserving fail-closed rejection of any
cross-version recomputation. It must also retain the P/U/legal-tail
initialization red-team and cannot treat a physical all-zero state as legal
for this nonzero-syndrome hard coset.

**Q=0 LOGICAL-STRATIFIED V0v2 IMPLEMENTED / BOOTSTRAP FAILURE / NOT YET SAMPLED**

The fresh successor contract is
`exp102.q0_logical_stratified.v0.v2`, documented in
`validation/015_q0_logical_stratified_v0b_20260723/`. It designates `nd-1`
as the one artifact producer and requires `nd-1/nd-2/nd-3/macmini` to audit
the exact frozen bytes algebraically rather than independently rerunning
BpLSD/MILP. It also splits every P/U/L family across both sampling nodes and
requires measurement-only rank-64 accepted label deltas plus leave-return
coverage for all basis and frozen nonbasis characters. The original source
validation passed 847 exp101+exp102 tests. The bootstrap repair adds a `set -u`
empty-prerequisite wrapper regression; its follow-up full suite passed `848
passed, 4 warnings` in 311.78 seconds. The subsequent portable-lock repair
repeated the full suite with `848 passed, 4 warnings` in 307.14 seconds.

The first clean-source V0v2 bootstrap attempt,
`exp102_q0_lsi_v0b_20260723_5aab1d7` (source
`5aab1d73d3ebf258ee9757f60b4a2343cb0c506a`, archive SHA256
`e69348066f188d63ee58d4b6c84ce8fdc66d80f0821320b7584c2f2788c1b2fb`,
source-manifest SHA256
`f7dfd3fe5b9671c1be531b75a20b25a1c4ce2fdb1a16ea1f8cb69e590e21533d`),
stopped before the artifact command or even its stage directory was created.
Under `set -u`, `run_v0_stage.sh` expanded an empty root-stage
`prerequisites` array. This is terminally `BOOTSTRAP_WRAPPER_FAILURE` for that
deployment, not an artifact, sampler, physics, or convergence result. It
created no artifact, manifest, cross-host audit, preflight, sampler raw,
transport result, tuning, held-out, or production task, and must not be
retried in place. The successor must use the repaired wrapper, a new clean
source archive, and a fresh run ID; it remains diagnostic-only even if it
passes.

The next clean-source attempt, `exp102_q0_lsi_v0c_20260723_a49910a`, reached
its Linux-only artifact and manifest stages (artifact-manifest SHA256
`9faaaafd4db1fd017d400967a4b989c0627835b46c64f13e0426297ecbebecfa`,
V0-manifest SHA256
`4d10720065998dd4d6acd1f38acbc58fb1f8c1878d96c9c71283005d4f736720`)
and all three Linux algebraic audits. Its source was
`a49910a07acf0949e5e7b6fe39532e5d30a81f9f`, archive SHA256
`b06ab2b48b9d57a32027eaec9b249582dd70f1ac346583a55a89e614d509051d`, and
source-manifest SHA256
`157d9ff2d7dd159868512da578409b84f2b026f572008abcd1c3f5fa17357695`.
The mandatory macmini audit stopped before module execution because the V0
wrapper called Linux-only `flock`, which is absent from conda-12. This is
`BOOTSTRAP_LOCK_PORTABILITY_FAILURE`, not a disagreement of the frozen
artifact and not sampler evidence. The local partial stage contains only an
empty lock file; no macmini audit payload, preflight, sampler raw, transport
result, tuning, held-out, or production task exists. The v0c artifact binds
the old source/archive identity and must not be reused after the portable-lock
repair; a fresh clean-source run is required.

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
