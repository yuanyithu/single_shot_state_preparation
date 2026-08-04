# exp102 历史归档

本文件是不进入常驻上下文的冷归档，保留 validation 001--066 期间从常驻文档下沉的原文。当前状态只读 `status.md`，单项证据从 `validation/INDEX.md` 进入对应 validation 目录。

## 来源与完整性

- 基准 commit：`bfaf15e93c36de8d17aa42574238f784d2f86159`。
- 旧 `status.md` 全文：blob `a46d4e0cc1ba8ba6240c0fb517313975f67ad39a`，SHA-256 `29ffdd6b9b1c293be26046feabeddf597043a0ad4412aaa24d1f7d6d33ececa6`。
- 旧 `AGENTS.md` 的 exp102 编年史（第 21--362 行）：blob `e37220a071983cf520f8f9109e95c94e2db60ef1`，本段 SHA-256 `9d4d33fd5da73dbc622f47748a942c6f15ac9007a85febd1fe7892c687184a8f`。其余仍有效规则已蒸馏至新 `AGENTS.md`，legacy 3D 材料继续由 Git 历史保留。
- 旧 `笔记/实验报告.md` 的 exp102 原文（第 10--83、186--190 行）：SHA-256 `e5ef33a7da0ee65e2c6bf057ba6033d11e0461b90cd31fb8c842f0fe5d301051`。
- 整理前 stash 中的 `CLAUDE.md`：blob `907c748254eed878874747f2ccc70e793a9f9b45`，SHA-256 `9cd8ca8d75847f2e31d6da1c92c5fd467e28573b251ab683f548dd460ef66fc3`。它是旧 `AGENTS.md` 的严格子序列，无独有文本，因此不再重复归档。

---

## 附录 A：旧 status.md 全文

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

---

## 附录 B：旧 AGENTS.md 中的 exp102 编年史

**exp102 当前为 `Q=0 HGP HARD-PAIR DIAGNOSTIC UNRESOLVED / PRE-PILOT`，不是已有物理结果。** 正式历史契约仍为
`exp102.physics.v1 / exp102.q0_pt.v1 / exp102.scan.v1`。固定 Q32 + multi-swap PT-v2 已因 96 条轨迹
认证往返总数为 0 而 `EXHAUSTED`；不得追加 S128、延长轮数或复用 raw。随后
`exp102.q0_pa.discovery.v1` 的四个 transport autopsy 因条件 attempts<200 均为 `INCONCLUSIVE`，
`C192-2/B96-1/B192-1/B96-2` 又在两个 hard cells 上因 genealogy 塌缩全部失败；PA 零通过分支同样
`EXHAUSTED`，禁止 B384-2 rescue。

2026-07-30，fresh local-only validation 066 已从 clean source
`bc47ae26dd26203f2b9c902feca2a10ea797c798` 完成 delivery-aligned gate calibration。五个冻结 selection
points 的 validity/outcome 依次为 `INVALID/FAIL/FAIL/INCONCLUSIVE/PASS`（首点在 report 中记录为
`INCONCLUSIVE + NO_FINITE_CALIBRATED_MULTIPLIER`），选中 32 IID multinomial trajectory groups x
16384 independent draws（不是 MCMC chains/clocks 或 ESS）、
multiplier `4.809673164164152`，fresh confirmation 为 PASS；终态为
**`LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED`**。4,372,205-byte report self-SHA/file SHA 为
`d255c67ee0a91985e933ccea8a9616c63e832e37c19cd16dc7eb5e35f05e5a0a` /
`f11a3eb137793ce2bbe43734db82240cde45bafbdd57a2a1e6f97d520dad6ed8`；独立 auditor 从 seeds 在冻结的
NumPy 2.4.1 + `default_rng` + PCG64 同环境逐位重建 histograms、full-label `q_top`/`D2_norm`、
delete-one、calibration、Wilson、selection 与 confirmation，audit self-SHA/file SHA 为
`485b789c3a86893662241ab0e529358fedde18b695c3514233adc492236261b3` /
`3975de5eb1d9cebcc467efdd67d956dcfa4b98e4c3b205011a549d6cf8d7822c`。receipt 不是 persistent trial raw，
也不提供跨 NumPy 版本 portable RNG 保证。两个 common-wrong `EXPECTED_KNOWN_BLIND` controls 的真实
`D2_norm=.0625` 而 candidate PASS rate=`1.0`，实证了共同错误收敛盲区，
所以该 PASS 只认证本地 scalar gate 的 operating characteristics，**不证明 mixing、transport、target-basin
或 unvisited-tail coverage**。项目状态保持 **`BLOCKED_BEFORE_REMOTE`**；large-k orthogonal confirmer
`LARGE_K_ORTHOGONAL_CONFIRMER_PORTFOLIO_UNFROZEN`、`FUTURE_SCHEMA_RUNTIME_COVERAGE_INCOMPLETE`、
`CAMPAIGN_BUDGET_UNAPPROVED`、`STAGE3_MULTI_COMPARISON_MULTIPLICITY_UNFROZEN` 四项仍未解决。不得据此
启动 m3、remote、formal、held-out 或 production；P/U 对抗初态和 transport/
Rhat/ESS/burn-crossing/独立确认门仍必须保留。接手先读 `validation/066_q0_delivery_gate_redesign_20260728/`
的 README、report、independent audit 与 `status.md`。post-run conda-12 完整回归为
`1090 passed, 4 existing warnings`。

`exp102.q0_global.discovery.v1` 已实现 logical catalog、hard-coset cluster/joint heatbath、独立
defect trace、m3 full-sector TI、三节点 digest/runtime、72h schedule 与 control freeze。其第三个且
终止性的 immutable run `exp102_q0_global_20260721_204b37d`（source
`204b37d8e00e7d11ffa2b6766b90d947892e179d`）三节点 worker 与 canonical digest 全过，所有 hard/defect
候选也都可用 T3；但必需的 TI contingency 在 nd-2/nd-3 投影为 116275/251241 秒，超过冻结的
79200 秒窗口，故专用 worst-node consensus 为 `RUNTIME_EXHAUSTED`，在 bias/screen 前终止。**节点
worker 的 SUCCESS 只表示测试与报告生成成功，不等于 preflight PASS；后续 stage 必须看到 aggregate
runtime/preflight status=PASS。** 旧 combiner 对合法 exhausted report 抛异常的问题已修为持久化
`RUNTIME_EXHAUSTED`，但下游 PASS 门槛未变。完整证据见
`validation/010_q0_global_runtime_exhausted_20260721/`；前两次基础设施失败审计见 008/009，不得原地
重跑。当前全范围结论只能是 `UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`，不能写 `IMPOSSIBLE`，
也不能外推为某个参数点的物理失败。该 discovery 自身的 screen/HARD2/confirmation/resolution/TI
sampler raw 均不存在。

用户随后批准了独立 `exp102.q0_global.screen_diagnostic.v1` 的 `HARD2+EASY3` 测试，最高权限仅为
`DIAGNOSTIC_SCREEN_PAIR_FOUND`。修复后的 fresh run
`exp102_q0_screen_diagnostic_20260721_342dd5b`（source `342dd5bc0fb2c7694dbc58a8d0f2d92689c24991`）
已通过三节点 preflight/digest/runtime，选择 T3，并完整运行、逐位 replay 15/15 bias 与 1280/1280
measurement raw（`reused=0`）。终态为 **`UNRESOLVED_NO_HARD_COSET_PASS`**、`selected_pair=null`：
`RC8-QC1/QC4/J08/J12/J16` 和 `DT16/DT32/DT64` 均为 0/5。hard-coset 的 25/25 cell summaries
均超过 P/U `q_top` 差的绝对 0.04 门槛，全部 U family 均 `Rhat>1.05` 且 ESS<400；480 条 defect-trace chains 的
fixed-clock D=0 observation 与完整 excursion 均为 0。故这是冻结 T3 预算内的 sampler 收敛失败，
不是基础设施失败、`IMPOSSIBLE` 或正式参数点结论；不得追加链长、改 gate 或直接进入 full-range。

本地 conda-12 verified-archive replay 的 raw/state/label/counter 仍逐位一致，所有门禁与终态一致；仅
62 个派生 `core_seconds` 和 18 个派生 ESS 跨平台相差最多 4 ULP。证据比较必须分别验证 report/package
self-hash，并只对这两个白名单字段允许已审计的 4-ULP 上限，不能把整个 report 做 byte equality，
更不能对 gamma/raw replay 使用 ULP 容差。4096 项 Decimal gamma 的 versioned SHA 仍固定为
`a2c459ec9438e23f863c44528ac093c5b93d891b6a8bec0278b873fe47f2459a`；禁止恢复平台 `libm`
fractional power。首个 `5e1f5aa` run 保持 `CONFLICT_CROSS_NODE_GAMMA_LIBM` 审计，15 个旧 bias raw
永不复用。接手先读 `GLOBAL_SCREEN_DIAGNOSTIC_CONTRACT.md`、`validation/011_*` 和 `status.md`。

2026-07-22 隔离 HGP v2 诊断已经完成，fresh immutable run 为
`exp102_q0_hgp_screen_v2_20260722_4d134ee`（source
`4d134ee7ca25125d341eb11cbfa34d6856514101`、archive
`ad72d2c7039192be721b87ce7c96c5da577af05acd37cacd9167e26a773d9027`、manifest
`5bafae76b06ff46557ae8315bb281a42256e7e4e50ed2e9dae868695114b8ff8`）。三 Linux 节点 full exact
consensus PASS 并选中 T3，本地 conda-12 对 portable projection 与四条 MAM acceptance-decision probes
逐位复核为 `PORTABLE_PASS`；12 个 full mismatch 全部且仅是预注册的 nonportable float。固定 ownership
下 384/384 measurement 与 2/2 IS 完成，nd-3 full replay 和本地 terminal audit 验证 386/386 raw；终态为
**`UNRESOLVED_MAP_MIXTURE_FAIL`**，不是 infrastructure `CONFLICT` 或 `IMPOSSIBLE`。terminal package
identity 为 `233e31e599180153f979a30dc971e8e8128be64505fd0572d68bc1ae87a64041`。

方法结果：HP64 为 5/5，是明确的 promising candidate；HP32 为 3/5，其中 m3 只是单个 B-character
`0.0404396>.04` 的边缘 fail，m5 则是明确 B 慢模态（U max Rhat `1.1552`、min ESS `327.0`、pooled
Rhat `1.1172`）。MAM-IMH8 仅 1/2，m8 的普通 P family 与 P/U B family 均未过 Rhat/ESS，16 个 B
characters 初始化族不一致。HP64/MAM 的四个 HARD2 family-cell comparison 全为 0/4：m6 的
`q_top=.14587/.16241` 绝对差虽小于 .04 但约 30 SE，m8 为 `.91317/.99273`，绝对差约 .0795。
因此一个 sentinel `(m,p)` 也未认证，HP32/HP64 属同一机制，不能互相当独立确认。

本 run 暴露的新关键坑是：**真实 state change 不等于 logical transport**。m8 的每条 MAM 链虽至少有
1947 次 measurement state changes、rate 至少 `.0594`，P/U 合计分别有 39899/40735 次，但其中只有
330/288 次改变 logical label（约 `.827%/.707%`）；典型链只见 3 个 label。两个不同的最小权重 m8
MAP anchors 的 64-bit logical coordinate 完全相同，`theta_logical=.08/.25/.5` 的 proposal components
在每族 524288 次总 attempts 中零接受，global IS ESS/总 acceptance 主要测到同扇区运动。后续 MAM
viability 必须预注册 anchor signature coverage、逐 component 的 accepted cross-signature moves 和
logical-character mixing；不得直接加长 T3 或只优化总 acceptance/state changes。

五个 sentinel syndrome 权重为 `83,160,39,58,125`，物理全零 bit string 全不在目标 hard coset；只有
显式定义 `x=e xor epsilon_true` 时，`x=0` 才对应现有 P 初态，并不是新的起点。16 条 P 链共享该合法
planted state 但 RNG 独立，16 条 U 链使用独立 exact-K0 hard-coset states。不得用“所有链从 0 开始”
制造表面收敛，也不得删掉 P/U；若扩充初态，应增加合法、结果无关、按 logical signature 分层的对抗
初态。HP 每轮精确重抽 `A|B` 仍可能用条件噪声掩盖 B，故 B-bit/row-column/dense-character 与 full
logical/energy/weight 门都必须保留。有限 characters 和 16 个 U 仍可能共同漏 basin，HP64 通过也不是
混合证明。接手先读
`HGP_GLOBAL_SCREEN_CONTRACT.md`、`validation/013_q0_hgp_global_screen_20260722/` 与 `status.md`。

v1 `exp102_q0_hgp_screen_20260722_2e6ba2a` 的 Linux preflight 虽 PASS，但本地发现 MAM float/IS full
digest 漂移而终止 `CONFLICT`，从未产生 measurement raw；仍不得续跑或复用。v2 最高权限本就只有
diagnostic，当前也没有 `READY_FOR_FORMAL`；EASY3 独立确认、fresh T/2T、扩展 panels、正式 tuning、
held-out 与 production 全部仍缺失。

2026-07-23 的 local-only `CSMC64-B8-S1-N128` 已终止为
`LOCAL_COLLAPSED_SMC_WEIGHT_OR_GENEALOGY_NOT_VIABLE`：它在 m8 hard cell 上从数学正确的
`lambda=0` iid Bernoulli B base 出发，八个 N128 populations 均通过 exact small-HGP / reference-Numba、
完整 seed replay 和独立 raw-only audit，但 63 次**无条件** systematic resampling 使 root ESS 在 stage 31
已降至 median `1.22/128`、终态仅 1--5 roots (ESS 1.00--2.74)。其 fresh 的无 resampling 后继
`CAIS64-B8-S1-N128` 也已终止为 `LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE`：八个 exact-base
paths 虽无 clone、且 replay/不调用 AIS engine 的 raw-only audit 均通过，终态 full-path ESS/N 仍只有
`.0078125--.0100431`（门槛 `.25`）、最大权重 `.872760--1`（门槛 `.10`），cold endpoint median ESS
约 `1.000002/128`。关键新坑是：**逐级 CESS 看似约 .9N 不保证独立根，去掉 resampling 也不保证完整
AIS 路径权重不塌缩；必须看 full-path ESS/最大权重及独立重算的 AIS 时序公式。** 两份 raw 均不得补长、
合并、作 q_top 或送 HARD2；不能据此说全部 collapsed SMC/AIS 不可能。后续若评估新 annealing，必须
另立 fresh config/seeds/raw，并同时报告 full-path weights、root ancestry（若有重采样）和为何其 AIS/SMC
weight formula 对所用可逆 mutation kernel 正确；不要用 P/U/L 或物理零态替换这个 exact-base initializer。

2026-07-24 的 fresh local IID-MIS m8 诊断终止为
`LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_UNRESOLVED`：每个 block 对多个 proposal 等量抽样时，目标
mixture density 必须是同样的**均匀** mixture，不能事后改 mixture weights。旧
MAM/T05/mixture 的 min block ESS 仅 `22.09/28.91/28.78`（门槛 `50`），max weight 为
`.1522/.1629/.1574`（门槛 `.10`）。其 component-provenanced 后继
`LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_UNRESOLVED` 又表明一个更细的契约坑：BP-SYS-F64/R64
单独都通过 ESS/最大权重 gate，但预冻结的三源等权 mixture 因 MAM-source blocks 只有 ESS `23.48`
和最大权重 `.14695` 而失败。**只要 stress proposal 被放进 estimator 的 mixture，mixture gate 就会让它
实际拥有 veto；不得在看见结果后剔除它并把 BP-only 说成通过。** 后继若真要评估 BP-only，必须另立
fresh contract/seeds/raw，明确 stress source 是仅报告还是 estimator 的一部分。

更根本的坑是：**full support、cross-proposal agreement、低 jackknife SE 甚至看似很高的 collision 值
都不能认证未观测 target tail；反过来，低温链未在 measurement 内遍历全部 logical directions 也不是
purity 正确性的数学必要条件。** 不得加样、放宽门或把诊断数报为 q_top；任何 successor 都必须保存
source 之外的 anchor/component provenance，并有独立 tail/normalizer 证据或确认方法。不要以所有链从
P/物理零态开始来规避这个问题；非零 syndrome 的物理零态仍不在支持上。

2026-07-24 的 strict depth-two collapsed-B envelope 又确认一个容易忽略的区别：**更快地算出一个
global upper bound，不等于该 bound 对交付目标足够紧。** 在 m8 hard sentinel 上，width-25 contraction
虽只用 `3.14s`、峰值约 `2.50GB`，但两个非 planted retained B marginals 的 tail/retained 上界仍是
`9.40e84`（目标 `<=.01`）。不要把 runtime PASS、更多 MCMC clocks、换成共同 P/零初态或漂亮的局部
ESS 当作缩小这 87 个数量级 global-mass 缺口的理由；要继续必须另立能说明 tightness 与资源的契约，且
depth-two 的负结果只排除该 factorized envelope，不可泛化成数学不可能。

2026-07-24 的 local HCA 调查也已完成。035 的 single-copy character WMC 在两种编码下 min-degree
width 都为 `378`，min-fill 在 120s 前已经超过 width `102`；036 的最佳 linear-code trellis exponent
为 `584`，均不能作为当前 exact-normalizer 路线。037/038 的 tensor-logical Houdayer 坐标在真实低能对上
只产生整对 replica exchange；039 唯一预注册的 canonical-reduced 坐标虽在 120 个 L/L 对中有 102 个
产生真实新 unordered pair（例如 `67+67 -> 63+71`），但 P/L 仍只是 whole swap。随后的精确
HCA-RHB1 pair kernel（每 replica 832 random-scan coordinate heatbath 加一 HCA；small-HGP 完整
stationarity、replay、raw-only audit 均过，固定 `128+1024` clocks runtime 约 22s）在 fresh PP/UU/LL/PL
八对各族屏幕终止为 `LOCAL_HOUDAYER_PAIR_TRANSPORT_UNRESOLVED`：PP/LL/PL 在 normalized pair weight
约 `.03886` 一致，LL 有 1091 个真新 pair event；但 exact-K0 U/U 为 `.1486354`，相对 PP 差
`.1097799`、basis gap `1.4603271`、early/late fail，且 8 条 U/U 的 1024 个 measurement HCA 每条都只
是 whole-pair exchange、零真新 pair。**可见的低能 L/L recombination 不能替代从均匀 hard coset 到目标
低能区的运输。** 不得删除 U/U、延长/合并 041 raw、改阈值或改成共同 P/物理零初态；非零 syndrome 下物理
零非法、平移坐标零就是 P。该结果只排除这个 HCA kernel/budget，不是 `IMPOSSIBLE`，也不授权 remote、
formal、held-out 或 production。接手需读 `validation/035_*`--`042_*`、尤其 041 raw audit 与
`status.md`；下一候选必须先证明其如何解决 U transport，并保持独立 tail/normalizer 或独立 confirmation
路径，不能只优化 HCA event 数、acceptance、ESS 或低能 L/L 图像。

同日的 `043_q0_collapsed_houdayer_structure_feasibility_20260724` 进一步排除了一个表面上很自然但
其实偏离慢变量的后继：在精确 collapsed-B 边缘上，完整 factor component 的 Houdayer swap 虽代数正确，
但 16 个 P/low-energy-L、120 个 L/L 和 64 个 P/rank-complete-L 冻结对的 B masks 全部相同；物理 logical
label 的低能变化全在已被 HP64 精确热浴的 A 中。独立 U/U 对有 284 个不同 B bit，却只有一个完整 component，
仍只是 whole-pair exchange。small-HGP 的转换、不变量、involution、detailed balance 和 stationarity 都已穷举
通过，故终态是 `COLLAPSED_B_HCA_NO_LOW_ENERGY_RECOMBINATION`，不是实现失败。**“对真实 state 有变动”或
“pair 代数正确”不等于推动交付所需的 B/logical 慢变量。** 不得据此实现 HP64+B-HCA、优化其 acceptance/state
changes 或把它包装成独立 confirmation；该结构性负结果只排除该 frozen collapsed-B HCA，不表示 HP64、HCA、
q=0 或后验数学上不可能。

同日的 `044_q0_bp_dominance_witness_feasibility_20260724` 钉住了 importance/rejection 路线的另一处
容易自欺的环节：两个 BP-systematic source 的有限 ESS 看起来很好，并不自动给出 `pi/q` 的全局上界。对
1691 个预冻结合法 planted/logical/systematic-coordinate witnesses，以精确三 component mixture density 和
outward Decimal rounding 计算后，唯一无需解决原问题的 normalizer 上界
`Z<=.96^-1600` 只给出 `sup(pi/q)` 的微小下界（forward `5.53e-63`、reverse `2.54e-53`），所以终态是
`BP_MIXTURE_REJECTION_ENVELOPE_WITNESS_INCONCLUSIVE`，绝不是 BP 通过。**一个 proposal 的局部 overlap、
低 jackknife SE 或 full support 都不能替代 tight global normalizer/tail bound；用过松的 `Pr(y)<=1` 也不能
把“没有找到坏 witness”说成 coverage。** 不得据此开 BP-only IID/rejection sampler、报告 q_top 或以 P/common
start 绕过 MCMC 对抗初态；要继续此路线须先独立得到紧的 hard-coset normalizer 上界，而那正是尚未解决的
global-mass 问题。

随后 fresh `exp102.q0_bp_imh.local.v1` 直接把 BP-SYS-F64/R64 的精确 full-support mixture 用作
independence-MH proposal；small-HGP 完整 transition matrix、detailed balance/stationarity、18 项 focused
测试、24/24 raw replay 和不调用 sampler/runner 的 55296-step `allow_pickle=False` audit 均通过，但终态为
`LOCAL_BP_IMH_TRANSPORT_UNRESOLVED`。P 与 8 个不同合法低能 L 在 burn/measurement 都是零真实移动；U
虽在 burn 用 1--3 次真实移动冷却，却全部落到同一个 weight-62、P-label state，measurement 仅 0--2 次真实
移动。P 最大 measurement log acceptance 也只有 `-53.13`，L 最好为 `-47.79`（最差 `-88.69`），说明
proposal 对 high-`pi/q` 低能态严重供给不足；大量 U accepted counters 是同态 self-proposal，不是运输。
P/L 的 full-label `D2_norm=1`，U/L 为 `.998413`。注意 full-label D2 是本次 raw 前补上的必要门：相同 purity
和全部 basis means 仍可能对应互不相交的 sector supports。045 v0 仅因 relative output receipt 路径错误在
首 raw 后终止为 infrastructure failure，raw 禁用；046 v1 使用 fresh contract/config/seeds，零复用。不得把
BP 当 U 冷却器再接旧 full-row Gibbs 就宣称成功：BP 把 U 全送入 P logical label，而旧 full-row 又把 P/L
送入同一冻结 B basin，三族一致可能只是共同塌缩。后继必须有结果无关的 high-`pi/q` signature/basin coverage
与独立 B/tail 证据，不能只优化 accepted/self-loop 计数、统一初态、延长链或直接送 HARD2/remote。

2026-07-24 的 047--051 又排除了一个“共同落入低能 basin 就算收敛”的盲区。truth-free dressed logical
XOR catalog 虽代数正确且 signature rank=64，但 T3 下 BASE/P 可达 rank 仅 `4/1`，并会把全部低能 L
向同一 label 拉回，故终止为 `LOCAL_CENTER_PRESERVING_STRUCTURE_NOT_VIABLE`。exact random-scan
full-B-column Gibbs 的 small-HGP detailed balance/stationarity 与 bit replay 通过；但 049 短跑中 P/L 的
B 几乎冻结，`A|B` 精确重抽仍会制造 visible logical-label changes，U 的 B weight/likelihood 仍完全分离，
所以 **logical/state change 不能替代 B 慢变量门**。050 的两个 truth-free MAP anchors 只证明 T1 下某一
两列桥有足够 expected first departures，不是 sampler pass；051 独立重算保留 047/049 失败和 050 的窄权限。

fresh `exp102.q0_random_full_column.t1_m8.v0` 已冻结并完成三节点 preflight，但没有 measurement raw。它不用物理零态（该非零
syndrome 下非法；shifted zero 就是 P），而用 P、独立 exact-K0 U、两个 B-distinct MAP 及 8 个低能
B/logical-distinct S starts 各 8 条，固定 `2048+8192` clocks。S 中故意保留一个与 MAP 同 B、不同 logical
label 的起点，以区分 A/logical redraw 与真实 B transport。三节点 clean-archive preflight 必须在固定四并发
下 exact digest 一致且 replay-inclusive 单 trajectory 投影 `<=2h` 才能启动；本地四并发超时不具有远端
判定权限。门禁必须保留 character-U-statistic q_top/D2、full/B weight、B likelihood、全部 B bit/row/column
和 dense characters、logical characters、Rhat/ESS、constant-character burn crossing、MAP 双向 basin visits。
immutable run `exp102_q0_rfcg_t1_m8_20260724_6fa489f` 的三节点 mass/transcript exact consensus 通过，
但 nd-1/2/3 replay-inclusive 单 trajectory 投影分别为 `24701.47/24812.06/29871.42s`，全部超过冻结的
7200s 上限，故 aggregate 终态为 `RUNTIME_EXHAUSTED`，measurement raw 数为 0。本地 conda-12 独立审计
复核全部 self-hash、control/schedule、40-task ownership、runtime 算术和 raw absence，audit SHA 为
`817425dbaa6a9e5d90d03d34efe16f957beb7424eddd27dcde7cf12d60d75c6d`。这不是收敛失败、物理参数点失败或
数学不可能，只说明该冻结实现/clock/replay/并发无法满足两小时资源契约；不得绕过 gate、缩短链或事后改 cap。
没有 m6/HARD2/formal/held-out/production 权限；若做性能后继必须另立 fresh contract/source/seeds/raw。

其 memory-streaming 后继 validation 053 在 macmini 上 12/12 完整 CDF byte equality、`4.9391x` speedup
和 `2432.39s` T1 投影均过，但 fresh 三节点 run
`exp102_q0_streaming_preflight_20260724_de68bbc` 终止为 **`CONFLICT` 且独立 runtime-exhausted**，没有
T1 raw。Linux 三节点都只有 `U0,column=11` 的 legacy-dense/streaming CDF byte mismatch；proposed
streaming CDF 的完整 SHA catalog 和四条 PortablePrng sampling/replay transcript 反而在 macmini/三节点
完全一致，所以不得把它误写成 streaming sampler 跨节点随机漂移，也不得事后忽略冻结的 any-mismatch
门。更独立的阻断是 nd-1/2/3 speedup 仅 `2.5911/2.5372/1.3823x`、T1 replay-inclusive 投影为
`8797.83/9144.89/17760.30s`，仍未过 `4.2x/7200s` 门。audit SHA 为
`6426a1a01c01747f474d587a10cdb6db9e53db09112193499a8f9307adb7640f`。后继若利用正质量范围改成直接
weight/fixed-block exact heatbath，必须 fresh source/contract 并重做 small exact、underflow、portable
replay 与三节点 runtime；不得复用 053、删 P/U/MAP/S、缩短 T1 或先看 q_top。

2026-07-24 的 direct-positive fixed-block 后继 validation 054 已通过。fresh immutable run
`exp102_q0_direct_block_preflight_20260724_61d605a`（source
`61d605a5e27db0970457736c72d1c45d72a12b10`、archive
`61bb87e70320f7371504ea99c320e49baf1140b4ac9d3050fc9a3b742d5a7bec`）在 macmini/三 Linux 节点精确复现
12 个 frozen block-subtotal SHA 与四条 P/M0/S0/U0 PortablePrng sampling+replay transcript；三节点
replay-inclusive T1 投影为 `4144.85/4139.52/5454.14s < 7200s`，aggregate=`PASS`。完整 `2^24`
权重检查的 worst scaled absolute/relative/TV 为 `2.020606e-14/7.290711e-14/4.148991e-15`，候选
log-weight lower bound `-221.658`，没有接近 binary64 underflow。独立审计为
`INDEPENDENT_AUDIT_PASS_DIRECT_BLOCK_PREFLIGHT_CONFIRMED`（SHA
`9646c6f92070024680728bf377e802e647b48a2b66ca6210c89c436fbd70f539`）。

该 PASS 只证明 `RFCG-C24-DPB12-S1` 的 exact conditional、portable replay 和资源可行，只授权 fresh m8
T1 diagnostic；不是混合、q_top 或参数点认证。T1 必须另立 contract/source/seeds/raw，保持
`2048+8192` fixed clocks、full replay 和 `P/U/M0/M1/S` 各 8 条独立轨迹，并保留 validation 052 的
full/B D2、weight/likelihood、B bit/row/column/dense、logical、Rhat/ESS、burn crossing、双向 MAP
basin 与 B-column/label-change 门。非零 syndrome 下物理零态不在支持集，shifted zero 已是 P；全部从
P/零态开始只会掩盖慢混合。T1 未通过前不得运行 m6/HARD2 或 formal/held-out/production。

fresh successor validation 055
`exp102.q0_random_full_column_direct_block.t1_m8.v1` 已终止为 **preflight
`RUNTIME_EXHAUSTED`，measurement raw=0**。它保持 validation 052 的 P/U/M0/M1/S 几何，但 fresh control
重抽全部 schedule seeds 与 logical/B characters；四类 seed 与 052 overlap 均为 0，并 byte-bind validation
054 的两个 sampler 源文件与 portable artifact。pre-run red-team 用真实 miniature direct raw+full replay
修复了 direct engine 身份、`state_label` import 和 B-likelihood sum-order 三个 dormant analyzer 坑。

前两个 schedule attempt 都在 control 前因提前创建 fresh run root 而基础设施失败；第三个 immutable run
`exp102_q0_direct_block_t1_m8_20260724_146ef55_r3` 才是权威证据。最终 source 的完整 054 portable/runtime
preflight 三节点 exact consensus PASS，T1 投影 `4216.16/4149.15/4549.57s`；但 055 自己冻结的 probe 只测
`2+8` updates，却把含固定初始化/runner 开销的总时间线性外推到 10240 updates 再乘 2，得到
`9272.13/8779.07/13638.99s > 7200s`，所以 schedule 正确阻止 measurement。独立审计状态为
`INDEPENDENT_AUDIT_PASS_PORTABLE_PASS_T1_RUNTIME_EXHAUSTED_CONFIRMED`（SHA
`00622194dc370a66e08a0b94a7108b324aa49322de648fda7656f2c6ed5fc665`）。这不是 sampler/参数点失败，
也不得事后解释成 PASS；后继须另立 fresh contract/source/schedule/seeds/raw，用代表 steady-state 且包含 full
replay 的 probe 或冻结 intercept/slope 估计，同时不改 T1、7200s cap、五类初态和统计门。合同与证据见
`RANDOM_FULL_COLUMN_DIRECT_BLOCK_T1_CONTRACT.md`、`validation/055_*/`。

fresh validation 056
`exp102.q0_random_full_column_direct_block.t1_m8.v2` 已终止为
**`UNRESOLVED_DIRECT_BLOCK_T1_M8`**。immutable run
`exp102_q0_direct_block_t1_m8_v2_20260724_6933e31`（source
`6933e319b27840976f34e27c0d11313b6973cbe3`）先通过完整 054 portable preflight 与 fresh 两长度 runtime
preflight，最坏 factor-two T1 投影 `6550.3213s<7200s`；随后固定 `14/13/13` ownership 完成 40/40 raw，
无复用。primary report/raw-set SHA 为 `e1bfb340...6015 / a267ded6...259`；不调用 sampler/replay/analyzer
的 raw-only audit 独立重算全部 states/B/labels/likelihood/q_top/D2/Rhat/ESS/family/pair/terminal gates，状态
`INDEPENDENT_RAW_ONLY_AUDIT_PASS`，SHA `ada30d3c...b08e`。

这是冻结 T1 下的 sampler 收敛失败，不是 runtime/infrastructure、物理 q_top 或 `IMPOSSIBLE`。P/M0/M1/S
虽互相给出 `q_top=.90378--.92260`、normalized weight 约 `.03888`，五族仍全部 Rhat/ESS fail；低能族
max Rhat `1.1335--1.3048`、min ESS `66.86--87.61`。U 更明确地停在 normalized state/B weight
`.097775/.101909`，而低能族约 `.03888/.0400`；U `q_top=.0000405`、max Rhat=`inf`、min ESS=`39.75`，
所有 U/低能 pair 的八个分布门全失败。U 每条 measurement 仍至少有 580 个 B-column 和 2406 个 label
changes，因此“链在动”不等于向目标输运。MAP bridge 双向全过也不能替代全局混合。不得补钟、合并 raw、
删 U/MAP/S 或全部从 P/零态开始；物理零态仍非法，shifted zero 就是 P。primary constant-character helper
还暴露 `uint8` 下溢 warning，但本 run 没有 globally constant B character，corrected audit 同为零 freeze
failures，终态不受影响；后继须用 signed arithmetic 并加回归。详见
`RANDOM_FULL_COLUMN_DIRECT_BLOCK_T1_V2_CONTRACT.md` 与 `validation/056_*/README.md`。

validation 057 的 collapsed physical-p PT 后继也已在本地终止为
**`LOCAL_T1_PAIR_UNRESOLVED`**，没有启动 nd-2/nd-3。CPPT 的 cold target、physical-p collapsed density、
swap ratio、`p=.5` endpoint 和 `k=64` reference/Numba identity 均通过 exact oracle；共享 CPPT32 m8
log-mass artifact 为 4 GiB、构造约 `10.38s`，所以失败不是基础设施或单表 runtime。冻结的一条 P 和一条
exact-K0 U 完整 T1 仍给 plug-in q_top `.900885/.144627`、logical/B D2 `.346827/.093028`、likelihood
per-factor 差 `2.50668`，两条均零 round trip，最小 swap rate `.00547/.03945`。raw-only audit 不调用
sampler/runner/analyzer，状态 `INDEPENDENT_RAW_ONLY_AUDIT_PASS_LOCAL_T1_PAIR_UNRESOLVED`，SHA
`1dd1260d...bf0`。`T1_PAIR_SUCCESS` 只表示 wrapper 正常写完 terminal report，不是算法 PASS。不得把该 raw
补长、复制、接 HP64 warm start、统一为 P/零初态或部署远端；CPPT32 失败不是 CPPT64/replica exchange
不可能，但 CPPT 与 HP64 同属 collapsed-B tempering，即使新后继通过也不能充当机制独立确认。后续优先做
直接攻击 B 慢变量的正交 hard-coset kernel 或严格 oracle，而不是继续优化 ladder 密度/round-trip 表象。

2026-07-24 的 validation 058 exact full-B-row elimination 已本地终止为
**`LOCAL CONDITIONAL FEASIBLE / STANDALONE LOW-ENERGY TRANSPORT NOT VIABLE`**，未启动服务器。m8 行条件的
deterministic min-fill width=`12`、最大 factor=`8192`；n=10/13 完整枚举、row detailed balance、full-sweep
stationarity 和 PortablePrng/cache replay 共 20 项通过。128 MiB mass 构造 `.316s`，row update
`.01291s`、17 MiB incremental peak、factor-two T1 投影 `264.39s`，所以 runtime 不是问题。关键反例是
P/M0/S0 的 median entropy=0、median expected row change 仅 `1.2e-21--1.9e-21`，完整 sweep 全部 0 move；
exact-K0 U 则 median expected change=`11.645`、首 sweep 改 24 rows/294 bits。独立 target-only elimination
复核到 `7.8e-13`，并给出 P/M0/S0 在 10240 cyclic updates 内至少一次 row move 的 union bound 均
`<9.9e-6`（primary/audit SHA=`0f99bba4...172da/3845759b...bd1`）。因此不得浪费资源跑 standalone T1 或
把“U 快速下降”当作全局混合；该 row block 只能在 fresh contract 下作为混合 kernel 的 U-collapse 组件，
另一个 move 必须负责低能 B-basin transport。它仍与 HP/direct-column 共享 collapsed-B 错误模式，不能充当
机制独立确认；P/U/MAP/S 与 B-character 门必须保留，不能统一到 P/零态。最终 exp101+exp102 回归为
`1020 passed, 4 existing warnings`。

2026-07-24 validation 059 把 058 full-row 与 056 direct-positive full-column 按每 clock 各一次组成严格 exact
hybrid，并以 source `1e9097644dbed0ccb6cd61da1dc80d57413ce4bb` 完成本地 P/U/M0/S0 各 4 条
`256+1024` pilot；16/16 raw 与 full replay 全过，但终态为
**`LOCAL_HYBRID_B_NECESSARY_GATES_FAIL`**。低能三族 late B weight/likelihood 约 `.039--.042/-5.1`，U 仍为
`.10823/-11.2326`；U/P B-weight、likelihood、B-bit-MSD 差为 `.06901/6.0030/.04992`，四条 U burn endpoint
零条通过宽松 `.065/-6.5` gate，U 自身 likelihood 前后段仍漂移 `.5695`。U burn 虽有 21--25 个 row changes，
measurement 只剩 1--3，说明 row block 约扫一轮后就在错误高能 basin 冻住，column block 未修复。raw-only
audit 不调用 sampler/kernel/analyzer，逐 clock 重建 B/cache/state/label/weight/likelihood/counter/seed/gate，状态
`INDEPENDENT_RAW_AUDIT_PASS_LOCAL_HYBRID_B_NECESSARY_GATES_FAIL`（primary/raw-set/audit SHA=
`2f25aa7c...873ba/db6a303e...cd88/443d461d...b7c`）。不得把它部署 nd-2/nd-3、补长、与 056 合并或报 q_top；
下一候选必须协调多行/多列或以别的机制跨过 collapsed-B basin barrier，而不是继续排列两个单 block exact
kernel。

2026-07-28 的 validation 062 character-gate 校准在五个冻结 operating points 上全部失败，终态为
`CHARACTER_GATE_REDESIGN_REQUIRED`。更重要的长期规则是：`max_u |delta m_u|<=.04` 对完整冻结 catalog
最多只给 `.08` 的 mean-square/purity 差界；只有 catalog 包含全部非零 logical characters 时，该量才直接是
`q_top`，有限 sampled large-k characters 对未观测 tail 没有任何覆盖。后续 gate 应以直接 `q_top`
equivalence 为 primary，并独立保留 full-label `D2`/分布门；character maximum 只作慢模态诊断，不能冒充
交付量或 tail certificate，也不能因最大校准点的 Wilson lower `.9779026<.98` 就只追加 trials。

validation 063/065 的 Nishimori 辅助校准仍为 `NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT`；065 又发现
三个 hard coset 的 logical-sector weight enumerator 存在数学 MAP ties，浮点 `argmax` 的不同合法选项造成
11 个 payload 字段不一致。凡 MAP-derived control 必须在浮点 posterior 比较前预冻结基于精确数学对象的
canonical tie 语义，并让 runner/oracle 审计同一语义；`terminal_gate_invariant=true` 只能说明最终失败判定
未变，不能写成 `full_payload_match` 或 audit PASS，更不能升级为 `q_top` bound。

validation 064 的 timing 只覆盖少量单 code/单 disorder，strict full-grid totals 全为 `null`；所有 scenario
arithmetic 都是 planning proxy，不是 confidence bound、campaign total 或远端启动权限。validation 060 的
唯一结构 survivor `MR2` induced width=`25`、单表下界 `512 MiB`，只是在 HP64 后续 Stage 3/4 真正失败后
才可考虑的一次同族 contingency，不能充当 large-k orthogonal confirmer。060--065 均未运行远端
measurement；当前停止是 estimator/审计/资源覆盖等科学门禁失败，不是在等待服务器。

---

## 附录 C：旧 笔记/实验报告.md 中的 exp102 原文

- 2026-07-24：完成 exp102 validation 059 exact row+column hybrid 本地必要条件 pilot，终态 **`LOCAL_HYBRID_B_NECESSARY_GATES_FAIL`**，没有部署服务器。source=`1e9097644dbed0ccb6cd61da1dc80d57413ce4bb`；每 macroclock 固定一次 uniform direct-positive full-B-column heatbath + 一次 uniform exact full-B-row elimination，再 exact draw A|B。小 HGP 完整 transition matrix 证明 ordered clock 保持 strict collapsed target，P/U/M0/S0 各 4 条 `256+1024` 共 16/16 raw 与 full replay 完成。低能 P/M0/S0 late normalized B weight=`.03922/.04065/.04159`、likelihood/factor=`-5.2297/-5.1555/-5.0977`；U 仍为 `.10823/-11.2326`，U/P 的 B-weight/likelihood/B-bit-MSD 差=`.06901/6.0030/.04992`，四条 U burn endpoint 零条过预注册 `.065/-6.5` collapse gate，U first/last likelihood drift=`.5695`。U burn 有 21--25 次 row changes、measurement 只剩 1--3，表明 row block 约早期扫一轮后在错误高能 basin 冻结，column block 未完成设想的低能输运。primary report/raw-set SHA=`2f25aa7c...873ba/db6a303e...cd88`；独立 raw-only audit 不调用 sampler/kernel/analyzer，逐 clock 重建全部 B transition、cache、hard-coset state、label、weight、likelihood、counter、seed、summary 和 gate，状态=`INDEPENDENT_RAW_AUDIT_PASS_LOCAL_HYBRID_B_NECESSARY_GATES_FAIL`、SHA=`443d461ddc5ab24e48394114902885831e2db38c88aff95dba6bf9081253bb7c`。runtime/replay 通过，故是 sampler transport failure，不是基础设施、物理点失败或不可能；不跑 nd-2/nd-3，不补长/合并/报 q_top。下一候选必须协调多行/多列或另用能跨 collapsed-B basin 的机制。
- 2026-07-24：完成 exp102 validation 058 exact full-B-row elimination 本地可行性与独立 target audit，终态 **`LOCAL CONDITIONAL FEASIBLE / STANDALONE LOW-ENERGY TRANSPORT NOT VIABLE`**，未使用服务器。目标为严格 collapsed `P(B[i,:]=v|B[-i,:],Y)`，m8 deterministic min-fill width=`12`、最大 factor=`8192`；n=10/13 zero/nonzero syndrome 的完整 conditional/normalizer、single-row detailed balance、full-sweep stationarity 与 PortablePrng/cache replay 共 `20 passed`，相关 collapsed/full-column 回归 `145 passed`，最终 exp101+exp102=`1020 passed, 4 existing warnings`。m8 128 MiB mass 构造 `.316s`，row update `.01291s`、incremental peak≈17 MiB、factor-two T1 投影=`264.39s<7200s`，primary feasibility=`LOCAL_FULL_ROW_CONDITIONAL_FEASIBLE`、SHA=`0f99bba44561bd48b0b1f9a3acf96eb475500e5f72f3ab0337c851e71b3172da`；这只说明算得对且算得动。真正阻断来自冻结 P/U/M0/S0：P/M0/S0 的 median entropy=0、median expected row change=`1.2e-21--1.9e-21`、minimum self probability≥`.9999999926`，首个完整 sweep 全为 0 move；U 则 median entropy=`2.619`、expected change=`11.645`，首 sweep 改 24 rows/294 bits。独立、不 import row sampler/statistics 的第二套消元把 expected change 复核到 `7.8e-13`，并将 P/M0/S0 在 10240 cyclic updates 内至少一次 row move 的 union bound 均压到 `<9.9e-6`，audit SHA=`3845759b77f53f642356439604527730b16373987d7846217a8d02815b40abd1`。因此不跑必然冻结的 standalone T1/remote；该 block 只可在 fresh mixed-kernel contract 下作为 U-collapse 组件，另一 exact move 必须承担 low-energy B-basin transport。它仍与 HP/direct-column 共享 collapsed-B 共同失效，不能作独立确认，也不能删 P/U/MAP/S 或统一为 P/零态。
- 2026-07-24：完成 exp102 validation 057 collapsed physical-p PT 的 exact oracle、资源审计和冻结 m8 T1 P/U 必要条件探针，终态 **`LOCAL_T1_PAIR_UNRESOLVED`**，未使用服务器。CPPT cold target、physical-p collapsed density、local/swap stationarity、`p=.5` uniform endpoint 以及 `k=64/bit63` reference/Numba identity 均过；新增 16 项测试，全量 exp102+exp101=`1000 passed, 4 warnings`。m8 CPPT32 共享只读 log-mass artifact 为 4 GiB、构造 `10.38s`，短 smoke 外推 T1=`474.99s<7200s`，说明资源可行但不代表混合。随后 source=`a90d3f01641f4ce1432f739d7a76cf6f9128885a` 以 fresh seed 跑一条 P 和一条 exact-K0 U 的 `2048+8192`：P/U plug-in q_top=`.900885/.144627`、normalized weight=`.038890/.061817`、B weight=`.040009/.059124`、likelihood/factor=`-5.19372/-7.70041`，logical/B D2=`.346827/.093028`；两条均 0 round trip，最小 swap rate=`.00547/.03945`、cold-origin fraction=`.4375`，transport 与宽松分布必要门同时失败。primary report SHA=`287d62b5...e1c1`，P/U raw SHA=`e771084a...6a27/dada68f8...7f`；不调用 CPPT sampler/runner/analyzer 的独立 raw-only audit 重算 support、B/state/label/weight、likelihood、characters、counters 与 terminal，状态=`INDEPENDENT_RAW_ONLY_AUDIT_PASS_LOCAL_T1_PAIR_UNRESOLVED`、SHA=`1dd1260d80469ac12f1061a289af540041cffb3a0d3857073299a2effbae0bf0`。`T1_PAIR_SUCCESS` 仅是 wrapper 完成 marker，不是 sampler PASS。按预注册规则不复制、不补长、不做 CPPT64 rescue、不部署 nd-2/nd-3；这不是物理 q_top、参数点失败或 CPPT64 不可能。CPPT 与 HP64 同属 collapsed-B tempering，后续优先开发直接攻击 B 慢变量的正交 hard-coset kernel/严格 oracle。
- 2026-07-24：exp102 validation 056 fresh runtime-corrected direct-block m8 T1 已完成并终止为 **`UNRESOLVED_DIRECT_BLOCK_T1_M8`**。immutable run=`exp102_q0_direct_block_t1_m8_v2_20260724_6933e31`、source=`6933e319b27840976f34e27c0d11313b6973cbe3`；完整 054 portable preflight 与 fresh 两长度 runtime preflight 均三节点 exact consensus PASS，最坏 factor-two T1 投影=`6550.3213s<7200s`，随后固定 `14/13/13` ownership 完成 40/40 raw、零复用。primary report/raw-set SHA=`e1bfb340...6015/a267ded6...2259`；不调用 sampler/replay/analyzer 的 `allow_pickle=False` auditor 独立重算初态、hard-coset、B transcript、states/labels/weights/likelihood、q_top/D2、Rhat/ESS、全部 family/pair/terminal gates，状态=`INDEPENDENT_RAW_ONLY_AUDIT_PASS`、SHA=`ada30d3cca844ede66b29e204f73eb1fe6fe2a297992ff0c28027878aa04b08e`。五族均 Rhat/ESS fail：P/M0/M1/S 的 q_top 虽集中在 `.90378--.92260`、weight 约 `.03888`，max Rhat=`1.1335--1.3048`、min ESS=`66.86--87.61`；U burn 后仍在 state/B weight=`.097775/.101909`，而低能族约 `.03888/.0400`，U q_top=`.0000405`、Rhat=`inf`、ESS=`39.75`，所有 U/低能 pair 八个分布门全失败。P/U `delta q_top=.90374`、logical/B D2 upper=`.93903/.20827`、466 个 B-character mean fail。U 每条仍至少有 580 次 B-column 和 2406 次 label change，M0/M1 双向 basin bridge 也全过，说明“链在动/已跨一个已知 basin”仍不等于全局混合。primary constant-character 路径另有 `uint8` 下溢 warning，但本 run 无 globally constant B character，corrected audit 同为零 freeze failure，终态不受影响。不得补钟、合并 raw、删 U/MAP/S 或全部从 P/零态开始；这不是物理 q_top、`IMPOSSIBLE` 或 m6/HARD2/formal 权限。后续 fresh exp102 服务器实验按用户新要求只用 nd-2/nd-3。
- 2026-07-24：exp102 validation 055 fresh direct-block m8 T1 已终止为 **preflight `RUNTIME_EXHAUSTED`，measurement raw=0**，没有测试 sampler 收敛或产生 q_top。前两个 schedule attempt 都在 control 前因 fresh run root 被提前创建而基础设施失败；修正 marker 位置后的权威 run=`exp102_q0_direct_block_t1_m8_20260724_146ef55_r3`、source=`146ef550591a72435641c47baa8794c338f7a27e`、archive=`b9602502...e3c48`、schedule SHA=`bbc2e268...ee731a`。最终 source 的完整 validation-054 preflight 在 nd-1/2/3 exact consensus PASS，replay-inclusive T1 投影=`4216.16/4149.15/4549.57s`；但 055 自己冻结的 probe 仅跑 `2+8` updates，随后把包含固定初始化/runner 开销的总时间线性外推到 10240 updates 再乘 2，产生 `9272.13/8779.07/13638.99s > 7200s` 的 false-negative resource estimate，workflow 因而正确阻止 40 条 measurement。conda-12 独立 audit 复核 source/control/schedule、两次 schedule failure、所有 self-hash、exact consensus、投影算术与 raw absence，状态=`INDEPENDENT_AUDIT_PASS_PORTABLE_PASS_T1_RUNTIME_EXHAUSTED_CONFIRMED`、SHA=`00622194dc370a66e08a0b94a7108b324aa49322de648fda7656f2c6ed5fc665`。不能事后把 055 改成 PASS，也不能写成 sampler/参数点失败；fresh 后继必须保留 T1、7200s cap、P/U/M0/M1/S、full replay 和全部统计门，只能修正为代表 steady-state 的 runtime probe 或预冻结 intercept/slope 估计，并用新 contract/source/schedule/seeds/raw。
- 2026-07-24：完成 exp102 validation 054 direct-positive fixed-block full-B-column 三节点预检，终态 **`PASS`**，只授权 fresh m8 T1 diagnostic。方法 `RFCG-C24-DPB12-S1` 保持 exact random-scan full-column heatbath，但用固定 `2^12` candidates/block 的正权重 subtotal 避免完整 CDF。最终 source=`61d605a5e27db0970457736c72d1c45d72a12b10`，run=`exp102_q0_direct_block_preflight_20260724_61d605a`，archive=`61bb87e7...a7bec`，aggregate SHA=`27f6d276...10612bc`。完整 `2^24` weight check 最坏 scaled absolute/relative/TV=`2.020606e-14/7.290711e-14/4.148991e-15`，log lower bound=`-221.658`；macmini/nd-1/2/3 精确复现全部 12 个 block-subtotal SHA 和 4 条 PortablePrng sampling+replay transcript。三节点 replay-inclusive T1 投影=`4144.85/4139.52/5454.14s < 7200s`。conda-12 独立 audit 不调用 combiner，复核 canonical/self-hash/source/config/reference/numeric/runtime/stage/log/consensus，给出 **`INDEPENDENT_AUDIT_PASS_DIRECT_BLOCK_PREFLIGHT_CONFIRMED`**（SHA=`9646c6f92070024680728bf377e802e647b48a2b66ca6210c89c436fbd70f539`）。这不是 mixing 或 q_top 结果；下一步仍须 fresh contract/seeds/raw 跑 m8 `P/U/M0/M1/S x8` 的 `2048+8192` fixed clocks 与 full replay，所有 052 的 full/B convergence gates 原样保留。非零 syndrome 的物理零态非法，shifted zero 已是 P，不能全从 P/零态开始掩盖慢混合。
- 2026-07-24：完成 exp102 validation 053 memory-streaming full-B-column 三节点预检，终态为 **`CONFLICT` 且独立 runtime-exhausted**，没有 T1/measurement raw。fresh run=`exp102_q0_streaming_preflight_20260724_de68bbc`、source=`de68bbc06aa729063b24c1f40ba23cc404a44c9c`、archive=`e8f14f85...6586`。macmini 的 12 个完整 `2^24` legacy/streaming CDF 全部 byte-identical，speedup=`4.9391x`、最坏 T1 replay-inclusive 投影=`2432.39s`；但 Linux 三节点都且仅在 `U0,column=11` 出现 legacy dense 与 streaming 的 byte mismatch，按预注册 any-mismatch 规则必须 fail closed。该 proposed streaming CDF 的完整 SHA catalog 以及 P/M0/S0/U0 四条 PortablePrng sampling+replay transcript 在 macmini/nd-1/2/3 实际全部一致，因此问题不是三台 streaming sampler 互相漂移；仍不能事后把冻结门解释掉。独立 runtime 也失败：nd-1/2/3 speedup=`2.5911/2.5372/1.3823x`，投影=`8797.83/9144.89/17760.30s`，未过 `4.2x/7200s`。conda-12 audit 复核 canonical JSON、self-hash、source/config、stage/log、唯一 mismatch、runtime 算术、CDF/transcript catalog，给出 **`INDEPENDENT_AUDIT_PASS_CONFLICT_AND_RUNTIME_EXHAUSTION_CONFIRMED`**（SHA=`6426a1a01c01747f474d587a10cdb6db9e53db09112193499a8f9307adb7640f`）。仍无 m8 T1、m6/HARD2、formal/held-out/production 权限；下一实现只可用 fresh contract/source，且不得删 P/U/MAP/S、缩钟、去 replay 或降低收敛门。

- 2026-07-24：完成 exp102 q=0 validations 047--051，并完成 052 fresh random-full-column T1 m8 的服务器三节点 preflight；终态为 **`RUNTIME_EXHAUSTED`**，measurement raw 数为 0。047 的 truth-free dressed logical-XOR catalog 代数/rank=64 虽通过，但 BASE/P 在 T3 下可达 signature rank 仅 `4/1`，所有低能 L 都有 downhill 路回同一 base label，full-rank 乐观瓶颈仅 `1.700868e-10` expected accepts/direction，终态 **`LOCAL_CENTER_PRESERVING_STRUCTURE_NOT_VIABLE`**，避免把“全链掉进同一 basin”错当收敛。048 因 reference mass path 在一条 P raw 后基础设施终止，raw 禁用；049 fresh Numba 版 12/12 raw/replay 完成但终态 **`LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_UNRESOLVED`**：P/L B-column 几乎不动，A|B 精确重抽却能制造 visible logical-label changes，U 的 B-weight/likelihood 仍远离，说明 state/label changes 不是全局混合证据。050 的两个 truth-free weight-62 MAP anchors 同 logical label、B 距离 6，正确两列顺序首步 conditional≈`.03846`、T1 预计首 departure≈`16.4`；短跑只预计约 `.5`，故只足以新开 T1。051 不调用旧 runner 的独立审计逐项复算 catalog、raw states/B/labels/weights/likelihood/counters/report/bridge，给出 **`INDEPENDENT_AUDIT_PASS_FAILED_RESULTS_PRESERVED`**（SHA=`c018e4af9b4aa5a78ae8a4c192e64c7a0beb8d53ca21e27c4c27176002a18767`），同时指出 049 historical source identity 漏列 transitive modules，只认证 raw 失败结论、不升级来源证明。

  052 在看结果前冻结 `P/U/M0/M1/S` 各 8 条、`2048+8192` clocks。物理零态在 syndrome weight=160 下非法，移位零就是 P；U 为独立 exact-K0，M0/M1 检验已知 B bridge，S 为 8 个 truth-free 低能、logical label 和 B block 均不同的起点，其中一条故意与 M0 同 B 但 logical 不同，用来区分 A/logical 重抽和真 B transport。门禁以 character U-statistic q_top/D2、全/B weight、B likelihood、全部 B bit/row/column+dense characters、logical characters、Rhat/ESS、恒定 character burn crossing 和双向 MAP basin visits 为主，不以 acceptance/state change 代替。immutable run=`exp102_q0_rfcg_t1_m8_20260724_6fa489f`、source=`6fa489f838dffea15b07e1ef3b3fbee3951dd3c0`：nd-1/2/3 的 mass table 和四条 portable transcript 逐位一致，`exact_consensus=true`；但 replay-inclusive factor-two 投影为 `24701.47/24812.06/29871.42s`，均高于 7200s 上限，故 workflow 正确阻止 40 条 measurement。独立 audit 给出 **`INDEPENDENT_PREFLIGHT_AUDIT_PASS_RUNTIME_EXHAUSTED_CONFIRMED`**（audit SHA=`817425dbaa6a9e5d90d03d34efe16f957beb7424eddd27dcde7cf12d60d75c6d`，evidence-package SHA=`1d4ec020e65a654aba21ecbe910f424b41401f4f30821cc2604310a022de0506`）。这是冻结实现/资源合同的 runtime failure，不是 sampler 收敛失败、物理点失败或数学不可能；不得事后缩短链、改并发/上限或绕过 gate。没有 m6/HARD2/formal/held-out/production 权限；任何性能后继必须另立 fresh contract/source/seeds/raw。

- 2026-07-24：完成用户授权的 exp102 q=0 BP-systematic independence-MH 本地 hard-sentinel 测试，没有启动服务器。045 v0 在写首个 raw 的 receipt record 时因 relative/absolute output path 混用终止为 **`INFRASTRUCTURE_FAILED_RELATIVE_OUTPUT_PATH`**，无 receipt/report，唯一 raw 永久禁用；046 v1 只修路径、换 fresh contract/config/seed namespace，和 v0 的 initialization/sampler seeds 零交集，科学参数不变。raw 前红队补上 full 64-bit label collision `D2_norm` 门，专门防止“相同 purity、相同 64 basis means、但 sector support 不相交”的假一致；complete small-HGP stationarity/detailed balance、path/source binding 等 18 项测试通过。v1 的 24/24 P/U/8 个 distinct-L raw 与 runner replay 全过，独立 `allow_pickle=False` auditor 不调用 sampler/runner，重新构造 hard coset、proposal density、55296 次 MH decision、state/label/weight/D2 后给出 **`INDEPENDENT_RAW_AUDIT_PASS_UNRESOLVED_CONFIRMED`**（report self-hash=`62a96e7f16cbbc020f8d4e893c413bd11ec54da928893ccf23abbf6c65983c58`，raw-set=`60ae69f3b829fd6037cf25979f0a55f3e74b52bc086fb988533f963ee70bc28c`，audit self-hash=`d7af8f008c500b72df512a546a051b53e1c049de5fc29a92b428cb9a35fd2ce0`）。终态 **`LOCAL_BP_IMH_TRANSPORT_UNRESOLVED`**：P/L 每条 burn 和 measurement 都零真实移动；U 只用 1--3 次 burn move 冷却并全部落到同一 weight-62/P-label state，measurement 仅 0--2 次真实移动。P 的最大 measurement log acceptance 至多 `-53.13`，L 最好 `-47.79`，说明 proposal 对 high-`pi/q` states 严重欠覆盖；U 的大量 accepted counter 只是 self-proposal。P/L `delta q_top=D2=1`，U/L 约 `.998413`。不能用 BP 先把 U 全冷却到 P label 再接旧 full-row 使三族表面一致：旧 kernel 也把 P/L 送到同一 B basin，这可能共同漏掉其它 B-sector 质量。该配置不进 HARD2/remote/formal/held-out/production；下一方法必须先解释 high-`pi/q` signature/basin coverage 和独立 B/tail 证据，不能只调 acceptance、统一初态或延长链。

- 2026-07-24：完成用户授权的 exp102 q=0 Houdayer pair-kernel 本地诊断链 `035--042`，没有启动服务器任务，也没有生成 posterior/q_top/正式实验结果。先做 exact-normalizer 结构排查：035 的 one-copy character-WMC 两种编码均有 min-degree induced width=`378`，min-fill 在 120s 前已超过 width=`102`；036 的七种 deterministic trellis order 最好 exponent=`584`，均远超冻结 actionability cap（width/exp=`24`），只排除这些表示法而非 q=0 或 exact WMC。随后 tensor-logical HCA（037/038）在 200 个冻结真实低能 P/L、L/L 对上全部只是整对 replica exchange；唯一预注册的 canonical-reduced 坐标（039）则有 `102/120` 个 L/L 对发生两 component 的真实重组，例如合法 `67+67 -> 63+71`，因此值得做一个严格 pair sampler，而不是把结构图直接当作混合。

  新鲜 `HCA-RHB1` 以每 replica 832 次 random-scan coordinate heatbath + 1 次 complete-component HCA 为一个时钟；小 HGP 全 transition-matrix stationarity、reference 逻辑、replay 与 raw-only audit 都通过，042 outcome-blind runtime 预估固定 `128+1024` clocks 仅约 22.0 秒/对（2x safety）。041 用 PP、两个独立 exact-K0 uniform hard-coset 的 UU、两个固定 low-energy distinct-label L 的 LL 和 PL control 各 8 对，终态为 **`LOCAL_HOUDAYER_PAIR_TRANSPORT_UNRESOLVED`**：PP/LL/PL 的 normalized pair weight 都约 `.03886`，LL 有 `1091` 个真正的新 unordered-pair HCA event；但 UU 仍为 `.1486354`，相对 PP 差 `.1097799`、basis-character 最大差 `1.4603271`，早/后段也不稳定，且每一条 UU 的 1024 个 measurement HCA 均只是 whole-pair exchange、真新 pair event=0。结论不是“初态太乱”，而是该 HCA 的真实低能 L/L 重组不能把精确均匀 hard-coset 质量冷却并运输到低能目标区。不能删除 UU、把链统一从 P/物理零开始、补长或合并 raw；非零 syndrome 下物理零非法，平移零恰好就是 P。041 只排除这个固定 kernel/budget，不代表 HCA、后验或 q=0 数学不可能，仍无 `READY_FOR_FORMAL`、held-out 或 production。

- 2026-07-24：完成 HCA 后继的 collapsed-B 结构性反证 `043_q0_collapsed_houdayer_structure_feasibility_20260724`，终态为 **`COLLAPSED_B_HCA_NO_LOW_ENERGY_RECOMBINATION`**；它没有运行 MCMC、没有产生 posterior/q_top，也没有启动服务器。该 probe 对精确边缘 `pi_B(B)∝(.04/.96)^|B|∏_j M_p(Y[:,j] xor B H[:,j])` 检验完整 factor-component Houdayer swap；small-HGP 穷举已验证 B conversion、factor-pair invariant、involution、row sum、detailed balance 与 stationarity。真实 m8 hard cell 上，16 个 P/low-energy-L、120 个 L/L 和 64 个 P/rank-complete-L 的 B masks 两侧全部相同，虽物理 logical labels 不同，说明冻结低能 logical variation 全在 A；独立 exact-K0 U/U 有 284 个 B-bit disagreement，却只形成一个 component，唯一动作仍为 whole-pair exchange、不会创造新 unordered B pair。故不能因为 move 精确、physical state 改变或 A 有重抽样，就把 HP64+B-HCA 当作解决 B/logical 慢变量的办法；不得实现该 hybrid、优化 acceptance/state changes 或把它算作独立确认。该结果只排除 direct collapsed-B HCA，不代表 HP64、HCA、q=0 或后验数学不可能。

- 2026-07-24：完成 exp102 q=0 BP-mixture dominance-witness 本地结构预检 `044_q0_bp_dominance_witness_feasibility_20260724`，终态为 **`BP_MIXTURE_REJECTION_ENVELOPE_WITNESS_INCONCLUSIVE`**；没有 MCMC、IID draw、posterior/q_top 或服务器任务。它直接问能否把 frozen BP-SYS-F64/R64 的精确三 component proposal density 变成严格 rejection-envelope：对 planted、64 个 canonical rank-complete reduced-logical 和两套 systematic basis 的全部 planted-plus-one-coordinate，共 1691 个预注册合法 hard-coset witnesses，以向上 Decimal mixture density 和 `Z=Pr_.04(H_Ze=y)/.96^1600<=.96^-1600` 算 `pi/q` 的保守下界。最大的下界仍仅 forward `5.53e-63`、reverse `2.54e-53`，远小于冻结 `1e6` cap；这**绝不是** BP proposal 已通过，而是 `Pr(y)<=1` 给出的 global normalizer upper bound 太松，因而无法从“没有坏 witness”推出 tail coverage。结论是不能据此开启 BP-only IID/rejection sampler、报告 q_top 或用它替代 P/U 对抗初态；要让这条路有意义，仍须先得到独立而紧的 hard-coset normalizer/tail certificate，正是当前尚未解决的全局质量问题。config SHA=`be78411d1459a6a33f835fc0780f70bd41cd4d0c2f45e9bb659dceb4f3faf180`，self-hash report=`d36815dce4662c922791409258cf1dbb43492f54465453cce116182e9862e20b`。

- 2026-07-24：完成用户授权的 exp102 q=0 strict depth-two collapsed-B tail-envelope 本地可行性测试（`m08_c06,p=.04,d00,attempt022`），终态为 **`DEPTH2_ENVELOPE_NOT_TIGHT_ENOUGH`**，没有启动服务器任务，也没有生成 posterior/q_top/正式实验结果。该测试不是 MCMC：用固定、非 planted 的两条 MAP-derived B marginal 作 retained lower mass，对全部 collapsed-B 配置做 depth=2、width=25 的 directed-rounding partition upper envelope；canonical config SHA=`2d1c27b769e5011139265cecac9d8c794f694c31acc03d92a90829e706933bb0`，自哈希报告 SHA=`dffacc4ac340c33b49e8578432ce17a3f8b89a65698d08985677662f3d23f147`。资源门本身通过：最大单表 512 MiB、peak RSS=`2504933376` bytes、envelope=`3.141538s`（上限 6 GiB/900s）；但这恰好暴露真正问题不在速度。全 partition upper=`3.110162637896501e-11`，两条 retained lower 合计仅 `3.3080527487949514e-96`，故严格 tail/retained 上界=`9.401792758683984e84`，相对目标 `.01` 仍差约 87 个十进制数量级。不能用更长 MCMC、不同初态、局部 ESS 或快运行来掩盖这个全局总质量缺口；该负结果只排除这个 depth-two factorized envelope，不表示 q=0、后验或所有 certified normalizer 方法不可能。任何更深/不同的 envelope 都须新建并审查 tightness/资源契约，当前仍没有 `READY_FOR_FORMAL`、held-out 或 production。

- 2026-07-24：完成用户授权的 exp102 q=0 BP-systematic IID-MIS 本地 hard-sentinel 测试（`m08_c06,p=.04,d00,attempt022`），终态为 **`LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_UNRESOLVED`**，没有启动服务器任务。冻结 schedule 用 BP-SYS-F64、BP-SYS-R64 和 rebuilt MAM-IMH8 各 `16*1024` 个 direct hard-coset draws，共 `49152`；没有 MCMC 初态、P/U/L、resampling、clone 或结果驱动延长。runner 的 deterministic generation/algebra replay 和独立 `allow_pickle=False` raw-only analyzer 均通过（raw SHA=`fd662ae5a30ce0e0aa70ebf6253882da91c7cf479db9669400affe972a1625da`，report SHA=`2a62ddf1d7bfc49b06e2a80e4d6d45f2d7558970bed4f7de28faedd0f25705fb`）。两条 BP source 单独都过冻结 ESS/最大权重 gate（F64 `730.10/.00750`，R64 `457.00/.00219`），且 q_top-derived diagnostic 差 `.0054695`、D2 gate 都过；但预注册的**三源等权 mixture**仍因 MAM-source blocks 失败（min ESS=`23.48<50`，max weight=`.14695>.10`）。所以这次只否定已冻结的三源 estimator，不能事后去掉 MAM 说 BP-only 已成功；若要测 BP-only，必须另立 fresh contract/seed/raw。更重要的是两个 BP proposal 都可能共同漏掉 remote tail，约 `.993--1` 的 collision diagnostic 不是 posterior/q_top，更不解锁远端、formal、held-out 或 production。下一步应该优先发展独立 tail/normalizer certificate，而不是为让这条曲线更好看而改初态、加样或调 BP。

- 2026-07-24：完成用户授权的 exp102 q=0 fresh local IID-MIS 诊断（`m08_c06,p=.04,d00,attempt022`），终态为 **`LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_UNRESOLVED`**，没有启动服务器任务。该实验从 MAM-IMH8、LSI-IMH-T05、LSI-IMH-T10 三个冻结 proposal 各做 `16*1024` 个独立 hard-coset draws，共 `49152`；没有 MCMC、P/U/L 初态、resampling 或 clone，因此不能把失败归因于“初态太乱”。raw-only replay 和现源码重算均通过，raw SHA=`6cc3c19710725ef5ab714e010d636c7a0a0e7928db71e68059b1029009382071`，report SHA=`d7dd5521b7292f68c01f0202d623da34968d060d3ae3422cd6f117e837a36e0a`。但预冻结 importance-weight gate 全败：MAM / T05 / equal-mixture 的最小 block ESS 分别为 `22.09/28.91/28.78 < 50`，最大权重为 `.1522/.1629/.1574 > .10`；虽 MAM/T05 的差异与 mixture 的小 jackknife SE 看似不错，仍不能抵消尾部权重不稳定。报告里约 `.98-.993` 的 collision 数只是失败诊断，绝不能报告成 q_top 或物理结果。全 support、proposal agreement 或低温链未遍历 64 个 logical directions 都不是充分结论：真正尚未证明的是未见 hard-coset modes 的总 target mass。冻结 raw 缺内部 anchor/component provenance，不能事后补样或改门；任何后继必须另建 fresh contract/seed/raw，并用独立 tail-normalizer 证据或确认方法，当前仍无 `READY_FOR_FORMAL`、held-out 或 production。

- 2026-07-24：完成用户授权的 exp102 q=0 `UASRE32-R1-A1/UASRE64-R1-A1`（auxiliary-stabilizer replica exchange）本地对抗初态预检，终态为 **`LOCAL_AUXILIARY_STABILIZER_TRANSPORT_UNRESOLVED`**，不是 q_top、后验估计、HARD2 或服务器正式实验。冻结 hard cell 为 `m08_c06,p=.04,d00,attempt022`，两种 32/64 replica 配置均使用 P/exact-K0 U/合法低能不同 logical-label 的 L 各 8 条、固定 `256+2048` clocks；manifest SHA=`1c5b931117a35b859c33a1a1abe348d0f8e547784395812e2ccb3884b2271c29`，48 条 raw run SHA=`c262bc5f9b6320d22fb066a3d70a61783fce5f1479fee437c50f1c4d23e9261f`。冻结 runner 的 raw validator、6-worker 逐位 replay（SHA=`d99d0b27d8edb13c3b58bce4d05b15974befa281146b0ca19c71e02f5591b669`）和不导入 sampler/runner 的 `allow_pickle=False` raw-only audit（SHA=`646c0ee7f40bac604adbd5c206c7bc25164b5fcc9c291d21e4baa8af5e09becf`）均通过，crosscheck 还确认全部预注册 gate summary 一致。两个配置仍均失败：P/L 的 pairwise agreement 不能代替混合证明，U 与二者在 normalized weight、complete score、全部 128 logical characters 和多数 B masks 上不一致，且 U 的 early/late fixed-clock checks 失败；部分 P/L B-mask time block 也不稳定。U minimum weight 是 32-replica 的 `135..174`、64-replica 的 `163..179`，而已知合法 P weight=63；但这里的保守 target-support upper bound 对每条 U 都是 `1`，故不能把 U 区域说成已证明的 negligible target mass，正确结论仅是冻结预算内未收敛。所有 64 个 B-mask 都非恒定也不构成全局平衡证据；把链统一从 P/“零态”开始只会掩盖初态依赖，非零 syndrome 下物理全零非法，而平移坐标的零本来就是 P。raw 不可补长、合并、重加权、算 q_top 或送 HARD2/正式流程；只排除这两个冻结配置和预算，不代表 q=0、后验或该类算法数学上不可能。详见 `data/expander_code/exp102/validation/024_q0_aux_stabilizer_v0_20260724/RESULT.md`。

- 2026-07-24：完成用户授权的 exp102 q=0 `UARE32-R1/UARE64-R1`（uniform-anchored full-energy collapsed-B replica exchange）本地对抗初态预检，终态为 **`LOCAL_UNRESOLVED_UNIFORM_ANCHOR_TRANSPORT`**，不是 q_top、后验估计、HARD2 或服务器正式实验。冻结 hard cell 为 `m08_c06,p=.04,d00,attempt022`，P/exact-K0 U/合法低能 L 各 8 条、固定 `256+2048` clocks，manifest SHA=`9098102f1612cb70630d936fb86b949e9a19baa428c187238741d6dbd2f1b560`，48 条 raw run SHA=`322a23b72f1fb443e435f95ce64088f7a524437a3005b3e1979e7bb2ff507761`。独立 raw-only audit V2 从 `allow_pickle=False` NPZ 重建 hard-coset、P/U/L、weights、labels、B/A traces、score 和 counters 后给出同一失败；冻结 runner 的 V1 仅在 post-replay time-half summary 有字典索引 defect，不能为修报告而改动其已绑定源码，因此另以隔离 V2 validator 调用原 runner 的 raw validator 和 sampler replay，48/48 全字段逐位复现（SHA=`f2c84bb8334d7b1ac6c7c56799ca9e4296c07a24274066e2d5983df2e0d767d4`）。两条 ladder 都不是 P/L 分歧而是 U 没有回到低能 target support：P 与 L 一致，U 最低测量 weight 仍为 `247..262`，而已知合法 P 为 63；hard coset dimension=832 时，即使用全 coset 作为最松乘法上界，U 区域后验质量仍至多 `3.148385600959564e-4 < .001`，并且 U 早/后半固定时钟也不稳定。因此不能把所有链换成 P/“零态”来制造表面收敛：该非零 syndrome 下物理全零非法，平移坐标的零本来就是 P。raw 不可补长/合并/算 q_top/送 HARD2 或正式流程；仅排除这两个固定 UARE 配置和预算，不代表 q=0、后验或该类算法数学上不可能。详见 `data/expander_code/exp102/validation/023_q0_uniform_anchor_pt_v0_20260724/RESULT.md`。

- 2026-07-24：完成用户授权的 exp102 q=0 `FRG-VE1`（collapsed-HGP 整行 variable-elimination Gibbs）本地对抗初态预检，终态为 **`LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE`**，不是 q_top、后验估计或服务器实验。冻结 hard cell 为 `m08_c06,p=.04,d00,attempt022`，P/U/合法低能 L 各 8 条、固定 `64+512` sweeps，manifest SHA=`430659be5aac3b1fe099b2c15eadda194878beba663fb7b11874fc05b4bf69a7`。n=10/n=13 exact oracle、详细平衡/平稳性、reference/Numba transcript 和 m8 bit-63 边界的 focused suite 共 `42 passed`；24 条 raw 的 deterministic replay（report SHA=`53604ebf941bd514867baa83f1c47abc901803d10ae83923036d340da509550d`）及不导入 sampler/runner 的 raw-only audit（SHA=`ca9556e01e0e7bdc0a26ddb69d067c1dd209f6439c90ad43ee3156ecb13cc561`）均通过。这里特别不把“64 个逻辑方向都 leave-return”单独当作收敛定义：低温下 P/L 的低波动可能正确；但审计后 P/L 都已到同一合法 weight-63 state，U 的全部测量仍为 weight≥248。正确 q=0 目标是 `pi(e|y)∝(.04/.96)^|e|`，能量没有误接成相对 planted error；该 hard coset 维数为 832，哪怕把全 coset 都给高权重区，`Pr(|e|≥248)≤2^832*(.04/.96)^(248-63)=1.3118273337331353e-5`。所以 U 的大量 B/label 变化仍被困在可证明的低目标质量区，不能误作另一平衡态，也不能删掉 U、把非零 syndrome 下非法物理零态拿来制造表面收敛。该 frozen FRG-VE1 预算不可补长、合并、算 q_top、送 HARD2/正式/held-out/production；只排除该配置，不代表 q=0、row conditional 或后验数学上不可能。

- 2026-07-23：完成用户授权的 exp102 q=0 无重采样 `CAIS64-B8-S1-N128` 本地 collapsed-AIS 路径权重预检，终态为 **`LOCAL_COLLAPSED_AIS_PATH_WEIGHT_NOT_VIABLE`**，不是 q_top、后验估计或服务器实验。它在同一 `m08_c06,p=.04,d00,attempt022` hard cell 冻结 HP64 的 64 层二次 bridge、精确 `lambda=0` iid `B~Bernoulli(.04)` base、每层一次可逆 8-bit B-block heatbath、无 resampling/clone，以及 `4` 个 column-major + `4` 个 row-major 独立 N=128 populations；因此 P/U/L 或物理全零态不能替代该已知 base。manifest SHA=`c3dc27a3e0d7a233ac66027c61f7e642e2cb343b5b01bc8120dd3e0211965ba6`；8 个 raw 通过完整 seed replay（SHA=`5e6ae5e47ca67e17692f12051fd71a65a400664e9633ead4d20f15558e662ac7`），并由不导入 AIS engine、也不调用 sampler 的审计独立重建 HGP syndrome、iid 初态、coset mass、B/A 代数和全部 AIS 权重后通过（audit SHA=`c211911b2ceaaf6e2b033950b8eef32d6ac4c9623e68ae2b5f4cdd6ce5317321`）。但所有 population 的终态 full-path ESS/N 仅 `.0078125--.0100431`（门槛 `.25`），最大权重 `.872760--1`（门槛 `.10`），单级最大权重 `.122396--.214436`（门槛 `.10`）；median cumulative ESS 从 stage 15 的 `85.93/128` 降至 stage 31 的 `1.22/128`、冷端 `1.000002/128`。这说明取消 SMC resampling 消除了家谱共祖，却没有消除后半 bridge 的完整路径权重集中；不能只看单级 CESS、B 更新或“没有 clone”就宣称可用。该 raw 不可补长、合并、事后重加权、计算 q_top 或送 HARD2；只排除该冻结 AIS 配置，不是 collapsed posterior、所有 AIS 或 q=0 数学不可能。

- 2026-07-23：完成用户授权的 exp102 q=0 `CSMC64-B8-S1-N128` 本地 collapsed-SMC 权重/家谱预检，终态为 **`LOCAL_COLLAPSED_SMC_WEIGHT_OR_GENEALOGY_NOT_VIABLE`**，不是 q_top、后验估计或服务器实验。该测试先用 n=10/n=13 HGP 穷举验证 collapsed B 边缘，reference/Numba transcript 一致；然后在 `m08_c06,p=.04,d00,attempt022` 冻结 64 层 HP64 二次 lambda bridge、`B~Bernoulli(.04)` 的精确 lambda=0 base、每层无条件 systematic resampling、一次 8-bit exact B-block heatbath，以及 `4` 个 column-major + `4` 个 row-major 独立 N=128 populations。manifest SHA=`ee3496f1d08e3e78db306f91b921a96d402c80a225b8c7e214978590e615f979`；8 个 raw 完整 seed replay SHA=`4f59ea1766432dece1b4d5bac263d906ba426cdef42c196e6fac0b016650b0f8`，不调用 sampler 的 raw-only audit SHA=`73aff5e55eda314b8382813bd6a1feb3c64a25d3eda6bc11071f9161db224a23`。所有 population 最终只剩 `1--5` roots、root ESS=`1.00--2.74`、最大 root mass=`.49--1.00`；median root ESS 从 stage 15 的 `57.49/128` 到 stage 31 的 `1.22/128`，尽管很多单级 CESS 仍约 `.9N`、最大单级权重约 `.01`。因此失败的正确解释是“63 次强制 resampling 的家谱共祖”，不是 collapsed posterior 不存在，也不是全部 SMC/AIS 都失败；不能拿本 raw 补长、合并、降低家谱门或计算 q_top。由于这个机制的精确 base 已经是 iid prior，P/U/L 或物理全零态不应被硬塞作初态；任何后继非/稀疏 resampling 算法须另建新契约、fresh seeds 与独立权重/祖先证明，不能把改参数包装成本次成功。

- 2026-07-23：完成用户授权的 exp102 q=0 DTC21-S1 本地缺陷温度链预检，终态为 **`LOCAL_D0_TRANSPORT_NOT_VIABLE`**，不是正式 q_top 或服务器实验。冻结 hard cell 为 `m08_c06,p=.04,d00,attempt022`，21 层 `Kq=4→0`（热端为精确 iid Bernoulli(p)）、256 burn + 2048 固定测量时钟、P/U/合法低能 L 各 8 条；manifest SHA=`751f76bec3831fd8fad39ee96972bd2a5e54a3da4a2e87a90ba202554decb337`。有限 Kq 层上固定时钟条件 `D=0` 的分布确实等于目标 hard-coset posterior，但这只保证目标公式，不保证有限时间全局混合。24 raw 先通过完整 deterministic seed replay（report SHA=`58f1dbb227d748edeb266fe42fefd74768dc2384d3bcf2dfc850b6339000e49c`），再由不调用 sampler 的 raw-only audit 重建矩阵、syndrome、P/U/L、label、defect、D0 mask、counter 和门禁并通过（audit SHA=`6990ea671153446e65592b29f4d1a3ad08c954abb9767476e1ff193e4df8cb2f`）。关键负结果是 P/U/L 虽分别有 `166/61/201` 次 D0 label change、每条链也有很多 D0 leave-return，却只达到 delta rank `1/2/3`（门槛 16），各仅 `1/8` 条链有至少 8 次变化（门槛 6/8）；P/U/L 的 basis leave-return 仅 `15/8/19` 个。说明局部 defect closure、swap 或普通 state change 不能替代全局 logical transport。该固定机制/预算不进 HARD2、READY_FOR_FORMAL、held-out 或 production，raw 不可补长/合并/作 q_top；非零 syndrome 下物理全零态仍非法，不能用它替换 P/U/L 来制造表面收敛；这不是数学不可能结论。

- 2026-07-23：完成用户授权的 exp102 q=0 CTT64-S1 本地逻辑运输预检，终态为 **LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE**，不是正式 q_top 或服务器实验。冻结 manifest 为 f77add0a8b1825b117ac49ed85b3a3a138045cb233bed43fb691cac9bd31ff85，硬 cell 为 m08_c06,p=.04,d00,attempt022，P/U/L 各 8 条、512+4096 固定时钟。启动前发现原先“全部 reduced logical 异或”的 L 重量为 229（P 为 63），并非有意义的近能竞争初态；因此在任何 raw 前冻结为 reduced single/pair/triple 中确定性的最小 |P xor d| 非零逻辑候选，L=67。24 raw 的完整 seed replay 与独立 raw-only audit 均通过（report SHA=9361b4290111a06b8e029b2b692df591c0e4e692bc463a5a3ee5f2ae7f2200b2，audit SHA=ce2acd3e9cbc38b8d1be270248ed8bb94c8f21c936cde90dd534b90eb0697c9e）；实际 .04/64-level reference/Numba transcript 回归 21 passed。但 P/U/L 的 label-delta rank 仅 3/13/3，basis leave-return 仅 20/30/21 个，均远低于 64；P/L 的 32768 个 CTT path proposals 零接受，U 仅 1 次。U 虽有 2414 次、P/L 各约 200 次普通 label change，仍只在小子群内活动，不能被误当作全局混合。该固定机制/预算不进入 HARD2、READY_FOR_FORMAL、held-out 或 production，raw 不可补长/合并/作 q_top；这不是数学不可能结论。

- 2026-07-23：完成 exp102 q=0 logical-signature V0v2 的 fresh V0d 运输诊断，终态为 **`UNRESOLVED_LSI_IMH_V0_TRANSPORT`**，而非“仍在运行”。run=`exp102_q0_lsi_v0d_20260723_9f0c473`，source=`9f0c47370bac65059ed50507c95582f594d66df3`，archive SHA=`edc677d396b5a89588dba526e4f38ce1fbb0480a52f476fc3498630f6b232d48`，manifest SHA=`6557c30e888ef59cd5ca61fdd7bb0fb305019f90dae5b934b5bb9be179554e0b`。48 条 immutable raw、Linux nd-3 analyzer 与本机 conda-12 independent replay 均完成；两份报告文件 SHA 一致为 `89e5d6c4aaf0792e35050a2dacff1e205e490d3a5250ed1c2f3734c46b3729c4`，内部 report SHA=`64a05c06d07d0af4c0b27daded97687e5f830f227f03886c92d7f117aadd65a2`。两个温度的 P/L 合法低能初态均为 0 次跨 label 接受；U 虽有 `.5/.1` 下 `57/44` 次变化，却仅 4 个 sources、rank 3、`10/64` basis 和 `54/64` nonbasis leave-return，远低于冻结的 `128,6/8,16,64,64/64` 门槛。因此不能用总 acceptance、普通 state change 或把初态统一成零态掩盖失败；该 LSI 机制在本预算内没有低温全局运输，raw 不可补长/合并/进入 q_top，且没有 HARD2、formal、held-out 或 production 授权。

- 2026-07-23：用户授权的 exp102 q=0 logical-signature V0 窄诊断在 artifact portability gate 处终止，终态为 **`CONFLICT_CROSS_ENV_ARTIFACT_IDENTITY`**，没有启动任何 sampler trajectory。clean-source run=`exp102_q0_lsi_v0_20260723_b9a08a4`，source=`b9a08a4905e4c8e999e0c9e5b3408f20e83c4436`，archive SHA=`a53515a6af914077303b040caa6d3b5046af0054cf8bc3683c10289e1548ae53`，source-manifest SHA=`f151754e619f233e8abd544ea4a5d1bb6ec58cfc6c7f999866bd08680e0712a0`，config SHA=`55d410248adb5975aa162b0cc0406ffe1a0bfa8199435a04d5862b999b803f8e`。nd-1 `01_artifacts` 唯一 SUCCESS，但本机 artifact manifest=`f90fc8d23be45e7b5122424e96fe5d6769aa73cf20339dcc0e6da814db67e64f`、nd-1=`6171de3b81a6f84ba070ba62fb7c52620687284c860d0a0bc9513b8a51d74b98`，不能冻结为同一个 proposal。矩阵/codebook/syndrome/frame/source/registry 都一致；根因是 builder stack 不同：本机 `ldpc/numpy/scipy=2.4.1/2.4.1/1.17.0`，nd-1=`2.3.7/2.3.4/1.16.3`，导致 MILP base anchor 已不同，113566 个 BpLSD candidate 中 112866 个 decoded states、112093 个 recorded weights 不同，进而 128-anchor catalog、S-tail 与两条 proposal SHA 都不同。两边 candidate 均 `113566/113566 valid`，本机对拉回 nd-1 artifact 的 128 个 retained anchors 做 GF(2) syndrome/label/transcript linkage 复核通过，故这是严格的跨环境 identity conflict，不是 raw 损坏、MCMC 未收敛或物理结论。未创建 `V0_MANIFEST`、preflight、raw、transport report、tuning、held-out 或 production；不得原地重跑或复用。后续必须先另立契约，统一 pin decoder/solver 环境，或冻结单一 artifact producer 并把异平台限制为代数验证；仍须保留 P/U/合法 tail 初态，不能把非零 syndrome coset 中非法的物理全零态当作收敛捷径。

- 2026-07-22：完成 exp102 HGP screen v2 fresh immutable run `exp102_q0_hgp_screen_v2_20260722_4d134ee`，冻结终态为 **`UNRESOLVED_MAP_MIXTURE_FAIL / PRE-PILOT`**。source=`4d134ee7ca25125d341eb11cbfa34d6856514101`，archive SHA=`ad72d2c7039192be721b87ce7c96c5da577af05acd37cacd9167e26a773d9027`，manifest SHA=`5bafae76b06ff46557ae8315bb281a42256e7e4e50ed2e9dae868695114b8ff8`。nd-1/2/3 full exact preflight consensus PASS，选择 T3=`8192+32768`；macmini conda-12 对 portable projection 和四条 MAM acceptance-decision transcript 逐位复核为 `PORTABLE_PASS`，12 个 full mismatch 全部且仅是预注册 nonportable float。固定 nd-2/nd-3 ownership 下 384/384 sampler trajectories 与 2/2 IS 完成，无迁移、补样、retry 或 resampling；nd-3 analyzer 和本地 terminal audit 验证 386/386 raw。measurement-acceptance SHA=`42e12338dac640b725728f25c46b4d853a23e02392b2c9f2471f519ffcf5bba1`，joint-terminal SHA=`7e9bd8d7efb657649c4a0b4f0d146b72063d4584b291479a49c805e6834ab4f1`，terminal-package identity=`233e31e599180153f979a30dc971e8e8128be64505fd0572d68bc1ae87a64041`，local attestation=`386e8a0eeadb5c24b376014b522dec36322456abf3b0d636c1ad16cc7681c755`。HP64 通过 5/5，是当前 promising candidate；HP32 为 3/5，m3 仅一个 B character 以 `0.0404396` 对 `.04` 边缘失败，m5 则明确慢（U B max Rhat `1.1552`、min ESS `327.0`）。MAM-IMH8 为 1/2；m8 P full family Rhat=`1.06088`、ESS=`379.74`，P/U B max Rhat=`1.08245/1.05662`、min ESS=`275.33/361.16`，16 个 B characters 不一致。HP64/MAM 四个 family-cell comparison 全败：m6 `q_top=.14587/.16241` 的差小于 `.04` 但约 30 SE，m8 为 `.91317/.99273`、差约 `.0795`。raw 红队进一步发现 m8 两个不同 MAP anchors 的 64-bit logical coordinate 完全相同；P/U 虽分别有 39899/40735 次 measurement state changes，logical-label changes 只有 330/288（`.827%/.707%`），典型链仅访问 3 个 labels，`theta_logical=.08/.25/.5` components 零接受。因此普通 acceptance、总 state changes 与 global IS ESS 优化的是同扇区运动，不能代表所需 global mixing；后续应先做结果无关的 logical-signature-diverse anchors/独立机制，并冻结 cross-signature transport gate，而不是直接延长 MAM。五个 syndrome 均非零，物理全零态不在支持上；P/U 不能被统一零初态替代。该 run 未认证任何 `(m,p)`，没有 `READY_FOR_FORMAL`、held-out、`FROZEN_HELD_OUT_PASS` 或 production。

- 2026-07-22：exp102 HGP screen v1 fresh run `exp102_q0_hgp_screen_20260722_2e6ba2a`（source `2e6ba2a864d7db6ae04e79867d1678dbcfe42580`）虽在三 Linux 节点 preflight aggregate PASS 并选中 T3，但 macOS 审计先发现 solver provenance 被误当成本地版本要求，继续审计又发现 MAM `log_q/acceptance` 的 1 ULP 差和两个 50000-draw IS full digest 漂移；真正使 v1 终止为 `CONFLICT` 的是后两项漂移，不是 solver provenance verifier 错误。v1 没有生成 measurement control/raw/result，禁止原地续跑。当时改为 **PRE-RUN** `exp102.q0_hgp_global.screen.v2`，config `q0_hgp_global.screen.v2.json`、SHA `38092ec030f6c283f163c0ddb3eed612aa850c76ce34f130520522646fa883dc`：三 Linux 节点做 full exact consensus，本地只对冻结 portable projection 与 MAM decisions 作 exact 复核，无 ULP 容差；anchor catalog 升为 v3，sampler/auxiliary namespaces 全新，并加入 HARD2 两 cell × P/U 的四条 `256+2048` MAM portability probes。红队结论保持：非零 syndrome 下全零物理态不在 hard-coset 支持上，不能靠非法零初态制造表面收敛；HP 也只能声明固定时钟离散输出与 transport counters 一致，不能声称隐藏内部决定逐位一致。有限 character/16 个 U 仍可能漏掉高阶 mode 或小体积高后验 basin，HP/MAM 也可能共同漏模，所以全门通过仍不是混合证明。该条记录时 v2 尚未启动，最高交付仍仅 `DIAGNOSTIC_HARD_PAIR_FOUND`。

- 2026-07-22：完成 exp102 q=0 新 HGP hard-pair 诊断的本地实现与启动前红队，状态仍为 **`PRE-RUN / DIAGNOSTIC TEST AUTHORIZED`**，服务器 preflight/measurement 尚未启动。新契约 `exp102.q0_hgp_global.screen.v1` 冻结 config SHA `3c65ef96ce231b4aea4499b5a6030f1cc82475117c5ee5ecc7633d972ef8edc9`：HP32/HP64 用 HGP exact collapsed marginal + likelihood-power replica exchange，MAM-IMH8 用 full-support multi-anchor independence MH；计划 384 条独立轨迹（HP 的 HARD2+EASY3 共 320，MAM 的 HARD2 共 64），每 cell 均为 P/U 两类合法对抗初态各 16 条。五个 syndrome 全非零，故全零态不在目标 hard coset，不能用“统一从 0 开始”换取表面收敛。盲区审计另发现每轮精确重抽 `A|B` 会用 A 的新鲜条件噪声掩盖真正慢变量 B，现 raw/replay 独立检查全部 B bits、行列 parity、64 dense characters、`|B|` 与 `L(B)`；P/U 和 HP/MAM 除整体 D2 外还逐 character 检查 `.04` 与 `3SE+.005`，避免单个冻结 bit 被平均稀释。HP/MAM 在 RNG 前强绑定 canonical observable frame，frozen MAP artifact 可跨 SciPy 版本重放但不能绕过 GF(2)/hash/proposal 检查；n=10,k=4 与 n=13,k=1 production-path exact oracle 覆盖 zero/nonzero syndrome、p=.04/.10/.25。013 工作流分成 preflight/measurement 两阶段：三节点 aggregate PASS 后必须拉回 macmini 用 conda-12 clean archive 逐位审计，当前 source 只接受 exact PASS attestation，才允许物化 control。HGP focused `124 passed`，exp102+exp101 全量 `747 passed`（2 个预期 deprecated-alias warnings），compileall、全部 exp102 shell syntax、所有 013 CLI help 与 `git diff --check` 均通过。即使后续 screen 全过，最高也只是 `DIAGNOSTIC_HARD_PAIR_FOUND`，不能生成 `READY_FOR_FORMAL`、held-out 或 6144-task production。

- 2026-07-22：完成 exp102 q=0 global sampler 的 fresh `HARD2+EASY3` diagnostic screen，冻结终态为 **`UNRESOLVED_NO_HARD_COSET_PASS / PRE-PILOT`**。immutable run `exp102_q0_screen_diagnostic_20260721_342dd5b` 使用 source `342dd5bc0fb2c7694dbc58a8d0f2d92689c24991`、archive SHA `4a54ba28f3ee2add94e93dd052e4bda567d5e008691f84a098c21768b4fe11f3`、manifest SHA `2b8ab6d238d6319ea73c4c5da0ecf815a3d2e2ea28932dddc30bd40afe158b01` 和 schedule-file SHA `f9aeccd95640a56fabe813796d0e1ce388cffa1bcccf2405a6bafcd913520832`。nd-1/2/3 preflight 全过，canonical digest 同为 `080b3170ca168dc3f237d22a4d18403eb2c0b0b2455e6d1e3ca876aae39c86a9`，4096 项 gamma SHA 精确为 `a2c459ec9438e23f863c44528ac093c5b93d891b6a8bec0278b873fe47f2459a`；最大共同资源档 T3=`burn 8192 + measurement 32768`。15/15 bias 与 1280/1280 fresh trajectories 全部完成，7 个 node marker 唯一 SUCCESS，raw 身份/SHA/代数/逐位 replay 均通过且 `reused=0`。五个 hard-coset 方法均 0/5：25/25 summaries 保留 P/U 初始化依赖，全部 U family `Rhat>1.05` 且 ESS<400，`|delta q_top|=0.06695..0.991999` 超过 0.04 门槛；三个 defect-trace 方法也均 0/5，480/480 chains 的 fixed-clock D=0 observation 和完整 leave-return excursion 都为 0，无法形成 conditional estimator。终端 package identity SHA=`0e0fb2f950eb609c984b29f5647321694c82f8f7a6810609fd1742d1472a990a`，`selected_pair=null`。本地 conda-12 从同一 verified archive 独立重跑 analyzer，raw replay、门禁和终态一致；跨平台仅 62 个派生 core-time 与 18 个派生 ESS 相差最多 4 ULP（最大绝对差 `1.82e-12`），已由白名单 verifier 审计。metadata evidence 位于 `validation/011.../completed_run_evidence/`，485 MiB raw 留在本地和远端 `runs/`、不入 git。结论只表示这些算法在 T3 预算内未收敛，不是基础设施失败、`IMPOSSIBLE` 或正式物理结果；`READY_FOR_FORMAL`、held-out、`FROZEN_HELD_OUT_PASS` 与 production 仍不存在。

- 2026-07-21：用户批准启动 exp102 q=0 global sampler 的隔离 `HARD2+EASY3` diagnostic screen。首个 immutable run `exp102_q0_screen_diagnostic_20260721_5e1f5aa` 已完成三节点 preflight（全 8 方法均入选 T3）和 15/15 bias tasks，但 nd-0 在物化 1280-task measurement control 前独立 replay 发现 `gammas` 跨节点不逐位一致，终止为 **`CONFLICT_CROSS_NODE_GAMMA_LIBM`**；measurement raw=0，不是 sampler 不收敛或参数点失败。根因是旧 `_tuning_gammas` 的 scalar fractional power 调用了平台 `libm`：nd-1/nd-3 只在 index 3171/4082 相差 1--2 ULP，但严格门禁正确拒绝。旧 run 永久关闭，metadata-only 证据和 verifier 位于 `validation/011.../failed_run_evidence/`，15 个 NPZ 不入 git。修复将 `.6` 明确解释为精确 `3/5`，以 96 位 Decimal + 固定 32 次整数五次根 Newton 生成 schedule；4096 项 versioned SHA 冻结为 `a2c459ec9438e23f863c44528ac093c5b93d891b6a8bec0278b873fe47f2459a`，运行时自检并进入三节点 digest v2。reference/Numba 已覆盖 DT16/32/64，fresh commit/archive/run 将从 schedule/preflight/bias 全部重跑；即使通过也最多是 `DIAGNOSTIC_SCREEN_PAIR_FOUND`，不授权正式 held-out/production。

- 2026-07-21：完成 exp102 q=0 global-sampling discovery 的终止性远端 preflight，冻结结论为 **RUNTIME_EXHAUSTED / PRE-PILOT**。第三个 immutable run `exp102_q0_global_20260721_204b37d` 使用 source `204b37d8e00e7d11ffa2b6766b90d947892e179d`、archive SHA `1583dce6b8bb81ad7780f323d21300b158ad435d710f3c0226b7b3028b8eb7f7`、manifest SHA `b69290798a11a3bf548483c6e223f96a64e0d9c7be0e48b89fa6e54a28a57ea3` 与 schedule-file SHA `7874a0d967ba866d8834cf380b408947af614bdf3bec7b50c0f30fb4a332465c`。本地 clean archive exp102+exp101 `590 passed` 且测试后逐文件复验通过；nd-1/2/3 verified workers 均写唯一 SUCCESS，canonical digest 同为 `a3730d7380575976f88e35f5490b24a9b6949e3817b2fb3880775736cf2ad364`，WMC 六点均为 `INCONCLUSIVE_WIDTH`。所有 hard-coset/defect 候选均可选 T3，全日程 2x 投影仅 1.307/2.441/2.055h；但必需 m3 full-sector TI contingency 在三节点为 78705/116275/251241s，nd-2/nd-3 超过冻结 79200s 窗口，worst-node consensus 因而合法 `RUNTIME_EXHAUSTED`。按契约未启动 bias、screen、HARD2 fresh、confirmation、resolution 或 TI sampler，sampler raw=0；全范围只能记 `UNRESOLVED_WITHIN_ALGORITHM_AND_72H_BUDGET`，不能写 `IMPOSSIBLE` 或外推为参数点失败。旧 combiner 只接受 PASS、未落 aggregate exhausted report 的审计缺口已修为持久化合法终止状态，下游仍严格要求 aggregate PASS，未改任何门槛/安全系数/deadline，也未重跑本 run。证据、全文件 SHA 与独立 verifier 位于 `validation/010_q0_global_runtime_exhausted_20260721/`；前两次基础设施失败继续永久归档于 008/009。`READY_FOR_FORMAL`、正式 sampler、held-out、`FROZEN_HELD_OUT_PASS` 与 6144-task production 均不存在。

- 2026-07-21：完成 exp102 q=0 PA discovery 与 PT transport autopsy，冻结结论为 **EXHAUSTED / PRE-PILOT**。worker source `f0dff0f8d3e055227b75c999a73c751e2a576768`、archive SHA `57811c43662b379524fb4f5099346f042d5577cc1e2c69a31299a11fd9c01324`；nd-1/2/3 canonical digest 同为 `f4ed9fff7512f8995a4f70c60072c1bba054aaf75e0440a4d00545880305f478`。nd-2 Linux runtime 四门禁全过（m8 最慢 `56.91 us/particle-sweep`、startup `1.80s`、单 population `0.373min`、2x 全日程 `1.064min`）。四个 autopsy raw 全部逐位复现旧 labels/swap/logical/transport/residual，但均因所需 outbound conditional attempts<200 判 `INCONCLUSIVE`；D0/D4 的 m6 分别有 3/5 次 hot update，四任务仍全为 0 return。64 个 PA hard-screen population 全部完整且身份/哈希可重算，但四方法在两个 hard cells 上均失败：每个 population 都发生 genealogy 塌缩，cell median family ESS≈1、distinct initial families=1--2（门槛 8/16），B96 部分任务另有 CESS/max-weight fail。按零方法通过分支禁止 B384-2 rescue，未冻结或运行 confirmation/resolution，不存在 `READY_FOR_FORMAL`、正式 PA config、held-out、`FROZEN_HELD_OUT_PASS` 或 production。报告位于 `validation/006_pa_discovery_20260721/{hard_screen_report.json,transport_autopsy_report.json}`，raw 已回收至 `data/expander_code/exp102/raw/pa_discovery/exp102_pa_discovery_20260721_f0dff0f_r2/`。首个同 SHA run 因直接在 shared source 运行 Python 留下 `__pycache__`，被 verified wrapper exit 67 全节点拒绝且无 raw；r2 使用 fresh bit-identical archive 并全程走 wrapper。事后 analyzer 仅修复证据可移植性：NumPy 2.3/2.4 的 `exp`/reduction 在全 64-task replay 中最大差异为 cumulative log-Z 的 4096 ULP（绝对 `5.68e-14`），现对 ladder/非累计/累计 log-Z 分别限 8/64/`32*G` ULP；所有离散 transcript、gate、parent/ancestry/counter/hash 仍严格一致，不改变数值结论。

- 2026-07-20：完成 exp102 PT-v2 固定 Q32 梯度 + multi-swap discovery，并按预定停止条件判为 **EXHAUSTED**，未启动确认面板、正式 pilot 或 6144-task production。clean source `da69528b43f4a9d1635083c21d713ba63ccec4ab` 在 nd-1/2/3 通过全套 exp102 preflight，三节点旧 S1/PT-v2 digest 分别一致为 `b9a5c8b22d8b2421723705b1567b825a5a1775a8efd20748e884436f8bee959f` / `38f29fe037bcce399883b6f6d20b4500f54ba11e94ea5e8b98b586e8e402f659`。9-cell screen：D0/D2/D3/D4 均 9/9，D1 为 8/9（单个 swap rate 0.1952<0.20）；随后 4 梯度×S{4,16,64}×2 hard cells 共 24 cells/96 条实例轨迹全部未过 transport。长程 group min swap 仍有 0.156–0.392，hot-logical/residual 也过门禁，说明梯度窄门已修；但只有 13 条轨迹真正经历热端局部更新（共 27 visit），未认证/认证/sector-changing 往返全为 0。每个 S64 候选均有 `min_hot_updated_visits=0`，故协议不允许追加 S128。raw 已回收到 `data/expander_code/exp102/raw/discovery/exp102_discovery_v2_20260720_da69528/`；新 report-v3 在读 NPZ 前严格核对 6 个 node manifest、control/ownership/LPT、source archive、status 和 SUCCESS，analysis SHA256=`957142537155e3bf57e03620a6e11cc2cfa1df24c5fa4e4b04f1e7fd9e4987a6`。远端部署 `repos/` 与 Numba cache 已清，`runs/` 69 个 NPZ 和 `logs/` 保留。正式版本继续停在 `exp102.q0_pt.v1`；若继续必须换经审查的算法/契约，不能延长轮数或降低 gate。

- 2026-07-20：按用户批准的方案完成 exp102 R96/R128 clean-SHA pilot，但 ladder 在上限处正式 `EXHAUSTED`，因此未启动 6144-task production。source `2b01d9dcb463ec47a1b30202fc9105430b95e18c` 在 nd-1/2/3 通过 preflight，三节点 reference/Numba digest 同为 `b9a5c8b22d8b2421723705b1567b825a5a1775a8efd20748e884436f8bee959f`；nd-2/nd-3 完成 23 个有序 ladder pair，共 13,056 raw cells。merge-select 选中 m3=`(p_hot=0.45,R=64)`、m4=`(0.49,96)`；最大 `(0.49,128)` 下 m5..8 仅 `94/96,94/96,93/96,94/96`，失败全部集中在 `p=0.04` 的 min swap-rate `<0.15`，不是 accepts、缺文件或身份问题。远端 13,270 文件已用 `SHA256SUMS` 回收并本地逐一验证，pilot report SHA256 为 `a122f77d5bfbb087ec25217c87dde3447c17d3d79aea57442fad0d9987d87c12`。按 fail-closed 契约停止，未跑 gamma/rounds/held-out，未生成 freezer/task plan；若继续必须新审查并提交更强 ladder/config，不能手工放行。

- 2026-07-20：exp102 ladder 完成后追加 stage-semantics 审计：发现 `ladder/gamma` 错误继承了 rounds 才需要的 character trace/constant gate，已新增 `require_trace_gate=False` 分支和单测；该错误使旧报告过度拒绝，但不改变 production blocker。直接按 raw counters 只重算计划规定的 residual+swap+hot gate 后，最大 `(p_hot=0.49,R=64)` 的 m4..8 仍仅 `93/96,89/96,85/96,84/96,87/96`，所有剩余失败均含 sub-0.15 swap edge，仍没有任何可冻结 pair。故当前必须停止；若扩展 R/ladder family，须从新 clean SHA 重跑，不得复用旧 stored valid。

- 2026-07-20：exp102 新 clean SHA `bbe72da` 在 nd-1/nd-2/nd-3 通过 preflight（各 14 tests，canonical digest 同为 `9af48083dff55741b662aed24815a88b82a2bf3d484e8231064f0b2dee753827`），随后用 full-round Numba 在 nd-2/nd-3 完成冻结 schedule 的全部 `10752/10752` ladder cells。raw merge-select 重算结论为 **PILOT_LADDER_FAILED**：只有 m=3 在 `(p_hot=0.45,R=64)` 达到 96/96；m=4..8 到最大 `(0.49,64)` 仍仅 `90/96,88/96,80/96,72/96,37/96`，失败含 swap<0.15 和 large-m constant-untrusted。按契约“最大候选仍失败则停止该 m”，未启动 gamma/rounds/held-out，未生成 freezer/6144-task plan，也未启动 production。正式实验当前没有合法 ETA；若用户批准扩展 R 或更换 ladder family，full-round Numba 本身使 MCMC 不再是主要墙时瓶颈。

- 2026-07-20：启动 exp102 正式实验前门禁流程，但尚未获准进入 6144-task production。旧 `70191fb` 的首轮 ladder `(p_hot=0.45,R=8,gamma=1,burn=500,measurement=2000)` 在 nd-2/nd-3 完成 `576/576`，各 m 均 `0/96 valid`，失败主因为 swap edge；随后 R12 跑到约一半时发现旧“Numba”仅 JIT 单个 delta、整条 PT 仍由 Python 驱动，因此把 R12 明确标为 `OBSOLETE_SOURCE` 并停止，保留 partial raw 作审计。现已实现 full-round Numba：CSR supports、rung→replica 映射、增量 `uint64` label、PortablePrng 镜像、稀疏末态 residual；reference/Numba canonical digest 保持 `9af48083dff55741b662aed24815a88b82a2bf3d484e8231064f0b2dee753827`，本地 m3/m8 benchmark 提速约 `177x/196x`，exp102+exp101 关键回归 `107 passed`。同时补齐 raw 重算的 `pilot merge-select`、不可手写绕过的 freezer、三节点 production shard/部署 manifest 工具和 fail-closed stage wrapper。下一步从新的 clean SHA 重跑三节点 smoke 和完整 pilot；真实 `FROZEN_HELD_OUT_PASS` 前仍禁止 production。

- 2026-07-19：继续完成 exp102 的生产前阻断修复，状态仍为 **IMPLEMENTATION / PRE-PILOT**，未启动 production。q=0 PT 新增 `reference|numba` 公共引擎；Numba 保持同一 PRNG 消费顺序并在非零 syndrome smoke 中与 reference 的 labels/swap/logical/round-trip 逐位一致。transport 改为热端到达标签与冷端返回标签的净变化，偶数次 logical flip 相消不再算 sector-changing round trip；character U-statistic 改为六对独立链 character mean 的乘积。worker/raw/aggregate/loader 增加 Numba、source、registry/config、section/frame、shape/dtype、6144 present task 等 fail-closed 身份校验，pilot freezer 必须验证 merge-select provenance 与 raw SHA256，删除 pre-pilot draft production task plan。registry SHA256 保持 `883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b`；production config SHA256 更新为 `758e2804476c5cb0422ef5813952a3779722c1a3ed47a7298f3948f9daee241f`。conda `12` 下 exp102 + exp101 HGP/logical/PT/loader/reference 回归 `83 passed`。远端只做了只读占用检查，nd-2/nd-3 当前空闲；clean commit、跨节点 preflight 与首轮 ladder screen 仍待完成。

- 2026-07-19：完成 exp102 随机 HGP `q=0` threshold 扫描的独立实现与 pre-pilot 冻结，当前状态 **IMPLEMENTATION / PRE-PILOT**，尚无物理曲线。新增 `exp102.physics.v1 / exp102.q0_pt.v1 / exp102.scan.v1`：hard-coset data-only PT 只做 stabilizer/logical move，四独立实例以 `uint64` logical label 保存，六对 label collision 给出 qtop，三级聚合对 disorder/code/m fail-closed，专用 publication loader校验 contract/registry/config/task/errorbar。系统熵生成并冻结 48 个 code（m=3..8 各 8 个），registry SHA256 `883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b`；尺寸 `(n,k)=(225,9)..(1600,64)` 全符合计划，6144 个 task identity 无缺失/重复。生成独立 pilot schedule，worker 在 held-out 配置成为 `FROZEN_HELD_OUT_PASS` 前拒绝生产。conda `12` 下 exp102 6 tests PASS，exp101 HGP/logical/PT/loader 关键回归 65 tests PASS。下一步只能按 schedule 跑 pilot 调参和全 p held-out；通过后再做三节点 smoke，不能直接启动 production。

- 2026-07-28：完成 exp102 validations 060--061 的本地结构筛选与 Stage-0 governance。060 的 one-shot/独立 bitset audit 终态为 **`LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND`** / `INDEPENDENT_STRUCTURE_AUDIT_PASS`（audit SHA=`719191b1e566f9ec9cdbc811dca73022082547de6cb15faba834cac232aa00fb`）：MR2 是唯一 survivor，induced width=`25`、单表下界 `512 MiB`；MR3/MR4/RC1 均超 width/memory gate。MR2 只可在 HP64 后续真实失败后作一次同族 contingency，不能充当 large-k 独立确认。061 在 canonical worktree 完成 dirty-root、worktree 与 validations 001--060 authority inventory；没有选择或复制脏文件，也没有产生 cell/formal/held-out/production 权限。
- 2026-07-28：完成 exp102 validation 062 本地 character-gate one-shot 与独立 audit，终态 **`CHARACTER_GATE_REDESIGN_REQUIRED`** / `INDEPENDENT_AUDIT_PASS_CHARACTER_GATE_REDESIGN_REQUIRED`（report SHA=`8a36cf41397e6332e9a9c789e5217cfd8d2274e68f85a4b4b591ede5e13d488a`）。五个共同 operating points 全失败；最大 `(32 trajectories,16384 draws)` 虽通过 exact logical、exact collapsed-B 和 synthetic logical-511，synthetic B-688/logical-4160 仍未满足 simultaneous coverage，Wilson lower=`.9779025636<.98`。校准同时证明 `.04` character maximum 对完整 catalog 至多给 `.08` purity 差界，有限 sampled characters 不覆盖未观测 tail；下一门禁须直接对齐 `q_top`，并独立保留 full-label `D2`，不能把 character maximum 当交付量。
- 2026-07-28：完成 exp102 validation 063 Nishimori 辅助校准及 065 fresh audit rebind。063 exact report 含 30 rows，14 个 correct-posterior groups 在 `N=2048` 未达到 `.01` simultaneous precision，科学终态保持 **`NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT`**（report SHA=`134228b993a7b856c143d5410de7940f51657f0d1b7c38bcad1fd6cd917af441`）；旧 auditor 又先因两个英文 failure prefix 不同而冲突。065 独立重算进一步发现三个 exact weight-enumerator MAP ties，使合法但不同的浮点 `argmax` 选择造成 11 个 payload mismatch、最大差 `.03400704`；终态为 **`CONFLICT_INDEPENDENT_NUMERICAL_RECOMPUTATION_MAP_TIE_SEMANTICS`**，独立 verifier 通过的是“冲突被正确记录”（audit/verification SHA=`5d49532e...13ab/03cb4d1e...95ccc`），不是原 report audit PASS。`terminal_gate_invariant=true` 不等于 `full_payload_match=true`；后续 MAP control 必须预冻结 exact canonical tie 语义。
- 2026-07-28：完成 exp102 validation 064 对 validation 013 HP64 raw 的四进程证据/资源校准，独立 package audit PASS（SHA=`ee349e4f35da59dc4244c873ae3d708a9c41181d78b65c6628d883bce8dca9c4`），资源终态 **`RESOURCE_SCENARIOS_ONLY_EMPIRICAL_COVERAGE_INCOMPLETE`**。审计确认 m8 `.91317/.99273` 是 HP64/MAM 而非 HP64 P/U，后者只差约 `.000648`；m6 P-family HP64/MAM 差 `.0165964`，约 `30.59 SE`。由于缺 m7、绝大多数 p 和跨 code/disorder timing，strict campaign totals 全为 `null`；8-trajectory T1 same-m proxy 的 full-grid `162495` safety core-hours、75-core ideal wall `2166.6h` 及 m3 easy block `76.96` core-hours都只用于规划，不是 confidence bound。060--065 全程没有启动远端 measurement；当前停止来自 character/auxiliary audit/资源覆盖等科学门禁，不是在等待服务器，仍无 certified cell、`READY_FOR_FORMAL`、held-out 或 production 权限。
- 2026-07-30：完成 exp102 validation 066 fresh local-only delivery-gate one-shot 与独立 seed replay。clean source=`bc47ae26dd26203f2b9c902feca2a10ea797c798`；五个 selection points 的 validity/outcome 依次为 `INVALID/FAIL/FAIL/INCONCLUSIVE/PASS`（首点在 report 中记录为 `INCONCLUSIVE + NO_FINITE_CALIBRATED_MULTIPLIER`），最终选中 32 个 IID multinomial trajectory groups x 16384 independent draws（不是 MCMC chains/clocks 或 ESS）、multiplier=`4.809673164164152`，fresh confirmation=`PASS`，终态 **`LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED`**。4,372,205-byte report self-SHA/file SHA=`d255c67ee0a91985e933ccea8a9616c63e832e37c19cd16dc7eb5e35f05e5a0a/f11a3eb137793ce2bbe43734db82240cde45bafbdd57a2a1e6f97d520dad6ed8`；独立 auditor 在冻结的 NumPy 2.4.1 + `default_rng` + PCG64 同环境从 seeds 重建 histogram、full-label collision `q_top`、`D2_norm`、四组 delete-one、group-wise SE、calibration、三态决策、Wilson、selection/confirmation，audit self-SHA/file SHA=`485b789c3a86893662241ab0e529358fedde18b695c3514233adc492236261b3/3975de5eb1d9cebcc467efdd67d956dcfa4b98e4c3b205011a549d6cf8d7822c`。receipt 不是 persistent raw，且不声称跨 NumPy 版本 portable replay。两个 common-wrong `EXPECTED_KNOWN_BLIND` controls 的真实 `D2_norm=.0625` 而 candidate PASS rate=`1.0`，实证了共同错误收敛盲区；因此本结果只确认 scalar delivery gate 的本地 operating characteristics，不证明 MCMC mixing、target-basin 或未访问 tail coverage。项目仍为 **`BLOCKED_BEFORE_REMOTE`**：`LARGE_K_ORTHOGONAL_CONFIRMER_PORTFOLIO_UNFROZEN`、`FUTURE_SCHEMA_RUNTIME_COVERAGE_INCOMPLETE`、`CAMPAIGN_BUDGET_UNAPPROVED`、`STAGE3_MULTI_COMPARISON_MULTIPLICITY_UNFROZEN` 四项仍未解决；不授权 m3、remote、formal、held-out 或 production。post-run conda-12 完整回归为 `1090 passed, 4 existing warnings`。
