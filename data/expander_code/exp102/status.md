# exp102 status

**PA DISCOVERY EXHAUSTED / PRE-PILOT — formal pilot blocked, production not started**

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
