# CTT V0 local logical-transport preflight

This directory defines a bounded local diagnostic for the exact collapsed
tempered-transition (CTT) kernel.  It is not a posterior experiment and cannot
produce `q_top`, `READY_FOR_FORMAL`, or a production authorization.

## Frozen input

- Contract/config/raw/report: `exp102.q0_hgp_ctt.v0` and its `.v1` artifacts.
- Cell: `m08_c06`, `p=.04`, `d00`, `attempt022`.
- Kernel: `CTT64-S1`, 64 fixed squared lambda levels, one reversible random
  8-bit B-block heatbath power per nonzero level, and the exact iid Bernoulli-B
  endpoint.  The palindromic path ends after the return `T_1`; it never appends
  a final cold-level update.
- Resources: 512 burn and 4096 fixed measurement macrosteps, eight trajectories
  each from P, exact-K0 uniform U, and deterministic near-energy L starts.
- Manifest: `local_m8_transport_ctt_v0/MANIFEST.json`, SHA256
  `f77add0a8b1825b117ac49ed85b3a3a138045cb233bed43fb691cac9bd31ff85`.

`P` is the planted error.  `U` is exactly uniform on the nonzero-syndrome hard
coset.  Physical all-zero is illegal for this cell, and shifted-coordinate
zero is already P.  The initial L design deliberately considers every reduced
logical single/pair/triple before any trajectory exists, then freezes the
minimum `|P xor d|` candidate with deterministic `(target weight, move weight,
signature, packed support)` ties.  For this frozen cell it has target weight
67 versus P's 63; the former all-directions XOR would have had weight 229 and
is not used as an adversarial low-energy start.

## Decision rule

Raw must be independently rerun from seed and match bit-for-bit apart from
timing.  Each P/U/L family separately requires at least 128 fixed-clock label
changes, six of eight chains with at least eight changes, rank 64 in observed
label deltas, and leave-return coverage of all 64 basis plus all 64 frozen
nonbasis characters.  CTT acceptance and B-bit changes are stored as path
diagnostics only: neither can substitute for logical-label transport.

The only success string is `LOCAL_LOGICAL_TRANSPORT_VIABLE_FOR_HARD2_SCREEN`.
It merely permits designing a fresh remote hard-cell screen with an independent
 confirmer.  A failure is terminal for this fixed local configuration: raw is
 not extended, pooled, or reinterpreted as a posterior estimate.

## Terminal result

All 24 trajectories completed, passed a full deterministic seed replay, and
then passed the separate raw-only audit in
local_m8_transport_ctt_v0/INDEPENDENT_AUDIT.json. The terminal result is
LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE, with report SHA256
9361b4290111a06b8e029b2b692df591c0e4e692bc463a5a3ee5f2ae7f2200b2 and
independent-audit SHA256
ce2acd3e9cbc38b8d1be270248ed8bb94c8f21c936cde90dd534b90eb0697c9e.
The frozen actual-path reference/Numba regression also passes (21 passed).

| family | label changes | delta rank | basis LR | nonbasis LR | CTT accepts |
|:--|--:|--:|--:|--:|
| P | 228 | 3 | 20/64 | 59/64 | 0/32768 |
| U | 2414 | 13 | 30/64 | 64/64 | 1/32768 |
| L | 212 | 3 | 21/64 | 57/64 | 0/32768 |

Every chain satisfies the deliberately weak eight-change count, which is
useful negative evidence: ordinary label changes and conditional A redraws can
remain inside a small logical subgroup. They do not replace the rank-64 and
all-character transport requirements. This fixed CTT configuration is
therefore not eligible for a remote HARD2 screen or any formal exp102 stage.
