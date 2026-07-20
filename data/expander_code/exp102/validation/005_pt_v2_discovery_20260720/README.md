# exp102 PT v2 discovery

This directory contains the isolated discovery launch and cross-node verification code for fixed
Q32 ladders plus multiple swap sub-sweeps per local round. Discovery raw uses
`exp102.discovery.raw.v2`; it is design evidence and is deliberately incompatible with the formal
pilot/freezer schema.

The frozen candidate and panel data live in `config/discovery.v2.json`. The stages are:

1. five-ladder screen on the nine attempt-022 failures;
2. transport search on the two frozen hard cells;
3. sequential 17-cell confirmation until two distinct ladder IDs pass.

`orchestrate_discovery.py` probes all three node loads before it freezes ownership. By default it
uses `nd-1/2/3`, but creates a fixed `nd-2/3` fallback before launch when `nd-1` exceeds the load
threshold. Ownership never changes during a stage. `run_discovery_wrapper.sh` embeds the combined
source/config/control/ladder/swap/m-set fingerprint in every terminal marker.

Local implementation smoke before the clean-source remote run:

- discovery cell: D0, `m05_c00/p=.04/d02`, `S=1`, `500/2000`, computed and independently reloaded;
- reference/Numba cross-node candidate digest: `8b775df42cbbc866717ccb9b995555bb778f2bdd95635c587dc15c104535ceb4`;
- no formal pilot, held-out, freezer, task plan, or production task was started.

## Clean-source result

Run `exp102_discovery_v2_20260720_da69528` used source
`da69528b43f4a9d1635083c21d713ba63ccec4ab`. All three Linux nodes passed the exp102 suite and
agreed on both canonical digests: old-S1
`b9a5c8b22d8b2421723705b1567b825a5a1775a8efd20748e884436f8bee959f` and PT-v2
`38f29fe037bcce399883b6f6d20b4500f54ba11e94ea5e8b98b586e8e402f659`.

The 45-cell screen result was D0=9/9, D1=8/9, D2=9/9, D3=9/9, D4=9/9. D1 alone failed because
one instance had `min_swap_rate=0.1952 < 0.20`. Transport therefore tested D0/D2/D3/D4 at
`S=4,16,64`, `burn=2000`, `measurement=8000` on both frozen hard cells. Every one of the 12
candidate groups had 0/2 passing cells. The group minimum swap rates stayed between
`0.156053125` and `0.39155`, and the only stored failures were round-trip and sector-transport
gates. Across 96 instance trajectories, 13 reached a later hot-rung update for 27 total visits,
but none returned to cold: uncertified round trips=0, certified round trips=0, and
sector-changing round trips=0.

Every S64 group has `min_hot_updated_visits=0`, including all four m8 hard-cell instances. The
protocol therefore forbids conditional S128 and declares the PT-v2 route `EXHAUSTED`; confirmation
was not run and no formal configuration is ready.

The report was regenerated with `exp102.discovery.report.v3` after fail-closed evidence hardening.
Before reading NPZ, it now verifies exact raw-manifest coverage and hashes, canonical control
tasks, deterministic LPT ownership, the common source archive, node status, stage fingerprint, and
exclusive SUCCESS marker. Evidence identities are:

- source archive: `5ce2f669258088669a3f9c19af840ce9b580aa56cd3c55ed33ae6ddbb3142c6b`;
- source manifest: `2643d84f31b9c7f0fa1320f9fd4055e12c7c922dd043a03c360a04722dbd1240`;
- screen control/ownership: `27dc31ddf0eb74356f735d18dbcef1583505000ceaf4fe4226b7b7284d907fe7` /
  `5b9f3c129c5aba93b2c851e30103a0fa7aaf72989b80b0b8c52e7b3197a7d72d`;
- transport control/ownership: `5480511a57d144644c308a4298f515ffc1d09afebefaeb09aceca1b64c89437b` /
  `43b4b21e4cbb55f1dbe06bc2ddf5b2e2f94a98952ccdca9ed9d84c6c2c5b0b47`;
- hardened analysis SHA256: `957142537155e3bf57e03620a6e11cc2cfa1df24c5fa4e4b04f1e7fd9e4987a6`.
