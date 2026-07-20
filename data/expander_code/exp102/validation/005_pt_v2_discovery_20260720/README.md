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
