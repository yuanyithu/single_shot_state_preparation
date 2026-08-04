# exp103 remote execution amendment

Amendment identity: `exp103.remote_execution.v1`.

This amendment records the user's authorization to qualify and, only after the
new remote gates pass, execute the existing `exp103.decoder_mc.v1` protocol on a
remote server. It supersedes only the local-only execution and resource clauses
of `EXPERIMENT_CONTRACT.md`. The original local Validation 002 result remains
immutable evidence and is not reclassified as a pass.

## Unchanged scientific protocol

The following items remain value-for-value frozen from the original protocol:

- all 48 registry codes with equal weight, including every `d=2` member;
- `sector=x_error`, `H_check=H_Z`, perfect syndrome and `q=0` Bernoulli-X noise;
- the 13-point `p=0.02,...,0.14` grid, four 2,500-trial shards per code-p and
  10,000 trials per code-p;
- the BpLSD algorithm, all decoder kwargs and `omp_thread_count=1`, with no
  fallback, section wrapper, redraw or code selection;
- the master seed, registry-bound seed derivation and the benchmark,
  measurement, replay and bootstrap namespaces;
- the failure scorer, complete independent replay, fail-closed aggregation,
  20,000-draw bootstrap family, crossing classifications and publication mask;
- the prohibition on asymptotic threshold, critical-exponent, FSS, `q_top`, MLD
  or preparation-channel claims.

Changing worker count or execution host cannot change a shard stream because
every measurement seed is derived only from the already frozen scientific
identity and shard key. Before formal execution, regression evidence must show
that every planned measurement seed under the remote profile equals the seed in
the original frozen config.

## Frozen remote execution profile

| Item | Frozen value |
|---|---|
| Entry route | `ssh yuany` through the project server entry |
| Compute node | exactly one node, `nd-3` |
| Process workers | exactly `64` for formal generation and full replay |
| Decoder threads | `omp_thread_count=1`; no nested OpenMP parallelism |
| Stage reserve | `2 * (generation + full replay + analysis + fixed overhead)` |
| Per-stage reserved core-hour cap | `1200` core-hours |
| Per-stage predicted wall cap | `24` hours |
| Projected peak RSS cap | `128` GiB |
| Remote Python environment | isolated prefix `/home/DATA1/users/yuany/.single_shot/cache/exp103_remote_v1_env` |
| Persistent root | `~/.single_shot/runs/` and `~/.single_shot/logs/` only |

No shard may be generated or replayed on another node. A failed remote gate does
not authorize moving the task to another host, changing the worker count, or
weakening a cap. Formal jobs run in `screen` with an explicit worker count.

## Environment identity and qualification gate

The build audit fixes the following remote environment identity before any
measurement-namespace trial:

| Identity | Frozen value |
|---|---|
| Runtime hostname / architecture | `nd-3` / `x86_64-linux-gnu` |
| Python executable | `/home/DATA1/users/yuany/.single_shot/cache/exp103_remote_v1_env/bin/python` |
| Python / NumPy / SciPy | `3.12.12` / `2.4.1` / `1.17.0` |
| ldpc | `2.4.1` |
| Official ldpc source commit | `d3429964cd4ffe1abfc041c6ec8b8425cb174f40` |
| Official source archive SHA256 | `76af0f01446ee7cbed33a47d6b597c10d8d12b2f10d508911b3d0763844d467e` |
| Linux BpLSD extension | `_bplsd_decoder.cpython-312-x86_64-linux-gnu.so` |
| Linux BpLSD SHA256 | `db3eb33b3afa4887994c9b949cdc7ae280614eab0fe4245a63226060740140e6` |

The PyPI 2.4.1 sdist omits `src_cpp/rng.hpp`; the isolated Linux extension was
built with no dependency resolution from the exact official GitHub release
commit above. The source archive, its `rng.hpp` SHA256, package versions and
direct BpLSD sanity decode are rechecked by the qualification gate. The shared
conda `11` environment and its `ldpc` 2.3.7 remain explicitly ineligible.
The frozen Linux solve could not provide the initially probed Numba 0.65.1;
`mamba` instead resolved Numba 0.66.0 with llvmlite 0.48.0, and those exact
support-package versions are bound by the canonical remote config and gate.

The canonical remote config binds these values together with the exp103 source
commit and package-tree SHA. Validation 003 subsequently records the config,
qualification, deployment manifest/archive and resource-preflight SHA values;
those evidence objects cannot be self-referential constants in this amendment.

`ldpc` 2.3.7, a missing compiled backend, a different decoder implementation,
or any unrecorded environment drift closes the gate. Qualification must run the
decoder identity/no-fallback tests and the contract/oracle regression suite in
the exact remote environment. Placeholder identity values can never authorize a
formal command.

Validation 003 uses only the existing benchmark namespace on the fixed
`m=3,5,8` and `p=.02,.08,.14` tasks. It records timing, RSS and infrastructure
identity, but neither saves nor inspects logical outcomes. Both Stage 1 and
Stage 2 must independently satisfy all three frozen remote caps. A failure is
reported honestly as `BLOCKED_REMOTE_RESOURCE_PREFLIGHT`; it is not repaired by
changing the scientific plan or selecting another benchmark.

## Verified deployment and immutable data

The active source remains the repository root, never a prior deployment or
validation snapshot. Before measurement, the amendment, remote config, source
and qualification evidence must be committed and pushed. A deployment manifest
must bind the pushed commit, source-tree SHA, contract and amendment bytes,
config SHA, registry SHA and every deployed file. The remote wrapper verifies
those bytes before execution.

Remote execution uses a clean source archive and disables bytecode creation with
`PYTHONDONTWRITEBYTECODE=1` and `python -B`. A pre-run or post-run
`__pycache__`/`.pyc` finding exits with code 67. Raw and replay evidence are
immutable and are never overwritten in place. Retrieved artifacts first enter a
staging directory, pass their complete SHA manifest, and only then move into the
canonical local exp103 raw/evidence paths.

## Revised validations and authority

1. Validation 003 freezes the single-node environment/deployment identity and
   applies the remote resource gate.
2. Validation 004 runs `m=3,4,5`, then performs complete independent replay and
   produces the restricted preliminary technical evidence.
3. Validation 005 runs `m=6,7,8` whenever Validation 004 is technically complete
   and its replay passes. This decision is unconditional on all Stage 1 curves.
4. Validation 006 loads all 48 codes through the publication loader and reports
   the finite-grid crossing classification and final checkpoint.

This amendment authorizes only exp103 remote qualification and, after a PASS,
the frozen exp103 formal stages. It does not alter exp102 evidence or status,
does not clear any exp102 blocker, and grants no exp102 remote, formal,
held-out, restricted or production authority.
