# Validation 003: nd-3 environment and remote resource preflight

Status: `BLOCKED_REMOTE_RESOURCE_PREFLIGHT`.

The exact remote environment passed qualification, and the frozen benchmark
was outcome-blind and infrastructure-valid. Stage 1 passed all three resource
checks, but Stage 2 exceeded both the reserved core-hour and predicted wall-time
caps. The remote amendment requires both stages to pass before any formal
measurement, so no measurement-namespace shard was launched.

## Frozen execution identity

- Host: `nd-3`; one process pool with 64 workers and one decoder thread.
- Prefix: `/home/DATA1/users/yuany/.single_shot/cache/exp103_remote_v1_env`.
- Core packages: Python 3.12.12, NumPy 2.4.1, SciPy 1.17.0, ldpc 2.4.1.
- Linux BpLSD SHA256:
  `db3eb33b3afa4887994c9b949cdc7ae280614eab0fe4245a63226060740140e6`.
- Frozen source commit: `ff20f045399f86c4bbbe87fa14e261ac8517773c`.
- Source-tree SHA256:
  `b7b4692defb487a21bb63de1a11817894d9c90c2a9b4559880961e41a7287b54`.
- Canonical remote config payload SHA256:
  `3897c83d2ff33044f9d433889ef4b8dd54b007551e385871f1a8bf653c34e378`.
- Registry SHA256:
  `883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b`.

Qualification ran from deployment commit
`d76433d9fd10346eb71197f6f508be3e3e864b7a`, manifest SHA256
`7424357823fe7f28d5a72fa4ce8d15e9a3c3870c77f5f8811d5d588ab3debe26`,
and archive SHA256
`6f9fe86c27633aadc2e02ca8bdaa5f087cf834148eab9ae40b6d3673e5594d71`.
The committed qualification was then included byte-for-byte in the preflight
deployment at commit `1fa9b4c729b11bf2656e789fff9ec3db60464e59`, manifest
SHA256 `1d3df0c610ee147f07acb94a265bf0dc5e5409a1bd74b04d02d242723a67ec19`,
and archive SHA256
`e3e2bdce0771ca33580481c480fe55970d6be1a0eb932a0948a49a84d6aed87c`.

## Qualification result

`environment_qualification.json` has SHA256
`9502027567f59e2ad537d0aeb19c52ef2d8c28f11cf130e8406d1b465f070eef`.
The exact remote environment passed `128` exp103, `58` exp101, and `17`
exp102 tests (`203/203` total), with zero skipped, xfailed, xpassed, or
deselected tests. The report also revalidated the official ldpc source archive,
compiled extension, package versions, clean-source wrapper, host architecture,
cores, RAM, and disk identity.

## Outcome-blind resource result

`remote_resource_preflight.json` has SHA256
`0c5bca7d1ee599021b7e93389a23e18075508ae8f2c34186f3622340cc4734c7`.
It contains only timing, RSS, identity, and resource arithmetic from the frozen
nine `(m3,m5,m8) x (.02,.08,.14)` benchmark tasks. It records
`outcome_blind=true`, `logical_outcomes_saved=false`, and the benchmark seed
namespace; no logical result was retained or inspected.

| Stage | Reserved core-hours | Predicted wall-hours | Peak RSS GiB | Result |
|---|---:|---:|---:|---|
| Stage 1, m3-m5 | 1027.3979769 / 1200 | 9.0109217 / 24 | 19.5095215 / 128 | `PASS` |
| Stage 2, m6-m8 | 9520.3885108 / 1200 | 75.3624102 / 24 | 25.5681152 / 128 | `BLOCKED_REMOTE_RESOURCE_PREFLIGHT` |

The Stage 2 estimate includes 2378.6501703 core-hours for generation and
2380.5440851 core-hours for full replay before the frozen reserve multiplier.
Its high-p m8 anchors measured about 2.00--2.74 seconds per trial for both
generation and replay, which drives the failed totals.

## Authority

This validation grants no formal measurement authority. Stage 1 cannot use its
isolated PASS because the preregistered gate required both stages to pass before
launch. The failure does not authorize a host change, worker-count change, code
or p-point deletion, resampling, relaxed caps, or a partial curve. Validations
004--006 remain `NOT_STARTED`; exp102 remains `BLOCKED_BEFORE_REMOTE`.
