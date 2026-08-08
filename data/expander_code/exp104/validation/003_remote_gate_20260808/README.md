# Validation 003: nd-3 qualification and remote resource gate

Status: **`PASS`**. Opens formal measurement. No physical result.

## Environment

Run root `~/.single_shot/runs/exp104_ensemble_v1_002`, deployed as a verified
archive and executed only through `run_verified_source.sh`, whose bytecode gate
exits 67. Bytecode clean before and after.

exp104 reuses the environment exp103 built and qualified rather than creating its
own, so the compiled decoder is the same object rather than a recompilation of
the same source. Re-verified here on nd-3: Python 3.12.12, numpy 2.4.1, scipy
1.17.0, ldpc 2.4.1, every frozen support package version matching, and decoder
extension `_bposd_decoder.cpython-312-x86_64-linux-gnu.so` with SHA256
`3a5a7dc2c1ed015eb137ef5823d7e2d13c2d851fe895788adc3bded4e4d0c079` — the exact
binary that produced the published exp103 measurement. The frozen ldpc source
archive is re-checked against its own hash and its `rng.hpp` hash.

nd-3: 96 logical CPUs, 48 physical cores, 503 GiB, x86_64, glibc 2.39.

## Qualification

`environment_qualification.json`, SHA256 `76abf501...45a322df`: **`PASS`**, with
131 exp104 tests, 58 exp101 and 17 exp102, every group fully executed and nothing
skipped, xfailed, xpassed or deselected.

The determinism regression gate runs here, in the remote interpreter against the
remote build, as permanent discipline 15 requires. It is what makes the 10
percent replay policy admissible.

### The first attempt failed, and that is recorded

`environment_qualification_attempt_001_FAILED.json` is the first run, in run root
`exp104_ensemble_v1_001`: 129 of 131 exp104 tests. Two tests assumed the local and
remote config fixtures differ, which is false under qualification because
`EXP104_TEST_CONFIG_PATH` points the frozen-config fixture at the remote config.
Both were rewritten to name the file they need. The failed attempt is kept, its
run root is not reused, and the corrected run is a separate run root.

This is worth stating plainly: the gate found a real defect in the test suite that
the local suite could not, which is the reason the remote suite is run at all.

## Resource gate

`remote_resource_preflight.json`: **`PASS`**.

| quantity | projected | cap |
|---|---:|---:|
| reserved core hours | 270.9 | 900 |
| predicted wall hours | 4.09 | 16 |
| projected peak RSS | 24.3 GiB | 128 GiB |

778 tasks, 12,000 codes, 432,000 trials, 64 workers. Generation 120.6 core-hours,
committed replay 12.9.

Every projection is an upper bound: per-trial cost is the maximum over the
benchmarked grid points and each `m` is benchmarked directly rather than
extrapolated from an anchor. The `m = 8` per-trial upper bound measured here,
2.7557 s, agrees with the 2.7383 s exp103 measured on the same node with the same
decoder, which is an independent check that the two experiments are running the
same object.

The committed replay subsample is fixed in this report, before any production
task runs: 83 tasks, `2, 3, 6, 11, 21, 40` for `m = 3..8`, block 0 of every `m`
included unconditionally. Because one task spans the whole `p` grid, this covers
every `(m, p)` combination.

## Authority

Remote resource gate only. Opens the production scan under the frozen caps.
Publishes no physical result, and clears no exp102 blocker.
