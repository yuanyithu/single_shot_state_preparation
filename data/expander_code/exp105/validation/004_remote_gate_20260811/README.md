# Validation 004: nd-3 environment qualification and remote resource gate

Status: **`PASS`**. Opens formal measurement. No physical result.

## Environment

Run root `~/.single_shot/runs/exp105_noisy_v1_004`, deployed as a verified
archive and executed only through `run_verified_source.sh`, whose bytecode gate
exits 67. Bytecode clean before and after.

exp105 reuses the environment exp103 built and exp104 qualified rather than
creating its own, so the compiled decoder is the same object rather than a
recompilation of the same source. Re-verified here on nd-3: Python 3.12.12,
numpy 2.4.1, scipy 1.17.0, ldpc 2.4.1, and decoder extension
`_bposd_decoder.cpython-312-x86_64-linux-gnu.so` with SHA256
`3a5a7dc2c1ed015eb137ef5823d7e2d13c2d851fe895788adc3bded4e4d0c079` — the exact
binary that produced the published exp103 and exp104 measurements.

nd-3: 96 logical CPUs, 48 physical cores, 4 sockets, 503 GiB, x86_64, glibc 2.39.

## Qualification

`environment_qualification.json`, SHA256 `bb43a9fa...83596d56`: **`PASS`**.

| group | passed | expected | skipped |
|---|---:|---:|---:|
| exp105 | 166 | 166 | 0 |
| exp101 | 58 | 58 | 0 |
| exp104 | 131 | 131 | 0 |
| exp102 | 17 | 17 | 0 |

Nothing skipped, xfailed, xpassed or deselected. exp104's suite is part of this
qualification because exp105 reuses its ensemble rule, its decoder identity and
its comparison codes; a change on either side has to surface as a failure here
rather than as a silent difference in what exp105 measures.

The determinism regression runs here, in the remote interpreter against the
remote build, as permanent discipline 15 requires. It is what makes the ten
percent replay policy admissible.

### The first attempt failed, and that is recorded

`environment_qualification_attempt_001_FAILED.json` is the first run, in run root
`exp105_noisy_v1_001`: exp105 102/165 and exp104 117/131, with zero skips, so
these were failures rather than absences. It found three real defects that the
local suite could not:

1. exp104's suite was pointed at its own macmini config, so on nd-3 it checked
   macmini identities and failed 14 tests for a reason unrelated to the
   environment. The qualification now sets `EXP104_TEST_CONFIG_PATH` the same way
   it sets exp105's.
2. Five exp105 tests assumed `frozen_config` is the pilot config, which is only
   true locally; under qualification it resolves to the remote one. They now name
   `pilot_config` where they mean the pilot, and the phase-agnostic ones select
   `p` tokens by position rather than by value.
3. The production registry was gitignored and therefore absent from the deployed
   archive, which failed 58 tests at fixture setup. It is tracked now. Rebuilding
   it on nd-3 would have been cheaper in bytes but wrong in kind: the deployed
   tree is verified byte for byte against its archive, so a file created there is
   an unlisted file and trips the tree gate at exit 67.

A later attempt also failed with `KeyError: 'p_tokens'` in the remote resource
preflight: exp105 replaced exp104's hard-coded benchmark grid with a positional
selection because the production grid is not frozen until the pilot runs, and only
the local entry point was updated. The regression test for that now lives in
`tests/test_remote_execution.py`.

The failed run roots are not reused, and each corrected attempt is a separate run
root.

## Resource gate

`remote_resource_preflight.json`, SHA256 `627b6b13...b097b34`: **`PASS`**.

| quantity | projected | cap |
|---|---:|---:|
| reserved core hours | 644.7 | 800 |
| predicted wall hours | 7.01 | 14 |
| projected peak RSS | 25.4 GiB | 128 GiB |

3,314 tasks, 17,617 codes, 1,057,020 trials, 64 workers.

### The gate blocked first, and it was right to

The first resource preflight projected **5,367.8 reserved core-hours against a
cap of 800** and blocked. The cause was mine: Validation 003 evaluated the frozen
allocation rule with costs measured on the macmini, but the rule spends a budget
of core-hours on the machine that runs it, and that machine is nd-3.

| m | per-trial upper bound, macmini | per-trial upper bound, nd-3 | per-code setup, nd-3 |
|---|---:|---:|---:|
| 3 | 0.0087 s | 0.0968 s | 0.186 s |
| 4 | 0.0281 s | 0.3075 s | 0.420 s |
| 5 | 0.0720 s | 0.7496 s | 0.980 s |
| 6 | 0.1639 s | 1.5488 s | 2.050 s |
| 7 | 0.3255 s | 2.8761 s | 4.520 s |
| 8 | 0.6078 s | 4.8821 s | 7.722 s |

Applying the same rule, the same 290 core-hour budget, the same grid and the same
trial cap to nd-3's own numbers gives 17,617 codes rather than 167,005. Nothing
about the rule, the budget, the grid, the estimand or the decision changed. What
changed is the precision that budget buys: the predicted pointwise standard
deviation of `Delta38` is `0.0041` rather than `0.0015`, still 3.8 times tighter
than the 200-code pilot's `0.0159`, against a contrast that runs from `+0.064` to
`+0.541`.

Worth recording separately: per-trial cost varies by a factor of **1,135** across
the grid at `m = 8`, from `0.0043 s` at `p = 0.001` to `4.8821 s` at `p = 0.07`,
because belief propagation converges immediately at low `p` and exhausts
`max_iter` at high `p`. Taking the maximum over benchmarked points, as the
contract requires, is therefore a genuinely conservative bound here rather than a
formality.

## Committed replay subsample

Fixed in this report, before any production task ran, by a seed derived from the
master seed, the replay namespace, the registry hash and the frozen `q`: 337
tasks, `20, 17, 18, 17, 19, 246` for `m = 3..8`, with block 0 of every `m`
included unconditionally. Because one task spans the whole `p` grid, this covers
every `(m, p)` combination.

## Authority end

This opens formal measurement and nothing else. It certifies no physical result,
and it clears no exp102 blocker.
