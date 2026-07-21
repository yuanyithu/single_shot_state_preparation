# 006 — q=0 PA discovery and PT transport autopsy

This directory implements the frozen protocol in `../../PA_DISCOVERY_CONTRACT.md`.
It is discovery-only and cannot certify the formal PT-v1 freezer or production.

## Evidence phases

1. Local exact/reference/Numba and analyzer tests under conda environment `12`.
2. Clean-source package plus identical canonical digest on nd-1/nd-2/nd-3.
3. Frozen runtime benchmark on m6/m8, coordinate/block4. Hard screen starts only
   when the report status is `PASS`.
4. Four old PT cell-config tasks replayed into transport-autopsy raw.
5. Four-method, two-cell PA hard screen (64 populations).
6. Zero pass stops; one pass permits only B384-2 rescue; two or more base passes
   prohibit rescue. A consistent pair freezes confirmation and resolution
   manifests before either fresh result is opened.
7. Complete confirmation/resolution can emit only `READY_FOR_FORMAL`.

The ignored `local_development_*/*.npz` files are pre-commit diagnostics, not
clean-source evidence. `local_runtime_precommit.json` is likewise non-authoritative
and is not part of the certification record.

## Local commands

```bash
conda run --no-capture-output -n 12 pytest -q \
  data/expander_code/exp102/tests
conda run --no-capture-output -n 12 python -m \
  data.expander_code.exp102.validation.006_pa_discovery_20260721.cross_node_pa \
  data/expander_code/exp102/registry/registry.json SOURCE_COMMIT \
  --require-verified-source
conda run --no-capture-output -n 12 python -m \
  data.expander_code.exp102.validation.006_pa_discovery_20260721.benchmark_pa \
  data/expander_code/exp102/registry/registry.json \
  data/expander_code/exp102/config/q0_pa.discovery.v1.json \
  SOURCE_COMMIT RUNTIME_REPORT --require-verified-source
```

Build the source bundle only from a clean pushed commit with the existing
`002_numba_smoke_20260719/build_source_package.py`. Remote commands use conda
environment `11`, explicit frozen worker counts, `--no-capture-output`, and the
verified-source wrapper.

## Status

Final discovery status is `EXHAUSTED`, not `READY_FOR_FORMAL`.

- Worker source is `f0dff0f8d3e055227b75c999a73c751e2a576768`; archive SHA256 is
  `57811c43662b379524fb4f5099346f042d5577cc1e2c69a31299a11fd9c01324`.
- nd-1/2/3 produced canonical digest
  `f4ed9fff7512f8995a4f70c60072c1bba054aaf75e0440a4d00545880305f478`.
- The nd-2 Linux runtime report passed: m8 slowest `56.91 us/particle-sweep`,
  startup `1.80 s`, maximum population `0.373 min`, and factor-two complete
  schedule projection `1.064 min`.
- All four autopsy raws reproduced the parent labels, swap/logical counters,
  transport totals, and residuals bit-for-bit. All four are `INCONCLUSIVE`
  because required outbound conditional attempts fall below 200.
- All 64 hard-screen populations are complete and hash-verified. Every method
  fails both cells; all populations fail final genealogy, with median family
  ESS about 1 and median distinct initial families 1--2. B96 variants also
  contain some CESS/maximum-weight failures.
- The frozen zero-pass branch forbids B384-2 rescue. No confirmation,
  resolution, formal PA config, held-out freezer, or production plan exists.

The authoritative summaries are `hard_screen_report.json` and
`transport_autopsy_report.json`. The first deployment attempt is audit-only:
control generation wrote `__pycache__` into the shared source tree, and the
verified-source wrapper correctly rejected every node with exit 67 before raw
execution. Run `exp102_pa_discovery_20260721_f0dff0f_r2` used a fresh,
bit-identical archive and invoked all control/launch commands through the
verified wrapper.

Post-run local replay found an evidence portability issue, not a physics or
sampling change: NumPy 2.3.4/2.4.1 vector `exp` and reductions differ by a few
ULP. The analyzer now permits at most 8 ULP for ladder floats, 64 ULP for
non-cumulative derived floats, and `32*G` ULP for cumulative log-Z while
retaining exact discrete decisions, parents, ancestry, counters, identities,
and hashes. Full 64-task replay observed a worst-case 4096 ULP (`5.68e-14`
absolute) in cumulative log-Z.

Final local certification under conda environment `12` is 138 exp102 tests and
365 exp101 regressions passing (the latter with two expected deprecated-alias
warnings). Python compileall, every exp102 shell script under `bash -n`, report
cross-hash/count checks, and `git diff --check` also pass.
