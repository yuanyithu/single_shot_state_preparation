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

At initial implementation time, 134 exp102 tests pass locally, including 57 new
PA/autopsy tests. One full D0/m06 parent task replayed all labels, swap/logical
counters, transport totals, and residuals bit-for-bit; it classified
`INCONCLUSIVE` because the final outbound conditional edges had fewer than 200
attempts. This diagnostic does not replace the frozen four-task autopsy.

The local pre-commit runtime projection passes all four budget checks, but it
is non-authoritative Darwin evidence. `orchestrate_pa.py` requires a matching
clean-source Linux runtime report for every PA stage and caps hard screen at
two hours; the remote Linux benchmark is still mandatory before launch.
