# 064 q=0 HP64 evidence and resource calibration

Status: `IMPLEMENTATION COMPLETE / CALIBRATION NOT EXECUTED / NO GENERATED REPORTS / NO REMOTE AUTHORITY`.

This validation implements the Stage 1 evidence/resource slice requested by
`reviews/EXP102_NEXT_STEP_PLAN_20260727.md`.  It keeps two questions separate:

- `discrepancy_audit.json` independently recomputes the validation 013 HP64/MAM
  headline values from raw logical labels and frozen character masks;
- `resource_calibration_report.json` uses only timing/identity inputs and emits
  every predeclared resource scenario without selecting one.

The validation 013 heavy raw are intentionally not copied into this clean
worktree.  `run_resource_calibration.py` first constructs the resource payload
without opening logical-label arrays, validates raw timing scalars, and only
then performs the separately implemented character audit with
`allow_pickle=False`.  The calibration command has not been run and no final
package has been generated.  When execution is authorized, run from this
directory with the original validation 013 run root:

Before the command can start, all implementation files and this config must be
tracked in one commit, the entire Git worktree must be clean, every SHA in
`resource_model_config.json:authority.implementation_files` must match, and no
`__pycache__`, `.pyc`, or expected output may exist.  The receipt records the
calibration source commit/tree separately from the historical validation 013
source identity.  The current untracked implementation therefore cannot yet
be executed as a calibration.

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python run_resource_calibration.py \
  --validation013-run /absolute/path/to/013/.../exp102_q0_hgp_screen_v2_20260722_4d134ee
```

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python -m pytest -q -p no:cacheprovider test_resource_calibration.py
```

After a future calibration run, verify its serialized package through the
independent implementation, which does not import the generator.  It writes a
one-shot source-bound audit artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python audit_resource_calibration.py --output-dir .
```

Artifacts that a future successful execution will generate are:

- `discrepancy_audit.json`;
- `timing_coverage.csv`;
- `resource_scenarios.csv`;
- `resource_calibration_report.json`;
- `RESOURCE_CALIBRATION_REPORT.md`; and
- `RUN_RECEIPT.json`.

The independent verifier subsequently creates:

- `independent_package_audit.json`.

Both stages are one-shot.  The runner refuses to start if any of the seven
expected paths already exists, and the verifier refuses to replace an existing
independent audit.  Preserve partial or failed packages for diagnosis; never
delete an output merely to rerun the same source/config identity.

Read `PRE_RUN_RED_TEAM.md` before using any number.  In particular, the full
grid has incomplete empirical coverage, all full-grid totals are scenario
arithmetic rather than certified estimates, and no row authorizes an
experiment launch.

At the current status none of the seven artifacts above exists.  The presence of
the implementation and passing focused unit tests is not a completed resource
calibration and does not authorize the m3 easy block, a remote launch, or any
formal/held-out/production stage.
