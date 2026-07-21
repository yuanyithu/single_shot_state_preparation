# exp102 q=0 global sampler diagnostic screen

Status: `CONFLICT_CROSS_NODE_GAMMA_LIBM` for immutable run
`exp102_q0_screen_diagnostic_20260721_5e1f5aa`; portable remediation awaits a
fresh remote run. The old run passed three-node preflight and produced all 15
bias raws, but measurement-control replay found two platform-dependent gamma
bit patterns and stopped. Therefore all 1280 measurement trajectories, report,
decision, and terminal package are absent. This is not a sampler-convergence or
physics result, and the old run must never continue.

The repaired implementation interprets `.6` as exact `3/5`, generates the
4096-value schedule with fixed-precision Decimal fifth-root arithmetic, freezes
versioned gamma SHA
`a2c459ec9438e23f863c44528ac093c5b93d891b6a8bec0278b873fe47f2459a`,
and binds it into digest v2. The metadata-only old-run audit is under
`failed_run_evidence/`; the 15 NPZ files remain ignored and retained locally and
on the server. A new commit, archive, run ID, and schedule are mandatory.

This workflow tests whether any hard-coset sampler and any defect-trace sampler
mix on `HARD2+EASY3`. Its strongest possible result is
`DIAGNOSTIC_SCREEN_PAIR_FOUND`; it cannot create formal-readiness, held-out, or
production authority.

## Frozen scope

- Contract: `exp102.q0_global.screen_diagnostic.v1`.
- Candidates: `RC8-QC1/QC4`, `RC8-J08/J12/J16`, and `DT16/DT32/DT64`.
- Panel: two HARD2 plus three EASY3 cells.
- Bias tasks: `3 * 5 = 15`.
- Measurement tasks: `8 * 5 * 2 * 16 = 1280`.
- Preflight: verified-source exp102+exp101 tests, canonical digest, and
  sampler-only runtime on `nd-1/nd-2/nd-3`. Full-sector TI and WMC are excluded.
- Execution: fixed `nd-1,nd-3`; the largest three-node-worst-case passing tier
  is mandatory.
- Schedule: preflight hour 8, bias hour 12, measurement hour 22, analysis hour
  24, as frozen by the diagnostic config.

## Files

- `common.py`: versioned adapter, schedule/runtime validation, ownership,
  verified-source launch, marker handling, and complete remote-evidence replay.
- `benchmark_screen.py`: sampler-only timing and worst-node tier consensus.
- `cross_node_screen.py`: reference/Numba and three-node canonical digest.
- `run_screen_preflight_node.py` / `orchestrate_screen_preflight.py`: immutable
  three-node preflight.
- `prepare_screen_schedule.py`: freeze the source/archive-bound 24-hour
  schedule from a normal CLI (no heredoc).
- `prepare_screen_controls.py`: materialize the 15-task bias control, then the
  bias-bound 1280-task measurement control.
- `run_screen_stage.py` / `orchestrate_screen_stage.py`: execute a control with
  fixed ownership on `nd-1,nd-3` through `run_verified_source.sh`.
- `analyze_screen.py`: verify both controls, both ownership files, all markers
  and node raw manifests, source archive, schedule, runtime, and digest before
  raw replay and terminal packaging.
- `run_screen_wrapper.sh`: exclusive markers; either RUNNING or FAILED forces a
  fresh deployment and cannot be cleared by rerunning the wrapper.

## Pipeline adapter

`common.py` accepts the following canonical API from
`exp102_pipeline.screen_diagnostic`: `load_screen_diagnostic_config`,
`build_bias_manifest`, `build_measurement_manifest`,
`validate_screen_control_manifest`, `run_screen_hard_task`,
`run_screen_bias_task`, `run_screen_defect_task`, the three raw validators,
`analyze_screen_measurement_stage`, `freeze_screen_schedule`, and
`validate_screen_schedule`. Compatibility aliases are deliberately limited to
the corresponding diagnostic names; no PT/PA/global-discovery loader or raw
version is accepted.

## Run order

After a clean committed source archive is deployed under
`~/.single_shot/repos/<run-id>/`, freeze the schedule with the pipeline API and
use these modules in order:

1. `orchestrate_screen_preflight`
2. `prepare_screen_controls bias`
3. `orchestrate_screen_stage` with the bias control
4. `prepare_screen_controls measurement` with the verified bias tuple
5. `orchestrate_screen_stage` with the measurement control
6. `analyze_screen` with both control/ownership tuples

Every stage must use a fresh run ID after any FAILED marker. The analyzer's
verified terminal package is the only final diagnostic artifact; a worker
SUCCESS marker only means that worker completed its assigned code path.

Build and deploy only from a clean committed local worktree. The remote repo
and run paths must both be absent before the one-time copy:

```bash
REPO='/Users/jarvis/Desktop/sync/project D'
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=all)"
COMMIT=$(git rev-parse HEAD)
RUN_ID="exp102_q0_screen_diagnostic_$(date +%Y%m%d)_${COMMIT:0:8}"
LOCAL_DEPLOY="$REPO/data/expander_code/exp102/deployment/$RUN_ID"
conda run -n 12 --no-capture-output python \
  data/expander_code/exp102/validation/002_numba_smoke_20260719/build_source_package.py \
  "$REPO" "$COMMIT" "$LOCAL_DEPLOY"
ARCHIVE_SHA=$(tr -d '\r\n' < "$LOCAL_DEPLOY/ARCHIVE_SHA256")
MANIFEST_SHA=$(shasum -a 256 "$LOCAL_DEPLOY/SOURCE_MANIFEST.json" | awk '{print $1}')
REMOTE_HOME=$(ssh yuany 'printf %s "$HOME"')
REMOTE_DEPLOY="$REMOTE_HOME/.single_shot/repos/$RUN_ID"
REMOTE_RUN="$REMOTE_HOME/.single_shot/runs/$RUN_ID"
ssh yuany "test ! -e '$REMOTE_DEPLOY' && test ! -e '$REMOTE_RUN' && mkdir -p '$REMOTE_DEPLOY'"
rsync -a "$LOCAL_DEPLOY/" "yuany:$REMOTE_DEPLOY/"
ssh yuany "printf '%s  %s\n' '$ARCHIVE_SHA' '$REMOTE_DEPLOY/SOURCE.tar' | sha256sum -c - && \
  printf '%s  %s\n' '$MANIFEST_SHA' '$REMOTE_DEPLOY/SOURCE_MANIFEST.json' | sha256sum -c - && \
  test \"\$(tr -d '\r\n' < '$REMOTE_DEPLOY/SOURCE_COMMIT')\" = '$COMMIT'"
```

The complete 24-hour chain must run from a persistent nd-0 driver outside the
verified `source/` tree, for example with `nohup setsid`; an interactive SSH
disconnect must not kill an orchestrator that already owns immutable markers.
Run the final 32-worker analyzer on a compute node, not nd-0.

Minimal CLI chain (run orchestration on nd-0 with remote conda environment
`11`; paths shown as shell placeholders). Every Python entry point, including
schedule/control generation and both orchestrators, must be entered through
`run_verified_source.sh`. Never run plain Python in the clean `source/` tree:
it can create `__pycache__` and make all later verified launches fail closed.
The final analyzer additionally rejects a Git worktree or any archive identity
other than the one bound into the schedule:

```bash
MOD=data.expander_code.exp102.validation.011_q0_global_screen_diagnostic_20260721
REG=data/expander_code/exp102/registry/registry.json
CFG=data/expander_code/exp102/config/q0_global.screen_diagnostic.v1.json
RUN_ID=replace_with_fresh_run_id
RUN_ROOT="$HOME/.single_shot/runs/$RUN_ID"
DEPLOY="$HOME/.single_shot/repos/$RUN_ID"
COMMIT=replace_with_full_40_hex_commit
ARCHIVE_SHA=replace_with_archive_sha256
MANIFEST_SHA=replace_with_manifest_sha256
SCHEDULE="$RUN_ROOT/control/SCREEN_DIAGNOSTIC_24H_SCHEDULE.json"
RUNTIME="$RUN_ROOT/control/screen_runtime_consensus.json"
DIGEST="$RUN_ROOT/control/screen_digest_consensus.json"
PREFLIGHT="$RUN_ROOT/control/screen_preflight_report.json"
BIAS_CONTROL="$RUN_ROOT/control/screen_bias_input.json"
MEASUREMENT_CONTROL="$RUN_ROOT/control/screen_measurement_input.json"
VERIFY_REL=data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh

run_verified() {
  printf '%s  %s\n' "$ARCHIVE_SHA" "$DEPLOY/SOURCE.tar" \
    | sha256sum -c - >/dev/null
  tar -xOf "$DEPLOY/SOURCE.tar" "$VERIFY_REL" | bash -s -- \
    "$DEPLOY" "$COMMIT" "$ARCHIVE_SHA" "$MANIFEST_SHA" \
    conda run -n 11 --no-capture-output "$@"
}

run_verified_on_node() {
  local node=$1 remote_command
  shift
  printf -v remote_command '%q ' bash -s -- \
    "$DEPLOY" "$COMMIT" "$ARCHIVE_SHA" "$MANIFEST_SHA" "$@"
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
    "$node" "$remote_command" <<'REMOTE_VERIFY'
set -euo pipefail
deployment_root=$1
expected_commit=$2
expected_archive_sha256=$3
expected_manifest_sha256=$4
shift 4
verify_rel=data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh
printf '%s  %s\n' "$expected_archive_sha256" "$deployment_root/SOURCE.tar" \
  | sha256sum -c - >/dev/null
tar -xOf "$deployment_root/SOURCE.tar" "$verify_rel" | bash -s -- \
  "$deployment_root" "$expected_commit" "$expected_archive_sha256" \
  "$expected_manifest_sha256" conda run -n 11 --no-capture-output "$@"
REMOTE_VERIFY
}

run_verified python -m $MOD.prepare_screen_schedule \
  --registry "$REG" --config "$CFG" --source-commit "$COMMIT" \
  --archive-sha256 "$ARCHIVE_SHA" --manifest-sha256 "$MANIFEST_SHA" \
  --output "$SCHEDULE"
SCHEDULE_SHA=$(sha256sum "$SCHEDULE" | awk '{print $1}')

run_verified python -m $MOD.orchestrate_screen_preflight \
  --run-id "$RUN_ID" --source-commit "$COMMIT" \
  --archive-sha256 "$ARCHIVE_SHA" --manifest-sha256 "$MANIFEST_SHA" \
  --schedule "$SCHEDULE" --schedule-file-sha256 "$SCHEDULE_SHA"

run_verified python -m $MOD.prepare_screen_controls bias \
  --registry "$REG" --config "$CFG" --source-commit "$COMMIT" \
  --schedule "$SCHEDULE" --runtime-report "$RUNTIME" --output "$BIAS_CONTROL"
BIAS_SHA=$(sha256sum "$BIAS_CONTROL" | awk '{print $1}')
BIAS_OWNERSHIP="$RUN_ROOT/control/screen_ownership_${BIAS_SHA:0:12}.json"
run_verified python -m $MOD.orchestrate_screen_stage \
  --run-id "$RUN_ID" --source-commit "$COMMIT" \
  --archive-sha256 "$ARCHIVE_SHA" --manifest-sha256 "$MANIFEST_SHA" \
  --schedule "$SCHEDULE" --schedule-file-sha256 "$SCHEDULE_SHA" \
  --preflight-report "$PREFLIGHT" --runtime-report "$RUNTIME" \
  --control "$BIAS_CONTROL"

run_verified python -m $MOD.prepare_screen_controls measurement \
  --registry "$REG" --config "$CFG" --source-commit "$COMMIT" \
  --schedule "$SCHEDULE" --runtime-report "$RUNTIME" --run-root "$RUN_ROOT" \
  --raw-root "$RUN_ROOT/screen_diagnostic/raw" --deployment-root "$DEPLOY" \
  --bias-control "$BIAS_CONTROL" --bias-ownership "$BIAS_OWNERSHIP" \
  --output "$MEASUREMENT_CONTROL"
MEASUREMENT_SHA=$(sha256sum "$MEASUREMENT_CONTROL" | awk '{print $1}')
MEASUREMENT_OWNERSHIP="$RUN_ROOT/control/screen_ownership_${MEASUREMENT_SHA:0:12}.json"
run_verified python -m $MOD.orchestrate_screen_stage \
  --run-id "$RUN_ID" --source-commit "$COMMIT" \
  --archive-sha256 "$ARCHIVE_SHA" --manifest-sha256 "$MANIFEST_SHA" \
  --schedule "$SCHEDULE" --schedule-file-sha256 "$SCHEDULE_SHA" \
  --preflight-report "$PREFLIGHT" --runtime-report "$RUNTIME" \
  --control "$MEASUREMENT_CONTROL"

run_verified_on_node nd-3 python -m $MOD.analyze_screen --run-root "$RUN_ROOT" \
  --raw-root "$RUN_ROOT/screen_diagnostic/raw" --deployment-root "$DEPLOY" \
  --schedule "$SCHEDULE" --runtime-report "$RUNTIME" --digest-report "$DIGEST" \
  --preflight-report "$PREFLIGHT" --bias-control "$BIAS_CONTROL" \
  --bias-ownership "$BIAS_OWNERSHIP" \
  --measurement-control "$MEASUREMENT_CONTROL" \
  --measurement-ownership "$MEASUREMENT_OWNERSHIP" --registry "$REG" \
  --config "$CFG" --report-output "$RUN_ROOT/control/screen_report.json" \
  --decision-output "$RUN_ROOT/control/screen_decision.json" \
  --package-output "$RUN_ROOT/control/screen_terminal_package.json" \
  --num-workers 32
```
