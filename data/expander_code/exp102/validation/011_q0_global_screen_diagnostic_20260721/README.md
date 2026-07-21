# exp102 q=0 global sampler diagnostic screen

Status: `UNRESOLVED_NO_HARD_COSET_PASS` for completed immutable run
`exp102_q0_screen_diagnostic_20260721_342dd5b`. The run completed normally;
this is a frozen sampler-screen result, not an infrastructure failure.

## Verified remediation run

- Source commit: `342dd5bc0fb2c7694dbc58a8d0f2d92689c24991`.
- Archive SHA256: `4a54ba28f3ee2add94e93dd052e4bda567d5e008691f84a098c21768b4fe11f3`.
- Source-manifest SHA256: `2b8ab6d238d6319ea73c4c5da0ecf815a3d2e2ea28932dddc30bd40afe158b01`
  with 854 files.
- Schedule file/identity SHA256:
  `f9aeccd95640a56fabe813796d0e1ce388cffa1bcccf2405a6bafcd913520832` /
  `cd09b4701d54b061f59db5ce50df191edac0b23da62e411ccc5a597400426cb9`.
- Three-node canonical digest:
  `080b3170ca168dc3f237d22a4d18403eb2c0b0b2455e6d1e3ca876aae39c86a9`.
- Portable 4096-value gamma SHA256:
  `a2c459ec9438e23f863c44528ac093c5b93d891b6a8bec0278b873fe47f2459a`.
- Selected resource tier: T3 (`burn=8192`, `measurement=32768`).
- Completed raw: 15/15 bias and 1280/1280 measurement trajectories, with
  `reused=0`, seven exclusive node SUCCESS markers, and complete raw replay.
- Terminal package file/identity SHA256:
  `83155d17e54fa2597ba8bce48ac99a8667a3dfc4296589a93e89dcc0cfd5cae7` /
  `0e0fb2f950eb609c984b29f5647321694c82f8f7a6810609fd1742d1472a990a`.

## Screen outcome

| mechanism | methods | passed cells | frozen failure |
|---|---|---:|---|
| hard coset | `RC8-QC1/QC4/J08/J12/J16` | each `0/5` | P/U initialization, D2, and family gates |
| defect trace | `DT16/DT32/DT64` | each `0/5` | no fixed-clock D=0 observations |

All 25 hard-coset method/cell summaries fail P/U `q_top` and D2 consistency;
23/25 also fail normalized-weight consistency. Every U family has `Rhat>1.05`
and bulk ESS below 400. The observed P/U `|delta q_top|` spans
`0.06695..0.991999`, versus the frozen 0.04 limit. QC4 is closest on every
cell but still fails, including `delta q_top=0.91815` on `m08_c06,p=.04`.

All 480 defect-trace measurement chains have `d0_count=0` and zero complete
leave-return excursions. Thus no D=0 conditional estimate or D0 ESS exists;
342/480 chains also exceed the 0.10 Dmax-boundary occupancy gate. All 75
cross-mechanism comparisons are consequently invalid. The prescribed decision
order names the terminal state `UNRESOLVED_NO_HARD_COSET_PASS`, although neither
mechanism produced a passing method. This means unresolved within these
algorithms and the T3 budget; it does not mean `IMPOSSIBLE` or establish a
formal physical failure at any parameter point.

## Independent evidence audit

The complete 485 MiB raw tree is retained locally and in the remote `runs/`
backup but is not committed. `completed_run_evidence/` contains controls,
preflight/stage metadata, all four node raw manifests, a 1295-entry raw SHA
manifest, the driver log, original terminal artifacts, an independent local
verified-archive replay, and a fail-closed verifier. The replay again validates
every raw bit-for-bit and reproduces every gate and terminal field. Its report
differs only in 62 derived `core_seconds` and 18 derived ESS values by at most
4 ULP (maximum absolute difference `1.82e-12`); the corresponding self-hashes
are independently valid. Run:

```bash
conda run -n 12 --no-capture-output python \
  data/expander_code/exp102/validation/011_q0_global_screen_diagnostic_20260721/completed_run_evidence/verify_evidence.py
```

The completed evidence closure manifest has SHA256
`7e01c730a13cd0b20df2080aacf25d46a6b2fad42350a0398c3130d0ffe93c96`.

The terminal decision has `selected_pair=null`, `formal_authorization=false`,
and `production_authorization=false`. Its remaining blockers are
`NO_T_VS_2T`, `NO_FRESH_HARD2_CONFIRMATION`,
`NO_CONF17_RES6_GAP8_SMALL6`, `NO_TI_OR_REVIEWED_INDEPENDENT_ORACLE`, and
`NO_HELD_OUT`. This workflow can never create formal-readiness, held-out, or
production authority.

## Predecessor conflict

The first immutable run `exp102_q0_screen_diagnostic_20260721_5e1f5aa`
remains closed as `CONFLICT_CROSS_NODE_GAMMA_LIBM`: it completed 15 bias raws
but stopped before all 1280 measurement trajectories. None of those raws was
reused. Its metadata-only audit remains under `failed_run_evidence/`. The fresh
run above uses exact `3/5` Decimal fifth-root arithmetic and digest v2; platform
`libm` fractional power must not be restored.

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
