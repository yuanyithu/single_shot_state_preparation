# Experiment Data Management

Updated: 2026-05-24

This repository keeps experiment code, README files, and durable text notes in Git. Generated experiment data are local artifacts and must not be uploaded to GitHub. The current policy is conservative local cleanup plus Git index cleanup only; do not rewrite Git history unless a separate repository slimming task explicitly calls for `git filter-repo` or BFG.

## Git Policy

- Do not track generated data: `*.npz`, `*.npy`, `*.json`, `*.csv`, `*.tsv`, `*.log`, `*.png`, `*.pdf`, `chunks/`, `remote_run/`, `remote_runs/`, `pooled/`, or `analysis/`.
- Do not track data-local launcher/preflight files such as `data/**/*.sh`, `data/**/*.txt`, or nested `data/**/.gitignore`; keep durable notes in Markdown instead.
- Do not track local profiling/optimization output directories under `data/3d_toric_code/with_measurement_noise/profile_*` or `prod_*`.
- Keep only documentation that describes retained data or reproducibility decisions.
- Do not use `git add .` for data governance commits.

## Current Retention

- Paper-critical no-measurement-noise references remain local:
  - `data/3d_toric_code/without_measurement_noise/exp04_q0_crossing_window_scout/`
  - `data/3d_toric_code/without_measurement_noise/exp10_q0_oneday_deep_relaunch/`
- With-measurement-noise corrected-observable data are reduced to one local `q_top` archive:
  - `data/3d_toric_code/with_measurement_noise/exp34_fixed_p050_q000_080_L34567_corrected_observable_20260524_final_stopped_after_L6q060_nd12/qtop_samples/exp34_corrected_observable_qtop_samples_only.npz`
- The local `q_top` archive is intentionally ignored by Git. It stores per-parameter-point metadata, per-disorder `q_top` samples, and per-chain `q_top` samples for diagnostics.

## Removed Or Ignored

- `with_measurement_noise/exp24*` and all earlier with-measurement-noise runs were removed because they were insufficient for threshold decisions.
- Old pre-corrected-observable with-measurement-noise runs after `exp24`, node-split `nd1`/`nd2`/`nd3` outputs, pooled raw NPZs, plots, JSON summaries, and logs were removed when possible.
- Profiling/optimization directories are no longer tracked. Some local files remain because they are owned by `root` or uid/gid `2004`; local deletion needs appropriate filesystem ownership or admin privileges.

## Operational Notes

- Old `q != 0` numerical results before the corrected observable path are historical diagnostics only and must not be used as paper physics conclusions.
- For plotting the corrected-observable with-measurement-noise fixed-`p=0.0500` scan, use the `qtop_samples` archive rather than node-specific raw results.
- After the 2026-05-24 cleanup, `data/` was reduced from about `3.2G` to about `111M`; about `13M` is permission-blocked profiling/optimization residue.
