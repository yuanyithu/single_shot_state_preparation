# exp106 raw evidence

Production raw NPZ files are written on nd-3 under
`~/.single_shot/runs/<run id>/raw/` and are **not tracked in Git**. There is one
file per task, named `m{m:02d}__b{block:04d}.npz`, and each is immutable: writing
over an existing task is an error, and a failed task is never rerun in place.

The compiled decoder is not bit-portable across platforms (exp104 Validation
002), so generation, replay and aggregation all happen on nd-3 against the pinned
nd-3 binary. Nothing here is ever mixed with a macmini artifact.

What comes back to this repository is the aggregate, the gate reports and the
replay report -- all of which the loader re-derives independently on macmini from
the stored per-code counts.

`pilot_v1/` holds the locating pilot's raw, which runs locally. Pilot raw is
never merged into production and never enters a published statistic.

**No filename anywhere in exp105 or exp106 encodes `q`.** The two experiments'
raw files, configs and aggregates are name-identical and are separated only by
`experiment_id` and `config_sha256` inside. Distinct run-root names are what keeps
them apart on nd-3.
