# exp105 raw evidence

Production raw NPZ files are written on nd-3 under
`~/.single_shot/runs/<run id>/raw/` and are not tracked in Git.

One file is one task: a contiguous block of codes at one `m`, covering the whole
`p` grid at the frozen `q = 0.05`, with six trials per code and `p`. Each file
carries the frozen config, registry, source-tree and decoder-binary identities,
the per-code graph seeds, logical-frame and observable-frame hashes, every
per-trial outcome, and four stream digests: data error, readout error,
correction and logical label.

Raw evidence is immutable. A task that did not complete is stored as `INVALID`
and is never rerun in place.

`pilot_v1/` holds the locating pilot's 44 tasks. It is planning evidence, not
physics: it is never merged into production and never enters a published
statistic. It is also not tracked in Git.
