# exp104 raw evidence

Production raw NPZ files are written on nd-3 under
`~/.single_shot/runs/<run id>/raw/` and are not tracked in Git.

One file is one task: a contiguous block of codes at one `m`, covering the whole
`p` grid, with four trials per code and `p`. Each file carries the frozen config,
registry, source-tree and decoder-binary identities, the per-code graph seeds and
logical-frame hashes, every per-trial outcome, and three stream digests.

Raw evidence is immutable. A task that did not complete is stored as `INVALID`
and is never rerun in place.
