# Validation 002: exp104 is the exp103 code path

Status: **`PASS`**. Authorizes Validation 003. No remote transfer, no production
compute, no physical result.

## The gated claim

Because exp104 replays only a committed ten percent of its own tasks, the
pipeline has to be tied to one that is already published. This validation runs
exp103's frozen 48-code registry, exp103's frozen seeds and the frozen decoder
identity through **both** packages on one machine and requires the per-trial
arrays to be bit-identical.

Six cells, three sizes, two classical distances, four grid points, **60,000
trials**. Every one of `failure_flags`, `logical_labels`, `syndrome_match`,
`bp_converged` and `bp_iterations` agreed element for element, with zero
mismatched trials.

The two packages are the same function of the same inputs. exp104's model
construction, decoder factory, error stream and scorer carry no drift from the
implementation that produced the published exp103 result.

## What the same run shows about the decoder, recorded but not gated

Neither package reproduces the counts exp103 published, because those were
produced on nd-3 and this comparison ran on macmini:

| cell | this machine | published from nd-3 | delta | relative | BP convergence here / there |
|---|---:|---:|---:|---:|---|
| m03_c00 p=0.02 | 409 | 398 | +11 | 2.8% | 0.9705 / 0.9705 |
| m03_c00 p=0.06 | 3992 | 3975 | +17 | 0.43% | 0.5140 / 0.5149 |
| m03_c03 p=0.04 | 1088 | 1091 | -3 | 0.27% | 0.8086 / 0.8085 |
| m04_c00 p=0.02 | 453 | 451 | +2 | 0.44% | 0.9814 / 0.9816 |
| m04_c05 p=0.03 | 720 | 727 | -7 | 0.96% | 0.9137 / 0.9133 |
| m05_c00 p=0.02 | 86 | 88 | -2 | 2.3% | 0.9893 / 0.9892 |

The differences are small, unsigned, and confined to which legal correction the
post-processor returns: belief-propagation convergence rates agree to within
0.001 everywhere, so the iterative stage is making the same decisions and the
divergence appears in ordered-statistics post-processing on the trials where BP
fails. The compiled extension is a different binary on the two platforms.

**This is not a defect and it is not gated here.** The operating contract is
explicit that different builds are not required to reproduce each other verbatim,
and exp104 never mixes artifacts across platforms: it will generate, replay and
aggregate entirely on nd-3, against the nd-3 decoder-binary SHA256 that its
remote config pins. The consequence for method is that a cross-platform
bit-exactness gate would be unsatisfiable by construction, exactly as exp103's
replay gate was unsatisfiable while its decoder was randomized.

The same-platform version of this check — running the exp104 code path against
the frozen exp103 raw shards that still live on nd-3 — belongs to Validation 003,
where the source is deployed and the binary matches.

## Evidence

- `cross_validate_exp103.py`: the comparison, rerunnable from the repository
  root. It is deliberately built from exp103's own `run_decoder_shard` on one
  side, not a reimplementation of it.
- `cross_validation.json`: per-cell results, both counts, the platform delta and
  the identity hashes of both configs, the exp103 registry and the exp103
  published diagnostics file.

## Authority

Implementation-equality gate only. Authorizes Validation 003. Publishes no
physical result, asserts nothing about any threshold, and clears no exp102
blocker.
