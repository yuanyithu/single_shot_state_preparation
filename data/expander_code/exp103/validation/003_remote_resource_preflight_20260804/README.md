# Validation 003: nd-3 environment and remote resource preflight

Status: `NOT_STARTED`.

This validation is the sole qualification and resource gate for the authorized
`exp103.remote_execution.v1` profile. It does not inherit PASS from Validation
001 and does not rewrite the blocked local result in Validation 002. No formal
measurement-namespace trial may run until this directory contains complete,
immutable PASS evidence committed and pushed from the active source tree.

## Frozen task and limits

- Compute on exactly one node: `nd-3`.
- Use exactly 64 process workers and `omp_thread_count=1`.
- Apply reserve multiplier 2 to generation, full replay, analysis and fixed
  overhead together.
- Require each stage to use at most 1200 reserved core-hours, at most 24
  predicted wall-hours and at most 128 GiB projected peak RSS.
- Benchmark only the frozen nine `(m3,m5,m8) x (.02,.08,.14)` tasks with the
  benchmark seed namespace. Do not retain or inspect logical outcomes.
- Keep the scientific protocol, panel, seeds, p grid, decoder, replay, statistics
  and crossing decision unchanged.

## Required evidence before PASS

1. Freeze the actual `nd-3` hostname, isolated conda environment
   `exp103_remote_v1_env` executable prefix, Python, NumPy, SciPy and ldpc
   versions, OS/architecture and available cores/RAM/disk. The shared conda
   `11` environment is ineligible.
2. Verify `ldpc==2.4.1`, direct `ldpc.BpLsdDecoder`, the exact Linux extension
   filename and SHA256, all frozen kwargs and absence of every fallback.
3. Run the exp103 contract/oracle, decoder identity, seed reproducibility,
   fail-closed and publication-loader regressions in that exact environment.
4. Prove all planned measurement seeds equal those from the original frozen
   config despite the change from 8 to 64 process workers.
5. Bind the pushed source commit, source-tree SHA, original contract, remote
   amendment, remote config, registry and deployment archive/file manifest.
6. Run the outcome-blind timing/RSS benchmark and evaluate Stage 1 and Stage 2
   separately against every frozen cap.
7. Verify the clean-source wrapper, `python -B`,
   `PYTHONDONTWRITEBYTECODE=1`, pre/post bytecode scan and exit-67 behavior.

## Unfrozen placeholders

The following values are unknown and are not evidence of readiness:

- runtime hostname: `TO_BE_FROZEN_BEFORE_FORMAL`;
- Python/NumPy/SciPy versions and Python prefix:
  `TO_BE_FROZEN_BEFORE_FORMAL`;
- ldpc identity: `TO_BE_VERIFIED_AS_2.4.1_BEFORE_FORMAL`;
- Linux BpLSD extension filename/SHA256:
  `TO_BE_FROZEN_BEFORE_FORMAL`;
- remote config SHA256, source commit/source-tree SHA256 and deployment
  manifest/archive SHA256: `TO_BE_FROZEN_BEFORE_FORMAL`.

Planned compact artifacts are an environment identity report, deployment
manifest and remote resource-preflight report. Their filenames and hashes must
be frozen before this README can record PASS. Missing evidence, an identity
mismatch, an invalid benchmark or any failed stage cap closes the formal gate;
it must not be repaired by changing node, worker count, code panel, seed, grid or
decoder.

Authority after PASS is limited to exp103 Validations 004 through 006 under the
remote amendment. It does not alter exp102 status or authorize any exp102
remote, formal, held-out, restricted or production work.
