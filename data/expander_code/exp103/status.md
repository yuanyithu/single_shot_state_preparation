# exp103 status

## Current state

**`BLOCKED_REMOTE_RESOURCE_PREFLIGHT`**

exp103 remains a BpLSD decoder-MC line for `q=0` block logical failure. The
exact `nd-3` environment and deployment passed qualification, but the frozen
outcome-blind remote resource gate failed for Stage 2. Because Validation 003
requires both stage gates to pass before any measurement-namespace trial,
Validations 004--006 remain closed. No formal shard, crossing bracket,
asymptotic threshold, or physical result exists. exp102 remains
`BLOCKED_BEFORE_REMOTE`; exp103 has cleared none of its blockers.

## Current gates

1. Validation 001 remains `PASS`; Validation 002 remains the immutable
   `BLOCKED_LOCAL_RESOURCE_PREFLIGHT` result.
2. Remote qualification passed on exactly `nd-3`: Python 3.12.12, NumPy 2.4.1,
   SciPy 1.17.0, ldpc 2.4.1, and all `203/203` frozen tests passed with no skip,
   xfail, xpass, or deselection.
3. The Linux BpLSD extension SHA256 is
   `db3eb33b3afa4887994c9b949cdc7ae280614eab0fe4245a63226060740140e6`;
   canonical remote config payload SHA256 is
   `3897c83d2ff33044f9d433889ef4b8dd54b007551e385871f1a8bf653c34e378`.
4. Remote Stage 1 passed its caps: `1027.3980 <= 1200` reserved core-hours,
   `9.0109 <= 24` wall-hours, and `19.5095 <= 128` GiB peak RSS.
5. Remote Stage 2 failed two caps: `9520.3885 > 1200` reserved core-hours and
   `75.3624 > 24` wall-hours; projected RSS `25.5681 <= 128` GiB passed.

The remote amendment forbids using the Stage 1 PASS alone, changing host or
worker count, weakening a cap, dropping codes, changing the p grid, or starting
a partial formal scan. A new experiment contract and explicit user authority
would be required to pursue a materially different execution design.

## Evidence map

- `EXPERIMENT_CONTRACT.md`: frozen scientific and statistical contract.
- `REMOTE_EXECUTION_AMENDMENT.md`: authorized single-node profile and gates.
- `config/decoder_mc.v1.json`: original scientific seed/grid/panel identity.
- `config/decoder_mc.remote.v1.json`: exact qualified remote identity.
- `validation/003_remote_resource_preflight_20260804/`: qualification and
  outcome-blind remote resource evidence.
- `validation/INDEX.md`: numbered evidence ledger.
- `raw/`: contains no formal exp103 shard.

## Latest evidence

- Validation 003: `BLOCKED_REMOTE_RESOURCE_PREFLIGHT`; Stage 1 PASS, Stage 2
  core-hour and wall-time checks FAIL; no logical outcomes were saved.
- Environment qualification: `PASS`, exact counts `128 + 58 + 17 = 203`.
- Validation 002: `BLOCKED_LOCAL_RESOURCE_PREFLIGHT`; its original evidence is
  unchanged.
