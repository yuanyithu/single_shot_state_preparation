# Validation 002: local implementation gate

Status: **`PASS`**. Controlled category: `IMPLEMENTATION_GATE`.

Local only, on macmini. Authorizes the locating pilot. Authorizes no remote
transfer, no production compute and no exp102 stage.

## What was run

Five groups, each in its own pytest process -- the same grouping nd-3
qualification will use, so the counts here are what that gate is later checked
against. Full transcript in `pytest_full_output.txt`.

| group | result |
|---|---|
| exp106 | 192 passed, 29 skipped |
| exp105 (exp106's qualification subset) | 166 passed |
| exp104 | 131 passed |
| exp101 (certified subset) | 58 passed |
| exp102 (certified subset) | 17 passed |

`source_tree_sha256` at gate time is `e88e8f67...596c18fb`, recorded at the top of
the transcript, and it is byte-identical to the value bound into both pilot
configs. The bytecode check is empty.

**The 29 skips are the point, not an exemption.** They are the whole
`test_remote_execution.py` module plus the frozen-plan assertion, and they skip
because `PRODUCTION_PLAN_FROZEN` is `False`. nd-3 qualification requires *zero*
skipped tests, so an unfrozen plan cannot reach the machine. After Validation 003
freezes the plan they become 221 passed and 0 skipped, and that is the number
`QUALIFICATION_EXPECTED_PASSES["exp106"]` will be set to.

The groups run as separate processes because exp104's and exp105's suites both
ship a `tests/conftest.py` with no package marker, and pytest's rootdir-relative
module naming collides when they are collected together. That is a pre-existing
property of the repository layout, not of exp106, and the qualification runner
already invokes one subprocess per group.

## The two equality gates

These are what actually certify the port, and both passed.

**exp104, at `q = 0`.** Byte-identical `H_Z`, `H_X` and logical frames rebuilt
from exp104 registry rows; identical corrections, iteration counts, verdicts and
labels over `p = 0.02, 0.06, 0.10`. Includes a negative control that perturbs one
model and requires the comparison to fire.

**exp105, at `q = 0.05`.** The stronger one. exp106 reproduces exp105 **bit for
bit** on exp105's own frozen production registry at `m = 3, 4, 5, 8` across
`p = 0.001, 0.01, 0.04`: identical corrections, iteration counts, logical labels,
readout-match flags and verdicts.

This gate exists because the exp104 comparison cannot see the augmented matrix,
the mixed error channel, the readout draw or the `q > 0` failure criterion --
with the readout channel off, none of them is exercised. Those are exactly the
parts a port could break silently. It also carries a live control requiring the
comparison to run where the verdict is not constant (`m = 3, p = 0.04`, where
exp105 published `P_fail = 0.5388`), because two packages that both always
succeed would agree for uninteresting reasons.

A third check in the same file requires the panels to remain **disjoint** from
both predecessors: reading their registries is necessary, drawing from them is
not.

## What else is covered that exp105 did not cover

- **`tests/test_pilot_rules.py`** -- the section 6 rules are arithmetic, so they
  are tested before the pilot runs rather than after. exp105 shipped these
  untested and had to patch the allocation rule mid-flight when the pilot handed
  it a degenerate input. This includes that exact input: with `sigma_c` measured
  as zero at every point, the `s`-form still produces a finite, positive split.
  It also checks that the primary split is the constrained optimum by perturbing
  it and requiring the contrast variance to rise.
- **`tests/test_cost_benchmark.py`** -- the nd-3 cost benchmark is new code that
  runs exactly once on a machine reachable only through a deployment round trip.
  It is exercised here with the real decoder and only the host identity
  substituted, including a check that no outcome field leaves it.
- **`test_qualification_refuses_unmeasured_pass_counts`** -- the gate refuses to
  run against a pass count nobody has measured, rather than comparing against
  `None` and reporting a count mismatch.
- **`test_every_qualification_group_path_exists`** -- a mistyped group path would
  shrink the gate silently instead of failing it.

## Determinism (permanent discipline 15)

`tests/test_decoder_determinism.py` is a resident regression gate, not an
assumption. It first asserts that the augmented decoder actually **exhausts
`max_iter` without converging** at the production operating point -- so that the
OSD post-processing is genuinely on the path being tested -- and only then asserts
bit-exact repetition, across fresh decoder instances and independent of call
order. exp103 lost a full Stage 1 to a decoder that turned out to be randomized;
this is the check that would have caught it.

## Evidence in this directory

- `pytest_full_output.txt` -- the complete transcript of all five groups, the
  `source_tree_sha256` they ran against, and the bytecode check

## Reproduction

```bash
conda run -n 12 --no-capture-output python -m pytest \
  data/expander_code/exp106/tests -q
```
