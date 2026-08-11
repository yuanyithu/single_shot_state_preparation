# Validation 002: local implementation gate

Status: **`PASS`**. Authorizes the locating pilot (Validation 003). No remote
transfer, no production compute, no physical result.

## What this gate is for

exp105 adds a `q > 0` branch to a machine exp104 already certified. Three things
therefore have to be established before any pilot task runs: that the new branch
is deterministic where the replay policy assumes it is, that the new scoring map
is the one the contract names, and that nothing in the shared machinery moved
while the branch was added.

## Evidence

`pytest_full_output.txt` and `pytest_exit_code.txt`, three groups run as separate
pytest sessions because exp104 and exp105 share test-module basenames and a
single session cannot import both.

| group | result |
|---|---|
| exp105 | 137 passed |
| exp104 | 131 passed |
| exp101 | 366 passed, 2 warnings |

The two exp101 warnings are its own expected deprecated-ensemble-alias warnings.
Nothing was skipped, xfailed, xpassed or deselected in any group. exp104's suite
is run unchanged to show that exp105 did not disturb the experiment it reuses
codes and comparisons from.

Source commit `6062833`, config
`config/noisy_mc.pilot.v1.json`, registry SHA256
`83b4602ad453e466e687d44bbfe594b1a33bc755ff1c3e6b413efe243bdb85e8`.

## The three contract-critical test files

**`test_decoder_determinism.py` — permanent discipline 15.** The replay gate
covers ten percent of tasks, which is only defensible if determinism is measured
rather than assumed. The file is deliberately awkward: it first asserts that the
production operating point actually exhausts `max_iter` without converging, so
that the determinism assertions run on the ordered-statistics path where an
implementation is most likely to depend on uninitialised state, and only then
asserts bit-exact repetition, agreement between independently constructed
decoder objects, and independence from call order. A determinism test that
quietly ran on converged instances would prove nothing about the run it protects.

**`test_label_map.py` — the scoring map.** `phi_r` is pinned against exp101's own
frame, against an independent reconstruction from the section's pivot rule, and
against the kernel characterisation `phi_r(v) = 0` iff
`v in rowspace(H_X) + im(r_sec)`.

It also records a structural fact that changes what "new at `q > 0`" means. For
this family the label basis **equals** `logical_Z`: `r_sec` places values only on
the RREF pivot columns of `H_Z`, exp101's `logical_Z` basis is supported entirely
off those columns, so the section term vanishes identically. The label map is
therefore *not* what changes at `q > 0`. What changes is the **criterion**:
exp104 also required the residual to have zero syndrome, and exp105 does not,
because the protocol's final perfect round measures the residual syndrome exactly
and removes it. `test_the_failure_criterion_differs_from_exp104_at_nonzero_syndrome`
asserts the two criteria are not interchangeable, and
`test_score_logical_class_ignores_the_readout_residual` asserts that a wrong
readout estimate alone never fails a trial. The pipeline still computes the label
through the certified frame rather than assuming the collapse.

**`test_exp104_equality.py` — cross-package equality.** exp104's production raw
lives on nd-3 and is not tracked in Git, so this is a package-to-package
comparison on shared codes and shared seeds, the same form exp104's own
Validation 002 used against exp103. For five codes spanning `m = 3, 4, 5` and
three `p` values, both packages build the model, both decode `H_Z` with the
exp104 decoder identity, and both score: `H_Z`, `H_X`, logical frames,
corrections, iteration counts, verdicts and labels all agree. A negative control
perturbs one matrix and requires the comparison to fire, so a comparison that
silently compared nothing would not pass.

## What is deliberately not here

`test_remote_execution.py` was removed rather than ported. It tests the remote
deployment artifacts against the production config, and the production config
does not exist yet by design. It is re-added at Validation 004, when the plan is
frozen and the remote config is written. Recording the gap here is the point;
carrying a vacuous test forward would not be.

## Authority end

Local only. This authorizes the locating pilot and nothing else. It does not
authorize a remote transfer, a production task, or any claim about physics.

## Reproduction

```bash
for grp in exp105 exp104 exp101; do
  conda run -n 12 --no-capture-output python -m pytest data/expander_code/$grp/tests -v
done
```
