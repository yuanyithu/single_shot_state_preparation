# HCA-RHB1 local adversarial-pair viability

This is a fresh local-only diagnostic following the current-source runtime
rebind in `042`.  It tests a real, exact two-replica Markov kernel, not a structural
component enumeration: each replica targets

```text
pi(e | y) proportional to (.04/.96)^|e|,  H_Z e = y.
```

The joint target is the product of those two marginals.  The true planted
error is used only for the legal adversarial P and deterministic L starts; it
does not occur in the energy, heatbath ratio, HCA identity, or estimator.

## Red-team contract

- The coordinate origin is the legal section state for `y`, not the planted
  error and not a physical all-zero string.  The hard sentinel has nonzero
  syndrome, so physical zero is outside the support.
- `PP`, `UU`, `LL`, and `PL` are all legal pair starts.  The two U members are
  independently exact K=0 uniform hard-coset draws.  The two L members are
  the *first two* label-distinct P-derived starts under the frozen canonical
  single/pair/triple catalog rule, not a pair chosen after inspecting this
  run.  `PL` is a deliberate whole-pair-swap control.
- A Houdayer label change that merely exchanges the original two replica
  states is recorded as a whole-pair exchange and counts as zero new unordered
  pair states.  Only `measurement_new_unordered_pair` is structural movement.
- The independent units for every comparison are the eight replica pairs,
  never their sixteen correlated member chains.  The diagnostics are the
  pair-averaged normalized weight and all 64 basis characters; this run does
  not calculate or report `q_top`.
- Exact small-code stationarity is required before this run.  A local pass
  still has no rigorous tail/normalizer bound and therefore authorizes neither
  remote work nor a posterior/physics/production claim.

The unchanged `128 + 1024` clock schedule, one 832-coordinate random-scan
sweep per replica per clock, four starts, and eight pair trajectories per
start are frozen in `q0_houdayer_pair.local.v0.json`.  No extra pairs,
extension, reweighting, or gate change is permitted after output exists.
