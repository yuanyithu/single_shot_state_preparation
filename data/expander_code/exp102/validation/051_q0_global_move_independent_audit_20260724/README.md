# Independent audit of validations 047, 049, and 050

`run_independent_audit.py` rechecks the immutable reports and then performs an
independent algebra/raw-only reconstruction.

- Validation 047: catalog hard-coset algebra, `signature=Wd`, uniqueness,
  rank, L routes, accessible rank, and the optimistic rank-basis bottleneck.
- Validation 049: every raw identity, seed identity, P/U/L initial state,
  B-update transcript, hard-coset state, B block, label, weight, likelihood,
  counter, family summary, comparison, raw-set hash, and failed gate.
- Validation 050: legal MAP anchors, B geometry, equal logical label, and the
  two ordered bridge probabilities from a separately implemented conditional
  formula.

Terminal status:

```text
INDEPENDENT_AUDIT_PASS_FAILED_RESULTS_PRESERVED
audit SHA256:
c018e4af9b4aa5a78ae8a4c192e64c7a0beb8d53ca21e27c4c27176002a18767
```

Thus 047 and 049 remain failed, while 050 remains only a structural reason to
run a fresh T1 screen.  The audit also records that validation 049's historical
source identity omitted transitive dependencies.  The raw algebra and failed
conclusion are certified, but that partial identity is not upgraded into a
complete executable-source provenance claim.
