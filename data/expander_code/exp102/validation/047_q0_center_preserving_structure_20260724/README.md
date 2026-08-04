# Validation 047: center-preserving logical XOR structure

This directory evaluates the immutable local-only contract
`exp102.q0_center_preserving.structure.v0`.  It contains no Markov-chain raw
and has no authority to estimate `q_top` or launch HARD2.

The review and frozen rule are in
[`CENTER_PRESERVING_REVIEW.md`](../../reviews/CENTER_PRESERVING_REVIEW.md) and
[`q0_center_preserving.structure.v0.json`](../../config/q0_center_preserving.structure.v0.json).

Run locally with:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n 12 --no-capture-output \
  python data/expander_code/exp102/validation/047_q0_center_preserving_structure_20260724/run_structure_probe.py \
  --config data/expander_code/exp102/config/q0_center_preserving.structure.v0.json
```

The terminal interpretation is written only after exact artifact replay,
hard-coset/signature/rank checks, and adversarial BASE/P/U/L one-step transport
profiles complete.
