# Stage A model anchor summary

Overall: PASS

No sampling was used.  The exact L=2 check enumerates all 2^24 x-space configurations.

## Gate Numbers

| Gate | Criterion | Result | Status |
|---|---|---:|---|
| A1 | 1200 random x energies match brute force exactly | 0.000e+00 | PASS |
| A2 | L=2 sum_g Z_g equals global Z, log error < 1e-9 | 4.441e-16 | PASS |
| A3 | Kp=0 gives w_g=1/8 and q_top=0 on L=2,L=3 | max abs dw=0.000e+00, max abs dq=0.000e+00 | PASS |
| A4 | q_top purity roundtrip for random w_g | 0.000e+00 | PASS |

## Details

- A2 section backend: `linear_elimination_fallback`.
- A2 exact point: L=2, p=0.173, q=0.197, eta weight=7, measurement-error weight=3.
- A3 rank increments:
  - L=2: rank(H)=14, rank([H;Z])=17, increment=3.
  - L=3: rank(H)=52, rank([H;Z])=55, increment=3.
