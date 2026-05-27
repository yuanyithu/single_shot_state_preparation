# PT temperature logical-sector diagnostics

Flip rate is sector flips divided by diagnostic sample transitions. For stride=4 runs, one diagnostic interval is 4 measurements. Values are temperature-slot diagnostics, not identity-tracked replica histories.

## Config summary

| config | chains | measurements | stride | samples | min swap | bottleneck | cold flips | cold flip rate | hot flips | hot flip rate | hot winding acc | strict delivery | proxy delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M_strict_qhot032_m128 | 4 | 128 | 1 | 128 | 0.172266 | 12 | 0.000 | 0.000000 | 95.500 | 0.751969 | 0.002727 | 0 | 7 |
| N_strict_qhot035_m128 | 4 | 128 | 1 | 128 | 0.108366 | 11 | 0.000 | 0.000000 | 93.750 | 0.738189 | 0.020724 | 0 | 3 |
| O_strict_qhot044_m128 | 4 | 128 | 1 | 128 | 0.008893 | 6 | 0.000 | 0.000000 | 97.500 | 0.767717 | 0.385348 | 0 | 0 |
| P_stride_qhot032_m512_s4 | 4 | 512 | 4 | 128 | 0.209445 | 12 | 0.000 | 0.000000 | 98.500 | 0.775591 | 0.003405 | 0 | 22 |
| Q_stride_qhot035_m512_s4 | 4 | 512 | 4 | 128 | 0.126374 | 11 | 0.000 | 0.000000 | 95.250 | 0.750000 | 0.020201 | 0 | 20 |
| R_stride_qhot044_m512_s4 | 4 | 512 | 4 | 128 | 0.013605 | 6 | 0.000 | 0.000000 | 91.500 | 0.720472 | 0.385099 | 0 | 3 |
| E_pilot2_static_qhot044_m512 | 8 | 512 | 1 | 512 | 0.011866 | 6 | 1.750 | 0.003425 | 387.000 | 0.757339 | 0.385821 |  | 11 |
| F_pilot2_capped_qhot044_m512 | 8 | 512 | 1 | 512 | 0.012795 | 6 | 1.000 | 0.001957 | 380.125 | 0.743885 | 0.386536 |  | 18 |
| I_pilot3_qhot032_wr1_m128 | 4 | 128 | 1 | 128 | 0.216733 | 12 | 0.000 | 0.000000 | 95.500 | 0.751969 | 0.003294 |  | 8 |
| J_pilot3_qhot032_wr4_m128 | 4 | 128 | 1 | 128 | 0.204216 | 12 | 0.000 | 0.000000 | 93.500 | 0.736220 | 0.003239 |  | 4 |
| K_pilot3_qhot035_wr1_m128 | 4 | 128 | 1 | 128 | 0.140975 | 10 | 0.000 | 0.000000 | 92.250 | 0.726378 | 0.020778 |  | 5 |
