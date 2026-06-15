# exp36 cold-sector histogram convergence summary

## Runs

### sector

- file: `/home/DATA1/users/yuany/.single_shot/exp36/016_lowq_sector_sanity_20260530/run01_sector_q008_d2_m1024_seed516000/run01_sector_q008_d2_m1024_seed516000.npz`
- initial mode: `sector`
- q_top mean: 1.000000
- chain q_top range: 1.000000 .. 1.000000
- chain first-half vs second-half TV: mean=0.0000, max=0.0000
- top sectors:
  -   0 `+++++++` count=8192 prob=1.0000

### all_zero

- file: `/home/DATA1/users/yuany/.single_shot/exp36/016_lowq_sector_sanity_20260530/run02_allzero_q008_d2_m1024_seed516000/run02_allzero_q008_d2_m1024_seed516000.npz`
- initial mode: `all_zero`
- q_top mean: 0.999721
- chain q_top range: 0.997770 .. 1.000000
- chain first-half vs second-half TV: mean=0.0002, max=0.0020
- top sectors:
  -   0 `+++++++` count=8191 prob=0.9999
  - 102 `+--++--` count=1 prob=0.0001

### random_high_weight

- file: `/home/DATA1/users/yuany/.single_shot/exp36/016_lowq_sector_sanity_20260530/run03_randomhigh_q008_d2_m1024_seed516000/run03_randomhigh_q008_d2_m1024_seed516000.npz`
- initial mode: `random_high_weight`
- q_top mean: 1.000000
- chain q_top range: 1.000000 .. 1.000000
- chain first-half vs second-half TV: mean=0.0000, max=0.0000
- top sectors:
  -   0 `+++++++` count=8192 prob=1.0000

## Pairwise TV Between Runs

| run | sector | all_zero | random_high_weight |
|---|---|---|---|
| sector | 0.0000 | 0.0001 | 0.0000 |
| all_zero | 0.0001 | 0.0000 | 0.0001 |
| random_high_weight | 0.0000 | 0.0001 | 0.0000 |

## Per-Disorder Gate

### disorder 0
- q_top: sector=1.000000, all_zero=1.000000, random_high_weight=1.000000
- q_top spread: 0.000000
- pairwise cold-sector TV:
| run | sector | all_zero | random_high_weight |
|---|---|---|---|
| sector | 0.0000 | 0.0000 | 0.0000 |
| all_zero | 0.0000 | 0.0000 | 0.0000 |
| random_high_weight | 0.0000 | 0.0000 | 0.0000 |
- top sectors by run:
  - sector: 0 `+++++++` p=1.0000
  - all_zero: 0 `+++++++` p=1.0000
  - random_high_weight: 0 `+++++++` p=1.0000

### disorder 1
- q_top: sector=1.000000, all_zero=0.999442, random_high_weight=1.000000
- q_top spread: 0.000558
- pairwise cold-sector TV:
| run | sector | all_zero | random_high_weight |
|---|---|---|---|
| sector | 0.0000 | 0.0002 | 0.0000 |
| all_zero | 0.0002 | 0.0000 | 0.0002 |
| random_high_weight | 0.0000 | 0.0002 | 0.0000 |
- top sectors by run:
  - sector: 0 `+++++++` p=1.0000
  - all_zero: 0 `+++++++` p=0.9998; 102 `+--++--` p=0.0002
  - random_high_weight: 0 `+++++++` p=1.0000
