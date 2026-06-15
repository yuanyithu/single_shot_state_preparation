# exp36 cold-sector histogram convergence summary

## Runs

### sector

- file: `/home/DATA1/users/yuany/.single_shot/exp36/015_common_disorder_sector_gate_20260530/run01_sector_q023_d3_m2048_seed515000/run01_sector_q023_d3_m2048_seed515000.npz`
- initial mode: `sector`
- q_top mean: 0.996842
- chain q_top range: 0.992204 .. 1.000000
- chain first-half vs second-half TV: mean=0.0018, max=0.0039
- top sectors:
  -   0 `+++++++` count=24542 prob=0.9986
  - 102 `+--++--` count=11 prob=0.0004
  -  85 `-+-+-+-` count=5 prob=0.0002
  - 120 `+++----` count=5 prob=0.0002
  -  51 `--++--+` count=5 prob=0.0002
  -  30 `+----++` count=4 prob=0.0002
  -  45 `-+--+-+` count=4 prob=0.0002

### all_zero

- file: `/home/DATA1/users/yuany/.single_shot/exp36/015_common_disorder_sector_gate_20260530/run02_allzero_q023_d3_m2048_seed515000/run02_allzero_q023_d3_m2048_seed515000.npz`
- initial mode: `all_zero`
- q_top mean: 0.996935
- chain q_top range: 0.988874 .. 1.000000
- chain first-half vs second-half TV: mean=0.0015, max=0.0039
- top sectors:
  -   0 `+++++++` count=24543 prob=0.9987
  - 102 `+--++--` count=13 prob=0.0005
  -  30 `+----++` count=7 prob=0.0003
  -  85 `-+-+-+-` count=6 prob=0.0002
  -  51 `--++--+` count=5 prob=0.0002
  -  45 `-+--+-+` count=1 prob=0.0000
  - 120 `+++----` count=1 prob=0.0000

### random_high_weight

- file: `/home/DATA1/users/yuany/.single_shot/exp36/015_common_disorder_sector_gate_20260530/run03_randomhigh_q023_d3_m2048_seed515000/run03_randomhigh_q023_d3_m2048_seed515000.npz`
- initial mode: `random_high_weight`
- q_top mean: 0.997492
- chain q_top range: 0.991093 .. 1.000000
- chain first-half vs second-half TV: mean=0.0013, max=0.0049
- top sectors:
  -   0 `+++++++` count=24549 prob=0.9989
  - 102 `+--++--` count=9 prob=0.0004
  - 120 `+++----` count=5 prob=0.0002
  -  30 `+----++` count=4 prob=0.0002
  -  85 `-+-+-+-` count=3 prob=0.0001
  -  51 `--++--+` count=3 prob=0.0001
  -  75 `--+-++-` count=2 prob=0.0001
  -  45 `-+--+-+` count=1 prob=0.0000

## Pairwise TV Between Runs

| run | sector | all_zero | random_high_weight |
|---|---|---|---|
| sector | 0.0000 | 0.0003 | 0.0004 |
| all_zero | 0.0003 | 0.0000 | 0.0005 |
| random_high_weight | 0.0004 | 0.0005 | 0.0000 |

## Per-Disorder Gate

### disorder 0
- q_top: sector=0.993872, all_zero=0.993594, random_high_weight=0.994150
- q_top spread: 0.000556
- pairwise cold-sector TV:
| run | sector | all_zero | random_high_weight |
|---|---|---|---|
| sector | 0.0000 | 0.0005 | 0.0005 |
| all_zero | 0.0005 | 0.0000 | 0.0009 |
| random_high_weight | 0.0005 | 0.0009 | 0.0000 |
- top sectors by run:
  - sector: 0 `+++++++` p=0.9973; 102 `+--++--` p=0.0010; 120 `+++----` p=0.0005; 85 `-+-+-+-` p=0.0004; 51 `--++--+` p=0.0004
  - all_zero: 0 `+++++++` p=0.9972; 102 `+--++--` p=0.0011; 30 `+----++` p=0.0007; 85 `-+-+-+-` p=0.0004; 51 `--++--+` p=0.0004
  - random_high_weight: 0 `+++++++` p=0.9974; 102 `+--++--` p=0.0007; 30 `+----++` p=0.0005; 120 `+++----` p=0.0005; 75 `--+-++-` p=0.0002

### disorder 1
- q_top: sector=0.997769, all_zero=0.999163, random_high_weight=0.999163
- q_top spread: 0.001394
- pairwise cold-sector TV:
| run | sector | all_zero | random_high_weight |
|---|---|---|---|
| sector | 0.0000 | 0.0007 | 0.0006 |
| all_zero | 0.0007 | 0.0000 | 0.0001 |
| random_high_weight | 0.0006 | 0.0001 | 0.0000 |
- top sectors by run:
  - sector: 0 `+++++++` p=0.9990; 102 `+--++--` p=0.0002; 45 `-+--+-+` p=0.0002; 85 `-+-+-+-` p=0.0002; 51 `--++--+` p=0.0001
  - all_zero: 0 `+++++++` p=0.9996; 102 `+--++--` p=0.0004
  - random_high_weight: 0 `+++++++` p=0.9996; 102 `+--++--` p=0.0002; 51 `--++--+` p=0.0001

### disorder 2
- q_top: sector=0.998884, all_zero=0.998048, random_high_weight=0.999163
- q_top spread: 0.001115
- pairwise cold-sector TV:
| run | sector | all_zero | random_high_weight |
|---|---|---|---|
| sector | 0.0000 | 0.0006 | 0.0002 |
| all_zero | 0.0006 | 0.0000 | 0.0006 |
| random_high_weight | 0.0002 | 0.0006 | 0.0000 |
- top sectors by run:
  - sector: 0 `+++++++` p=0.9995; 120 `+++----` p=0.0001; 51 `--++--+` p=0.0001; 102 `+--++--` p=0.0001; 45 `-+--+-+` p=0.0001
  - all_zero: 0 `+++++++` p=0.9991; 85 `-+-+-+-` p=0.0004; 51 `--++--+` p=0.0002; 102 `+--++--` p=0.0001; 30 `+----++` p=0.0001
  - random_high_weight: 0 `+++++++` p=0.9996; 120 `+++----` p=0.0001; 85 `-+-+-+-` p=0.0001; 102 `+--++--` p=0.0001
