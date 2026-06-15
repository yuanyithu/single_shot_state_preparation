# exp36 cold-sector histogram convergence summary

## Runs

### sector_m2048

- file: `/home/DATA1/users/yuany/.single_shot/exp36/014_cold_sector_histogram_convergence_20260530/run04_sector_q023_m2048_seed514001/run04_sector_q023_m2048_seed514001.npz`
- initial mode: `sector`
- q_top mean: 0.986653
- chain q_top range: 0.982227 .. 0.991093
- chain first-half vs second-half TV: mean=0.0039, max=0.0059
- top sectors:
  -   0 `+++++++` count=8144 prob=0.9941
  -  30 `+----++` count=11 prob=0.0013
  - 102 `+--++--` count=10 prob=0.0012
  -  75 `--+-++-` count=8 prob=0.0010
  -  85 `-+-+-+-` count=6 prob=0.0007
  - 120 `+++----` count=6 prob=0.0007
  -  51 `--++--+` count=4 prob=0.0005
  -  45 `-+--+-+` count=3 prob=0.0004

### all_zero_m2048

- file: `/home/DATA1/users/yuany/.single_shot/exp36/014_cold_sector_histogram_convergence_20260530/run05_allzero_q023_m2048_seed514001/run05_allzero_q023_m2048_seed514001.npz`
- initial mode: `all_zero`
- q_top mean: 0.988040
- chain q_top range: 0.985546 .. 0.991093
- chain first-half vs second-half TV: mean=0.0039, max=0.0078
- top sectors:
  -   0 `+++++++` count=8149 prob=0.9948
  - 102 `+--++--` count=13 prob=0.0016
  - 120 `+++----` count=11 prob=0.0013
  -  51 `--++--+` count=5 prob=0.0006
  -  45 `-+--+-+` count=4 prob=0.0005
  -  30 `+----++` count=4 prob=0.0005
  -  75 `--+-++-` count=3 prob=0.0004
  -  85 `-+-+-+-` count=3 prob=0.0004

### random_high_m2048

- file: `/home/DATA1/users/yuany/.single_shot/exp36/014_cold_sector_histogram_convergence_20260530/run06_randomhigh_q023_m2048_seed514001/run06_randomhigh_q023_m2048_seed514001.npz`
- initial mode: `random_high_weight`
- q_top mean: 0.985546
- chain q_top range: 0.984439 .. 0.987766
- chain first-half vs second-half TV: mean=0.0051, max=0.0068
- top sectors:
  -   0 `+++++++` count=8140 prob=0.9937
  -  30 `+----++` count=14 prob=0.0017
  - 120 `+++----` count=12 prob=0.0015
  - 102 `+--++--` count=9 prob=0.0011
  -  75 `--+-++-` count=7 prob=0.0009
  -  45 `-+--+-+` count=4 prob=0.0005
  -  85 `-+-+-+-` count=4 prob=0.0005
  -  51 `--++--+` count=2 prob=0.0002

### sector_m4096

- file: `/home/DATA1/users/yuany/.single_shot/exp36/014_cold_sector_histogram_convergence_20260530/run07_sector_q023_m4096_seed514001/run07_sector_q023_m4096_seed514001.npz`
- initial mode: `sector`
- q_top mean: 0.986654
- chain q_top range: 0.983333 .. 0.992204
- chain first-half vs second-half TV: mean=0.0028, max=0.0044
- top sectors:
  -   0 `+++++++` count=16288 prob=0.9941
  - 102 `+--++--` count=23 prob=0.0014
  -  30 `+----++` count=23 prob=0.0014
  - 120 `+++----` count=22 prob=0.0013
  -  45 `-+--+-+` count=9 prob=0.0005
  -  51 `--++--+` count=7 prob=0.0004
  -  85 `-+-+-+-` count=7 prob=0.0004
  -  75 `--+-++-` count=5 prob=0.0003

## Pairwise TV Between Runs

| run | sector_m2048 | all_zero_m2048 | random_high_m2048 | sector_m4096 |
|---|---|---|---|---|
| sector_m2048 | 0.0000 | 0.0018 | 0.0012 | 0.0010 |
| all_zero_m2048 | 0.0018 | 0.0000 | 0.0020 | 0.0010 |
| random_high_m2048 | 0.0012 | 0.0020 | 0.0000 | 0.0010 |
| sector_m4096 | 0.0010 | 0.0010 | 0.0010 | 0.0000 |
