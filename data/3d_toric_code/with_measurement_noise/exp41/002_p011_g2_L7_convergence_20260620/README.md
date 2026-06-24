# exp41/002 — G2 L=7 收敛 gate（恢复手册）

目的：在投 ~6500 核时进 P1 前，验证 L=7 在生产参数 kp=129/burn=512 下未被 kp-grid 偏差污染。见 `../plan.md` §3.2（精简为配对 kp A/B）。

## 设计（精简版，配对）

- L=7 only，q=0.034,0.040,0.048（跨 w0~0.033 与 q_W~0.042 两个 crossing），64 disorder，**两 run 同 seed_base=901000 → 逐 disorder 配对**。
- 依据：G1 实测 L=7 在 q=0.040 的 pass_fraction=0.938、grid_q_top_abs_diff mean=0.0054，与已被 exp40 接受的 L=6（0.922 / 0.0054）相当 → 预期 kp=129 足够；本 gate 做定量确认。
- burn-in 不单独 A/B：max_eff=512 的 cap 对 L=6 已验证可用，L=7 同 cap，风险低；若 kp 测试异常再补 burn A/B。

| run | 节点 | kp | burn | MASTER_RUN_ID |
|---|---|---|---|---|
| baseline | nd-1 | 129 | 512 | `exp41_g2kp129_20260620_191408` |
| kp test | nd-2 | 257 | 512 | `exp41_g2kp257_20260620_191408` |

固定其余参数同 plan §3.0（max_eff_burn=512, meas=8192, stride=2, block=128, boot=800, winding=1, projection=linear, seed_scope=disorder_index, realization=rng_stream, use_numba）。

## 远端路径
```
runs/exp41_g2kp129_20260620_191408/nd1/collected/p0p11/   log: logs/exp41_g2kp129_20260620_191408_nd1.log
runs/exp41_g2kp257_20260620_191408/nd2/collected/p0p11/   log: logs/exp41_g2kp257_20260620_191408_nd2.log
screens: exp41_20260620_191408_kp129_nd1 / exp41_20260620_191408_kp257_nd2
```

## 回收
```bash
ssh yuany 'tar -C ~/.single_shot/runs/exp41_g2kp129_20260620_191408/nd1/collected -cf - p0p11' | tar -xf - -C kp129/
ssh yuany 'tar -C ~/.single_shot/runs/exp41_g2kp257_20260620_191408/nd2/collected -cf - p0p11' | tar -xf - -C kp257/
```

## 判据
逐 disorder 配对比较 w0 / q_W（softmax(-delta_f) 重算）。kp=129 与 257 的配对均值差 |Δ| 在各 q 点都 < 配对 SEM（且物理上 < ~0.005）→ **kp=129 通过，P1 用 129/512**。否则 P1 升 kp 或补 burn A/B（见 plan §3.2 分支）。

## seed
G2=901000（nd-1/nd-2 同种子配对）。不与 G1(900000)、P1(910000/911000/912000)、exp40(86xxxx) 冲突。
