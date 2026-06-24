# exp41/003 — P1 L=7 生产（384 disorder，两节点）恢复手册

目的：p=0.11 的 L=7 生产数据，与复用的 exp40/004 L=3,4,5,6 合并，得到 L=3..7 完整集，压低 q_c 误差。见 `../plan.md` §3.3。

## 决策依据
- G2 gate 通过：L=7 在 crossing 区（q=0.034–0.040）kp129≈kp257（|Δ|<0.004，配对 SEM 内）；用 kp=129 与复用的 L3-6 同参，合并最干净。见 `../002_.../README.md` + paired 结果。
- nd-3 仍被他人占满（load 80）→ 按用户决定 (a) **两节点 nd-1/nd-2 直接开跑**，不等 nd-3。

## 配置
- L=7 only，kp=129，burn=512，max_eff_burn=512，meas=8192，stride=2，block=128，boot=800，winding=1，projection=linear，seed_scope=disorder_index，realization=rng_stream，use_numba。
- q 网格（= exp40/004，使可合并）：`0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070`
- 384 disorder = nd-1 192（seed **910000**）+ nd-2 192（seed **911000**）。
- **MASTER_RUN_ID：`exp41_p011L7_20260621_034148`**（TS=20260621_034148；screens `exp41_20260621_034148_nd{1,2}`；started ~15:47 nd-0 time）。
- 预计墙钟 ~42h（两节点并发，76 worker，L=7 全载 ~6033s/task，1920 task/节点）；总核时 ~6450。

## 远端
```
runs/<MASTER_RUN_ID>/nd1/collected/p0p11/   log: logs/<MASTER_RUN_ID>_nd1.log   screen: exp41_<TS>_nd1
runs/<MASTER_RUN_ID>/nd2/collected/p0p11/   log: logs/<MASTER_RUN_ID>_nd2.log   screen: exp41_<TS>_nd2
```

## 回收（两节点各一次）
```bash
ssh yuany 'tar -C ~/.single_shot/runs/<MASTER_RUN_ID>/nd1/collected -cf - p0p11' | tar -xf - -C nd1/collected/
ssh yuany 'tar -C ~/.single_shot/runs/<MASTER_RUN_ID>/nd2/collected -cf - p0p11' | tar -xf - -C nd2/collected/
```

## 合并分析
```bash
cd ..   # exp41/
E40=exp40_qtop_phase_boundary_20260610/004_p011_highstats_20260611
python analyze_exp41_p011.py --nboot 10000 --out 003_p011_L7_prod_384dis_20260621/p011_L34567_summary.json \
  --globs "$E40/nd1" "$E40/nd2" "$E40/nd3" "$E40/nd1_L6" "$E40/nd2_L6" "$E40/nd3_L6" \
          003_p011_L7_prod_384dis_20260621/nd1 003_p011_L7_prod_384dis_20260621/nd2
```
（q 网格须与 exp40 完全一致，脚本会 assert；w0/msigned/q_W/q_purity 全部从 delta_f 重算。）

## seed 台账
G1=900000 / G2=901000 / **P1=910000(nd1),911000(nd2)** / 预留补充 912000.. ；exp40 复用=86xxxx。两 shard seed 空间 [910000,910191]∪[911000,911191] 不重叠。
