# exp41/005 — P2 step2：p=0.22 全 L=3,4,5,7（近 p_c knee 猎取）恢复手册

目的：step1(p=0.21)发现边界在 p∈[0.11,0.21] 是**平的**(q_c≈0.034)、没向 p_c≈0.227 收口。本步把点推到 **p=0.22**(更贴 p_c)以判定：q_c 是继续平台、还是在 p∈(0.21,0.227) 窄窗里**陡降到 0**(knee)。用户选 0.22(比 0.225 稳、近临界收敛更可控)。

## 配置（与 exp41/004 + exp40 同参以便横向对比）
- **全 L=3,4,5,7**（p=0.22 无 exp40 复用，小 L 也自跑；L 集与 004 一致便于直接比，**故意不含 L6**：L6-L7 在有序侧简并、对 headline 无增益）。
- kp=129，burn=512，max_eff=512，meas=8192，stride=2，block=128，boot=800，winding=1，projection=linear，seed_scope=disorder_index，realization=rng_stream，use_numba。
- q 网格 = exp40 十点：0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070（与 0.21 完全一致；[0.012,0.040] 内 7 点，足够分辨 crossing 是否向下移）。
- 384 disorder = nd-1 192（seed 930000）+ nd-2 192（seed 931000）。nd-3 仍被他人占用(load 84) → 两节点。
- workers=76/节点。每节点 4 L×10 q×192 = 7680 task，L7 主导墙钟，预计 ~40–45h。
- MASTER_RUN_ID：`exp41_p022L7_20260627_122412`（TS=20260627_122412；screens exp41_20260627_122412_nd{1,2}）。经 ssh ControlMaster 多路复用启动（socket `~/.ssh/cm-exp41-yuany`）。

## 启动命令（已执行；恢复时复用）
launcher = `001_p011_g1_smoke_20260620/launch_exp41.sh`，env 覆盖：
```
SOCK=$HOME/.ssh/cm-exp41-yuany
QGRID=0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070
STAGE_DIR=<abs 005 dir>  MASTER_RUN_ID=exp41_p022L7_20260627_122412  RUN_TIMESTAMP=20260627_122412 \
FIXED_LATTICE_SIZES=3,4,5,7  NUM_DISORDER_SAMPLES=192  NUM_WORKERS=76 \
SSH_CTL="-o ControlPath=$SOCK -o ControlMaster=auto -o ControlPersist=900" \
CELLS="nd-1|0.22|930000|$QGRID;nd-2|0.22|931000|$QGRID" \
bash 001_.../launch_exp41.sh
```

## 收尾（跑完后）
1. scp 两分片 collected NPZ+summary 到 `005_…/nd{1,2}/collected/p0p22/`，md5 校验。
2. 合并分析（无 exp40 复用，输入只有本 run 两分片）：
```
cd ..  # exp41/
python analyze_exp41_p011.py --nboot 10000 --out 005_p022_L7_knee_384dis_20260627/p022_L34567_summary.json \
  --globs 005_p022_L7_knee_384dis_20260627/nd1 005_p022_L7_knee_384dis_20260627/nd2
```
3. 出图（仿 004 的 plot 脚本，改路径/p 标题）。

## 判据
- **若 w0 大 L 对 crossing 明显 < 0.034**（向 0 移）→ knee 出现、收口证据。**若仍 ≈0.034** → 平台延伸到 0.22，knee 更靠 p_c（需再推 p=0.225）。
- **盯 L=7 crossing 区 pass_fraction**：p=0.22 近临界，冻结扇区风险升（虽 sector-TI 法对此鲁棒）。若 crossing 区(预计 q≈0.02–0.034)pass<0.9，需补 burn 或 kp。
- 若 crossing 被推到网格低边缘(≤0.018)，补 q={0.006,0.009} 低点 bracket。

## seed 台账
phase-1：G1=900000/G2=901000/P1=910000,911000。phase-2 step1(p=0.21)=920000,921000。**phase-2 step2(p=0.22)=930000(nd1),931000(nd2)。**

## 状态
启动于服务器时间 2026-06-27 00:29(nd1)/00:31(nd2)。两 screen Detached、BEGIN cell p=0.22、workers=76。watcher 监控完成/故障。
