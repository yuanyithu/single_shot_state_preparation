# exp41/006 — P2 step3：平台固化 p=0.05 + p=0.17（加 L=7）恢复手册

目的：phase-2 step1/2 已证 q_c(p) 在 [0.11,0.22] 平坦 ≈0.035、p_c≈0.227 处近垂直收口。用户选「固化平台、收尾」(不再硬抠 knee)。本步**补 p=0.05 与 p=0.17 的 L=7**（复用 exp40/005 p0p05/p0p17 的 L3-5×384），把平台坐满，相图定为「平台≈0.035 + p_c≈0.227 近垂直收口」。

## 配置（与 exp41/003,004,005 同参以便横向）
- **L=7 only**，384 disorder/点（复用 exp40/005 同 p 的 L3-5×384）。
- kp=129，burn=512，max_eff=512，meas=8192，stride=2，block=128，boot=800，winding=1，projection=linear，seed_scope=disorder_index，realization=rng_stream，use_numba。q 网格 = exp40 十点。
- 每节点**顺序两 cell**：nd-1 先 p=0.05(seed940000) 再 p=0.17(seed942000)；nd-2 先 p=0.05(seed941000) 再 p=0.17(seed943000)。每点 192+192=384。
- workers=76/节点。p=0.05 约 ~30h 先出，p=0.17 约 ~60h 收尾（L=7 only，每节点 1920 task/cell）。
- nd-3 仍被他人占用(load 74) → 两节点。
- MASTER_RUN_ID：`exp41_plateaufill_20260630_100510`（TS=20260630_100510；screens exp41_20260630_100510_nd{1,2}）。ControlMaster socket `~/.ssh/cm-exp41-yuany`。

## 启动命令（已执行；恢复时复用）
launcher = `001_p011_g1_smoke_20260620/launch_exp41.sh`，env：
```
SOCK=$HOME/.ssh/cm-exp41-yuany
QGRID=0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070
STAGE_DIR=<abs 006 dir>  MASTER_RUN_ID=exp41_plateaufill_20260630_100510  RUN_TIMESTAMP=20260630_100510 \
FIXED_LATTICE_SIZES=7  NUM_DISORDER_SAMPLES=192  NUM_WORKERS=76 \
SSH_CTL="-o ControlPath=$SOCK -o ControlMaster=auto -o ControlPersist=900" \
CELLS="nd-1|0.05|940000|$QGRID;nd-1|0.17|942000|$QGRID;nd-2|0.05|941000|$QGRID;nd-2|0.17|943000|$QGRID" \
bash 001_.../launch_exp41.sh
```

## 收尾（跑完后，每点独立）
1. scp 两分片 collected NPZ → `006_…/nd{1,2}/collected/{p0p05,p0p17}/`，md5 校验。
2. 合并分析（各点 = 本 run 两分片 L7 + exp40/005 同 p 三节点 L3-5）：
```
cd ..  # exp41/
E5=exp40_qtop_phase_boundary_20260610/005_boundary_highstats_20260612
# p=0.05
python analyze_exp41_p011.py --nboot 10000 --out 006_…/p005_L34567_summary.json \
  --globs "$E5/nd1/collected/p0p05" "$E5/nd2/collected/p0p05" "$E5/nd3/collected/p0p05" \
          006_…/nd1/collected/p0p05 006_…/nd2/collected/p0p05
# p=0.17 同理换 p0p17
```
   注意 glob 精确到 `.../collected/p0pXX`（exp40/005 同级有别的 p 子目录）。
3. 出图（仿 005 plot 脚本）。
4. **最终相图**：汇总 p=0.05,0.11,0.17,0.21,0.22 的 w0 L3-L7 q_c → 画 q_c(p) 平台 + p_c≈0.227 近垂直收口。
5. 提交全部 phase-2 交付（step1/2/3 + launcher SSH_CTL 改动）。

## seed 台账
phase-1：900000/901000/910000,911000。step1(p=0.21)=920000,921000。step2(p=0.22)=930000,931000。**step3：p=0.05=940000(nd1),941000(nd2)；p=0.17=942000(nd1),943000(nd2)。** exp40 复用：879xxx(005,各 p)。

## 状态
启动服务器时间 2026-06-29 22:08(nd1)/22:10(nd2)。两 screen Detached、BEGIN p=0.05、workers=76。watcher 盯全 run 完成(两 cell × 两节点)。
