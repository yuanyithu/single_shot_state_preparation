# exp41/001 — G1 运维 smoke（恢复手册）

目的：验证 exp41 管线（workers/numba/projection/NPZ）并实测 L=7 单点 walltime，不下物理结论。详见 `../plan.md` §3.1。

## G1 run

- MASTER_RUN_ID：`exp41_g1smoke_20260620_155118`
- launcher：`launch_exp41.sh`（从 exp40/002 复制改造；STAGE_DIR 可 env 覆盖，机制零改动）
- cell：`nd-1 | p=0.11 | seed_base=900000 | q=0.040`，`L=3,4,5,6,7`，`64 disorder`
- 固定参数（plan §3.0）：projection=linear, kp=129, burn=512, max_eff_burn=512, meas=8192, stride=2, block=128, boot=800, winding=1, seed_scope=disorder_index, realization=rng_stream, common_disorder_across_q, use_numba
- 远端：
  - run_root：`~/.single_shot/runs/exp41_g1smoke_20260620_155118/nd1`
  - log：`~/.single_shot/logs/exp41_g1smoke_20260620_155118_nd1.log`
  - screen：`exp41_20260620_155118_nd1`

### 启动命令（已用）
```bash
cd <STAGE_DIR>
RUN_TIMESTAMP=20260620_155118 MASTER_RUN_ID=exp41_g1smoke_20260620_155118 \
  bash launch_exp41.sh
```

### 健康检查
```bash
ssh yuany 'tail -40 ~/.single_shot/logs/exp41_g1smoke_20260620_155118_nd1.log'
ssh yuany "ssh nd-1 'uptime; screen -ls | grep exp41'"
# 预期：日志 workers≈76、use_numba=True、projection_mode=linear、出现 [k/total] 进度行；load≈76
```

### 回收
```bash
ssh yuany 'tar -C ~/.single_shot/runs/exp41_g1smoke_20260620_155118/nd1/collected -cf - p0p11' \
  | tar -xf - -C <STAGE_DIR>/nd1/collected/
```

## exp41 phase-1 seed 规划（避免与历史重叠；历史用到 850000–862xxx）

| 用途 | seed_base | 节点 |
|---|---|---|
| G1 smoke | 900000 | nd-1 |
| G2 收敛 gate | 901000 | nd-1/2（按 §3.2 四组，单节点串行或分节点） |
| P1 L=7 生产 | 910000 / 911000 / 912000 | nd-1 / nd-2 / nd-3 各 128 disorder |
| disorder 补充(预留) | 913000 / 914000 / 915000 | 按需 |

注：exp40/004 复用数据 seed=860000/861000/862000（L3-6×384），不得与上表冲突。

## 节点状态（launch 时）
- nd-1 idle(80c)、nd-2 idle(80c)、nd-3 **load≈80（被他人占用）**。G1 选 idle 的 nd-1 以得到干净 walltime。P1 三节点分片前需重新确认 nd-3 是否空出。
