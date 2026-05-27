# exp35 正式扫描启动记录

- 启动时间: 2026-05-25 00:32:15 CST
- 代码版本: `1ad4bc65d67740340af84e558ffcb91d4dab96d7`
- 主 run id: `3d_toric_exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_003215`
- 本地回收目录: `data/3d_toric_code/with_measurement_noise/exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_nd12`
- 远端节点:
  - nd-1: `3d_toric_exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_003215_nd1`
  - nd-2: `3d_toric_exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_003215_nd2`
- screen:
  - nd-1: `ssprep_exp35_20260525_003215_nd1`
  - nd-2: `ssprep_exp35_20260525_003215_nd2`
- 远端日志:
  - nd-1: `/home/DATA1/users/yuany/.single_shot/logs/3d_toric_exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_003215_nd1.log`
  - nd-2: `/home/DATA1/users/yuany/.single_shot/logs/3d_toric_exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_003215_nd2.log`

## 参数

- 固定 `p=0.0500`
- `q=0.0800,0.0900,...,0.2300`
- `L=3,4,5,6`
- 每节点每点 `1024` disorder，总计目标每点 `2048` disorder
- `chunk_size=4`
- `pt_ladder_mode=sync_enlarge`
- `pt_num_temperatures=9`
- `pt_p_hot=0.44`, `pt_q_hot=0.44`
- `adaptive_pt_rounds=3`
- `adaptive_pt_calibration_sweeps=512`
- `single_bit_proposal_fraction=0.05`
- `observable_temperature_mode=cold`
- cluster update 已关闭

## 启动检查

- nd-1/nd-2 均通过远端 `conda env 11` 环境检查，`ldpc_available=1`，`numba_available=True`。
- 两个节点首个点 `L=3,q=0.0800` 均通过 quick exact/preflight。
- 两个节点均已启动 80 worker 的 chunk 生产阶段。
- 初始检查未发现失败标记。

## 2026-05-25 12:40 CST partial 状态

- nd-1 screen 仍在: `ssprep_exp35_20260525_003215_nd1`
- nd-2 screen 仍在: `ssprep_exp35_20260525_003215_nd2`
- nd-1 已生成 final NPZ: 23/64，当前正在运行 `L=4,q=0.1500`
- nd-2 已生成 final NPZ: 23/64，当前正在运行 `L=4,q=0.1500`
- 两节点失败标记均为 0。
- partial 回收到本地后，pooled 点数为 23/80：`L=3` 全部 q 完成，`L=4` 已完成 `q=0.0800..0.1400`。
- 回收的 46 个 final NPZ 均确认 `cluster_update_enabled=False` 且 `cluster_update_requested_enabled=False`。
- 回收的 46 个 final NPZ 均为 `adaptive_pt_rounds=3`。
- 当前 production partial adaptive PT 汇总: mean flow error `0.1495`，local 墙时比 `0.8744`，min swap `0.2492`，mean swap `0.5768`。

## 2026-05-26 partial 状态

- nd-1 screen 仍在: `ssprep_exp35_20260525_003215_nd1`
- nd-2 screen 仍在: `ssprep_exp35_20260525_003215_nd2`
- nd-1 已生成 final NPZ: 53/64，当前正在运行 `L=6,q=0.1300`
- nd-2 已生成 final NPZ: 53/64，当前正在运行 `L=6,q=0.1300`
- 已完成范围: `L=3,4,5` 全部 q；`L=6` 已完成 `q=0.0800..0.1200`。
- 剩余范围: `L=6,q=0.1300..0.2300`，共 11 个 q 点。
- 两节点失败标记均为 0。

## 2026-05-26 运行检查

- nd-1 screen 仍在: `ssprep_exp35_20260525_003215_nd1`
- nd-2 screen 仍在: `ssprep_exp35_20260525_003215_nd2`
- nd-1 已生成 final NPZ: 58/64，当前正在运行 `L=6,q=0.1800`
- nd-2 已生成 final NPZ: 58/64，当前正在运行 `L=6,q=0.1800`
- 已完成范围: `L=3,4,5` 全部 q；`L=6` 已完成 `q=0.0800..0.1700`。
- 当前 `L=6,q=0.1800` chunk 进度: nd-1 为 181/256，nd-2 为 160/256。
- 剩余范围: `L=6,q=0.1800` 后半段和 `q=0.1900..0.2300`，共约 6 个 q 点。
- 两节点失败标记均为 0。

## 2026-05-27 运行检查

- nd-1 screen 仍在: `ssprep_exp35_20260525_003215_nd1`
- nd-2 screen 仍在: `ssprep_exp35_20260525_003215_nd2`
- nd-1 已生成 final NPZ: 62/64，当前正在运行 `L=6,q=0.2200`
- nd-2 已生成 final NPZ: 62/64，当前正在运行 `L=6,q=0.2200`
- 已完成范围: `L=3,4,5` 全部 q；`L=6` 已完成 `q=0.0800..0.2100`。
- 当前 `L=6,q=0.2200` 生产 worker 已启动约 10-15 分钟，尚未写出生产 chunk；worker CPU 正常占用。
- 剩余范围: `L=6,q=0.2200` 和 `q=0.2300`，共 2 个 q 点。
- 两节点失败标记均为 0。

## 2026-05-27 完成状态

- nd-1/nd-2 的 screen 已退出，生产进程已全部结束。
- 两节点最终都是 `final_npz=64/64`，`failed_markers=0`，`active_submit_procs=0`。
- 已回收本地并完成 pooled 分析，`L=3,4,5,6` 全部 64 个 `(L,q)` 点完整。
- pooled 主图已生成: `analysis/fixed_p050_q080_230_exp35_joint_pq_adaptive_pt_nd12_pooled_sem95.png` 与 `analysis/fixed_p050_q080_230_exp35_joint_pq_adaptive_pt_nd12_pooled_gap_ci95.png`。
- adaptive PT 生产汇总为 `adaptive_pt_rounds=3`，cluster update 已关闭，参数与 launch 配置一致。

## pilot 参数结论

exp35 adaptive PT pilot 的 1/3/5 轮比较显示 f flow 误差和 local update 墙时比变化很小；正式扫描采用 `adaptive_pt_rounds=3` 作为稳健默认。

## 回收命令

```bash
HOST_RUN_IDS=3d_toric_exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_003215_nd1,3d_toric_exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_003215_nd2 \
LOCAL_RUN_ID=exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_nd12 \
scripts/collect_exp35_fixed_p050_q_scan_nd12.sh
```
