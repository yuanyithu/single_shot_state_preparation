# nd-2 分阶段 pilot 审计结论

本轮按预注册的“分阶段最小闭环”在 nd-2 执行 `q=0` A/B 与单点 `q=0.001` transport 哨兵。两条轨道均触发预设停止规则，因此没有启动 `q=0` 正式 L9/L11 补点，也没有扩展其他 `q>0`。本文件是审计说明，不是论文图或新的 threshold 结论。

## 运行与 provenance

- 固定源码提交：`70ea84cd5fe800948a619e4a070c693e684e5b4b`。
- 收集入口：[`nd2_runs/2d_final_nd2_staged_20260811_210600/`](nd2_runs/2d_final_nd2_staged_20260811_210600/)。
- 机器审计：[`pilot_audit.json`](nd2_runs/2d_final_nd2_staged_20260811_210600/pilot_audit.json)；逐轨摘要：[`pilot_summary.csv`](nd2_runs/2d_final_nd2_staged_20260811_210600/pilot_summary.csv)。
- 收集清单与逐文件 SHA-256：[`collection_audit.json`](nd2_runs/2d_final_nd2_staged_20260811_210600/collection_audit.json)。remote chunks、q-positive task parts、preflight scratch、cache/`__pycache__` 与自动 PNG 均未收集，仍原地保留。
- 远端 runner 最终 phase 为 `done_rc_21`：表示 q0 pilot 按 gate 停止，不是进程崩溃。A/B manifest 都是 `16/16 completed`、`0 failed`、`0 pending`。
- 64 workers 均以 `nice=10` 运行并限制在逻辑 CPU `0–31,40–71`；监控每 5 分钟记录 load、内存、进程和 manifest。全程未触发 64 GiB 内存紧急线，也没有操作其他用户进程。

关键聚合文件 SHA-256：

| 资产 | SHA-256 |
|---|---|
| `q0_pilot_A.npz` | `f29859b3997d5b68c6bd1a1f7cf16759960e63aa1381ef52e607e7873e0f179d` |
| `q0_pilot_B.npz` | `b88202bd2fd4dc00a64dc3ab5ae8f05ce791afec418d9ed84b6205df6cf54423` |
| `qpositive_sentinel_A.npz` | `f2c8872c60650f5b782b535b666b818dcf8940fb67346900526af9b876cddf24` |
| `qpositive_sentinel_B.npz` | `d160adb5c31f1bd30144719e600cc317f3c2ed13e84906b108865e60fd2f3dd5` |

## `q=0` pilot：停止正式补点

参数为 `L=11`、`p={0.1000,0.1100}`、128 disorder/点、chunk16、四起点；A 为 `2000/10/600`，B 为 `4000/10/1200`，两臂使用相同 seed `620260811` 和 common-random disorder。

| arm | `p=0.1000` mean `q_top` | `p=0.1100` mean `q_top` | 两点平均四起点 spread |
|---|---:|---:|---:|
| A | 0.666983 | 0.491943 | 0.164833, 0.157778 |
| B | 0.663154 | 0.494703 | 0.110745, 0.117669 |

- A/B mean `q_top` 差（B−A）为 `-0.003829, +0.002760`，通过逐点绝对差 `≤0.01` 的一致性检查。
- A 的两个 spread 都超过 `0.10`，不能选择 A。
- B 虽明显降低 spread，但两个点仍都超过 `0.10`，触发“停止正式 L11 补点且不继续加长链”的预注册规则。
- NPZ 字段/shape、有限值、聚合均值、四起点 spread 重建、common-random 配置、seed、source SHA 和 manifest 完整性均通过；停线原因是 mixing 诊断门禁，不是 provenance 或落盘失败。

因此，计划中的 768 个正式 q0 chunks 一个也没有启动；`p=0.1125…0.1250` 的 L9/L11 缺口仍然存在，不能生成新的 crossing、未 bracket crossing 更新或单边界限。

## `q=0.001` transport 哨兵：停止 q-positive

哨兵使用 `L=11, p=0.0875, q=0.001`，精确重建 legacy chunk0 的完整 32-disorder RNG 顺序后截取前 16 个 disorder；四起点×2 replicas；A/B 分别 600/1200 measurements；base burn-in 2200（L11 有效值 29578）、间隔12；启用修复后的 zero-syndrome sweep，关闭 cluster，不使用 PT。

legacy disorder 的四组冻结哈希（syndrome/data uniforms 与 threshold 后 bits）全部匹配配置；新 MCMC seed 与 legacy disorder seed 分开记录。外部门禁先检查每条链是否真实改变 logical sector，再读取 Rhat/ESS，冻结链的虚假好数值不会获得通过。

| 指标 | A | B | gate |
|---|---:|---:|---:|
| max Rhat | 1.215964 | 1.109233 | `<1.05`，失败 |
| min ESS | 3.802 | 5.806 | `>200`，失败 |
| mean `q_top` spread | 0.189713 | 0.163367 | `<0.03`，失败 |
| mean winding acceptance | 3.286e-4 | 3.344e-4 | `>1e-4`，通过 |
| 每个 disorder 最多未换 sector 链数 | 3 | 2 | 必须为 0，失败 |
| 每条链最小占据 sector 数 | 1 | 1 | 必须至少 2，失败 |
| 前/后半 occupation mean TV | 0.09719 | 0.07203 | 仅报告 |
| mean absolute block drift | 0.07172 | 0.05852 | 仅报告 |
| 相对 legacy per-disorder `q_top` 的平均配对差 | -0.21398 | -0.22669 | 仅诊断 |

winding acceptance 单项通过不能抵消真实 logical transport、Rhat、ESS 和多起点 spread 的联合失败。结论是 `STOP_Q_POSITIVE`：该哨兵不授予旧 q-positive 曲线论文权限，也不自动授权其他 `q` 或更长链。

## 后续权限

1. `q=0`：在重新设计/论证 L11 mixing 方法前，不得直接启动原计划正式补点，也不得仅继续加长同一路径。
2. `q>0`：本轮已给出修复后 sampler 在目标哨兵点仍不满足 transport 的直接证据；其余 legacy q-positive 数据继续保持 `LEGACY_VALIDATION_REQUIRED`。
3. 本轮没有最终论文图、正式 crossing、FSS 或 threshold 数值；`analyze_q0_formal_extension.py` 只作为未来正式数据通过 gate 后的重分析工具，当前没有可供它处理的 formal extension。
