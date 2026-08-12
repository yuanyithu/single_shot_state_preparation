# 2D Toric Code 实验取舍与证据审计

## 1. 审计范围与结论

本报告筛选现有 2D toric code 资产，并记录 2026-08-11 经用户授权执行的一次 nd-2 分阶段 pilot；不生成最终论文图，也不修改 legacy raw。证据权限分为：

- `RETAIN_FOR_FORMAL_REANALYSIS`：值得进入正式重分析，但尚不等于可发表。
- `LEGACY_VALIDATION_REQUIRED`：数据量和 per-disorder tensor 足够做 A/B、选窗与诊断；通过修复后 sampler 的 logical-transport 审计前，不得进入正式物理结论。
- `ARCHIVE_ONLY`：只保留方法开发史、被覆盖结果或否定旧说法的审计记录。

当前结论是：保留两组 `q=0` 生产数据做正式重分析；nd-2 q0 pilot 未通过 mixing gate，因此没有启动正式 extension；候选 `q>0` 数据全部维持 legacy validation，且修复后单点 transport 哨兵也明确失败；其余实验只归档。旧目录名中的 “final” 或 “no threshold” 不改变这个权限。

## 2. `RETAIN_FOR_FORMAL_REANALYSIS`

| 实验 | Canonical source | 网格与样本 | Provenance / 完整性 | 正式用途与 caveat |
|---|---|---|---|---|
| [`q0_threshold_deep_nd3_20260420_221142`](../2d_toric_code/without_measurement_noise/q0_threshold_deep_nd3_20260420_221142/README.md) | `data/2d_toric_code/without_measurement_noise/q0_threshold_deep_nd3_20260420_221142/scan_result_multi_L_q0_geometric_multistart_threshold_deep.npz` | `q=0`；`L={3,5,7}`；`p=0.0900:0.0025:0.1250`（15 点）；每点 512 disorder；四起点 `00/10/01/11` | nd-3；manifest `720/720 completed`；tensor `(3,15,512)`；记录 SHA `a15c3326…` | 现有最强小/中尺寸 `q=0` 数据。正式重分析应从 per-disorder/per-start tensor 重建 CI、crossing 和起点 spread；旧 README 的 `p≈0.10–0.106` 不是预先接受的最终定值。 |
| [`q0_control_extension_nd3_20260421_225303`](../2d_toric_code/without_measurement_noise/q0_control_extension_nd3_20260421_225303/README.md) | `data/2d_toric_code/without_measurement_noise/q0_control_extension_nd3_20260421_225303/scan_result_multi_L_q0_control_extension.npz` | `q=0`；`L={9,11}`；`p=0.0950:0.0025:0.1100`（7 点）；每点 1024 disorder；同四起点 | nd-3；manifest `448/448 completed`；tensor `(2,7,1024)`；记录 SHA `a197215b…` | 大尺寸校准。共同网格上 `q_top(L9)-q_top(L7)` 始终为正（到 `p=0.1100` 仍为 `+0.00222`），所以 L7–L9 crossing 未被 bracket；不得把边界趋势画成普通 crossing。 |

保留的含义是“进入正式审计”，不是“现状可直接作图”。两份数据的四起点诊断都必须保留，不能只使用起点平均后的平滑曲线。

### 2.1 nd-2 `q=0` extension pilot（`PILOT_GATE_STOP`）

2026-08-11 的授权计划先在 `L=11, p={0.1000,0.1100}` 上用 128 disorder/点、四起点和 common-random disorder 做 A/B，再决定是否启动 L9/L11 正式 extension。固定源码为 `70ea84cd5fe800948a619e4a070c693e684e5b4b`；完整配置、NPZ、manifest、日志和机器审计见 [`ND2_STAGED_PILOT_AUDIT.md`](ND2_STAGED_PILOT_AUDIT.md)。

- A=`2000/10/600` 的平均四起点 spread 为 `0.164833, 0.157778`，失败。
- B=`4000/10/1200` 降至 `0.110745, 0.117669`，但仍都超过预注册上限 `0.10`，失败。
- A/B mean `q_top` 差为 `-0.003829, +0.002760`，通过 `|Δ|≤0.01`；两份 manifest 均 `16/16 completed`，NPZ integrity/provenance 也通过。

按预注册规则，B 仍失败就停止正式 L11 补点且不继续加长链。因此计划中的 768 个正式 chunks 未启动，现有 `q=0` crossing 证据没有变化：L7–L9 仍未被 bracket，`p=0.1125…0.1250` 的 L9/L11 缺口仍开放。pilot 数据只解释停线决策，不作为正式 extension 曲线拼接进 retained 数据。

## 3. `LEGACY_VALIDATION_REQUIRED`

### 3.1 Canonical `q>0` 输入

| `q` | Canonical source | `L` | `p` 网格 | disorder/点 | Manifest | 平均接受率范围 |
|---:|---|---|---|---:|---:|---:|
| 0.0010 | `data/2d_toric_code/with_measurement_noise/no_threshold_final_nd3_20260421_225039/q_0p0010/scan_result_multi_L_q0p0010_no_threshold_final_common_random.npz` | `3,5,7,9,11` | `0.0750:0.0025:0.1000`（11 点） | 1024 | `1760/1760` | `1.97e-4–4.64e-4` |
| 0.0025 | `data/2d_toric_code/with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427/q_0p0025/scan_result_multi_L_q0p0025_measurement_noise_threshold_search_common_random.npz` | `3,5,7` | `0.0850:0.0025:0.1100`（11 点） | 2048 | `1056/1056` | `7.72e-4–1.10e-3` |
| 0.0050 | `data/2d_toric_code/with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427/q_0p0050/scan_result_multi_L_q0p0050_measurement_noise_threshold_search_common_random.npz` | `3,5,7` | `0.0800:0.0025:0.1075`（12 点） | 2048 | `1152/1152` | `1.51e-3–2.24e-3` |
| 0.0075 | `data/2d_toric_code/with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427/q_0p0075/scan_result_multi_L_q0p0075_measurement_noise_threshold_search_common_random.npz` | `3,5,7` | `0.0725:0.0025:0.1000`（12 点） | 2048 | `1152/1152` | `2.15e-3–3.17e-3` |
| 0.0100 | `data/2d_toric_code/with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427/q_0p0100/scan_result_multi_L_q0p0100_measurement_noise_threshold_search_common_random.npz` | `3,5,7` | `0.0600:0.0025:0.0900`（13 点） | 2048 | `1248/1248` | `2.43e-3–3.85e-3` |
| 0.0150 | `data/2d_toric_code/with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427/q_0p0150/scan_result_multi_L_q0p0150_measurement_noise_threshold_search_common_random.npz` | `3,5,7` | `0.0450:0.0025:0.0800`（15 点） | 2048 | `1440/1440` | `2.94e-3–5.20e-3` |
| 0.0200 | `data/2d_toric_code/with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427/q_0p0200/scan_result_multi_L_q0p0200_measurement_noise_threshold_search_common_random.npz` | `3,5,7` | `0.0350:0.0025:0.0700`（15 点） | 2048 | `1440/1440` | `3.57e-3–6.59e-3` |

上述全部 run 来自 nd-3，`q=0.0010` 记录 source SHA `fa3750de…`，threshold-search 六个 `q` 记录 `cfd76e86…`。数据量与 tensor 结构足够重分析，但不具正式证据权限，原因不是样本数不足，而是：

1. 它们均生成于通用 q-positive sampler 修复 `bf6e670`（2026-04-22 20:43 +08:00）之前。该修复让 `q>0` 链也执行 zero-syndrome/kernel sweeps，直接针对 single-bit 路径冻结风险。
2. 主批次平均接受率只有约 `1.97e-4–6.59e-3`。低接受率本身不是失败证明，但与修复时间线共同构成必须复核的硬风险。
3. NPZ 只有单一聚合链的 `logical_observable_mean_values_per_disorder_tensor`；没有 q-positive per-start tensor、Rhat、ESS、sector occupation 或真实 logical-transport 指标。标量 `q0_num_start_chains=4` 不能证明 `q>0` 跑了四起点。
4. manifest 的 `completed` 只证明任务落盘，不证明链跨越了慢逻辑扇区。

因此，这些数据当前只能用于确定旧窗口、设计 common-disorder A/B 和生成带 `legacy diagnostic` 标记的内部预览。未经修复后 sampler 的多起点 transport 审计，不得据此声称 finite crossing、no-threshold、临界指数或 FSS。

### 3.2 Observable 修复为什么不是这里的首要 blocker

2026-05-18 的 nonlinear-section observable 修复把观测量写成

\[
O_u(c;\eta)=(-1)^{\langle z_u,\,c+\eta+r(H_Zc)+r(H_Z\eta)\rangle}.
\]

历史 2D 路径使用线性高斯消元 section。对线性 `r`，新旧观测量只相差一个对固定 disorder 不随 MCMC 样本变化的符号；而归档 `q_top` 是 `mean_u(m_u^2)`，对这个符号不变。per-disorder `q_top` 也能由归档的三个 `m_u` 平方平均逐项精确重建。因此后来的 nonlinear-section 修复不会自动推翻旧 linear-section 2D `q_top`。

这不是对旧数据的认证：运行时记录的 source SHA 已不在当前 Git 对象库，无法做字节级源码重建；更关键的 sampler freezing 与 logical transport 仍未解决。

### 3.3 修复后 nd-2 transport 哨兵（`STOP_Q_POSITIVE`）

同一授权计划只运行 `q=0.001, L=11, p=0.0875` 单点哨兵：严格重建 legacy chunk0 前 16 个 disorder，四起点×2 replicas，A/B 为 600/1200 measurements，base burn-in 2200、间隔12；启用修复后的 zero-syndrome sweep，关闭 cluster，不使用 PT。disorder uniforms/bits 的固定 SHA-256 全部通过，新的 MCMC seed 独立记录。

外部硬门禁先检查真实 sector 变化，再检查 Rhat/ESS；结果为：

| 指标 | A | B | 要求 |
|---|---:|---:|---:|
| max Rhat | 1.215964 | 1.109233 | `<1.05` |
| min ESS | 3.802 | 5.806 | `>200` |
| mean `q_top` spread | 0.189713 | 0.163367 | `<0.03` |
| mean winding acceptance | 3.286e-4 | 3.344e-4 | `>1e-4` |
| 每个 disorder 最多未换 sector 链数 | 3 | 2 | `0` |
| 每条链最小占据 sector 数 | 1 | 1 | `≥2` |

两臂只有 winding acceptance 通过；冻结、Rhat、ESS 和 spread 均失败。前/后半 occupation mean TV 为 `0.09719/0.07203`，mean absolute block drift 为 `0.07172/0.05852`；相对 legacy per-disorder `q_top` 的平均配对差为 `-0.21398/-0.22669`。因此结论是 `STOP_Q_POSITIVE`，不扩展其他 q-positive 点。该哨兵是 go/no-go 诊断，不是论文物理证据；旧 q-positive 主数据继续保持 `LEGACY_VALIDATION_REQUIRED`。

## 4. `ARCHIVE_ONLY`

| 实验 | Canonical source / 网格 | Provenance | 归档理由 |
|---|---|---|---|
| [`baseline_multisize_local`](../2d_toric_code/without_measurement_noise/baseline_multisize_local/README.md) | `data/2d_toric_code/without_measurement_noise/baseline_multisize_local/scan_result_multi_L.npz`；`q=0`，`L=3,5,7`，`p=0.06:0.01:0.14`，120 disorder/点 | 本地聚合文件；无 manifest/source SHA | 早期单聚合基线，无 per-disorder tensor；只保留方法史。 |
| [`kernel_mix_local`](../2d_toric_code/without_measurement_noise/kernel_mix_local/README.md) | `data/2d_toric_code/without_measurement_noise/kernel_mix_local/scan_result_multi_L_kernel_mix{,_focus,_highp,_farhigh}.npz`；主扫 `p=0.08:0.005:0.13`/100 disorder；focus 40、high-p 30、far-high 20 disorder/点；均 `L=3,5,7` | 本地开发文件；无 manifest/source SHA | 小样本 kernel 开发系列，无正式 provenance/transport 认证。 |
| [`q0_geometric_multistart_local`](../2d_toric_code/without_measurement_noise/q0_geometric_multistart_local/README.md) | `data/2d_toric_code/without_measurement_noise/q0_geometric_multistart_local/scan_result_multi_L_q0_geometric_multistart.npz`；`q=0`，`L=3,5,7`，`p=0.08:0.005:0.13`，12 disorder/点，四起点 | 本地原型；无 manifest/source SHA | 验证几何 multistart 接口；被 512-disorder deep run 覆盖。 |
| [`measurement_noise_overnight_nd3_20260421_004035`](../2d_toric_code/with_measurement_noise/measurement_noise_overnight_nd3_20260421_004035/README.md) | `data/2d_toric_code/with_measurement_noise/measurement_noise_overnight_nd3_20260421_004035/q_0p{0100,0200,0300}/scan_result_multi_L_*_measurement_noise_threshold_deep_common_random.npz`；`q={0.01,0.02,0.03}`，`L=3,5,7`，`p=0.09:0.0025:0.125`，2048 disorder/点 | nd-3；每个 q 的 manifest `1440/1440`；记录 SHA `5cfb0832…` | 原窗口摸底，被更细的左移 threshold-search 覆盖；同样早于 sampler 修复且无 transport 认证。 |
| [`q0_control_summary_20260422`](../2d_toric_code/without_measurement_noise/q0_control_summary_20260422/README.md) | `data/2d_toric_code/without_measurement_noise/q0_control_summary_20260422/q0_control_{sem95,crossing_drift}.png`；无新的 raw | 旧派生图，来源为 deep/control | 正式重分析必须从 canonical NPZ 重建，不复用旧图。 |
| [`no_threshold_evidence_nd3_20260422`](../2d_toric_code/with_measurement_noise/no_threshold_evidence_nd3_20260422/README.md) | `data/2d_toric_code/with_measurement_noise/no_threshold_evidence_nd3_20260422/`；聚合 `q={0.001,0.0025,0.005,0.01}` 的旧图表和四份 NPZ 副本 | 派生 summary；四份 NPZ 的 SHA-256 与 canonical source 各自相同 | 不是独立证据；机器摘要 `paper_claim_supported=false` 应保留为否定旧过强表述的审计证据。 |

## 5. 只读完整性与 provenance 核验

本轮对两组 retained、七组 legacy 以及三组 overnight 共 12 份 canonical NPZ 做了只读核验：

- 所有要求的曲线、per-disorder `q_top`、接受率与三个 logical observable tensor 均为有限值。
- `q_top_curve_matrix` 与 per-disorder tensor 均值、`q_top_std_curve_matrix` 与 `ddof=1` 标准差、接受率曲线与接受率 tensor 均值的最大绝对差都是 `0`。
- 每个 per-disorder `q_top` 与三个归档 `m_u` 的平方平均最大绝对差为 `0`。
- 12 份 manifest 合计 `14736/14736 completed`，`0 failed`、`0 pending`；manifest SHA 与对应 NPZ 内记录一致。
- summary 目录的 `q=0.001/0.0025/0.005/0.01` 四份 NPZ 与 canonical source 的 SHA-256 分别完全相同，确认只是重复副本。

Manifest/NPZ 记录的五个 source SHA 如下；当前仓库对象库对它们均无法 `git cat-file`，所以保留 provenance 字符串但不能声称已恢复精确运行源码：

| Run family | 记录的 source SHA |
|---|---|
| q0 threshold deep | `a15c3326fcc07844e06cc02ff176cf39ab7c0bbb` |
| q0 control extension | `a197215bd18e9ffc160b4864b7f54239ff4e39da` |
| q-positive overnight | `5cfb0832b1fec7c017a6673b4d74967722d69ea0` |
| q-positive threshold search | `cfd76e861cba70cec506e2c5f3bbcb681a004b09` |
| q=0.001 no-threshold final | `fa3750ded9109d64d275fc103cb19e1963a62bf0` |

新增 nd-2 pilot 使用当前可解析的固定提交 `70ea84cd5fe800948a619e4a070c693e684e5b4b`。q0 A/B manifest 都是 `16/16 completed`、`0 failed`、`0 pending`；四份聚合 NPZ 的 SHA-256、字段/shape、有限值、聚合重建、配置和 source SHA 均经远端与本地独立审计一致。收集目录只保留聚合 NPZ、manifest、日志、控制 provenance 和审计 JSON/CSV；remote chunks 原地保留，cache、preflight scratch 和自动 PNG 未进入交付。逐文件哈希见 [`collection_audit.json`](nd2_runs/2d_final_nd2_staged_20260811_210600/collection_audit.json)。

## 6. 未来目标图（本轮不生成）

主文候选是 `q={0,0.001,0.005,0.01,0.02}` 的 `q_top(p)` 分面图：每个面板按 `L` 分线，并从 per-disorder tensor 给出 disorder 置信区间。`q={0.0025,0.0075,0.015}` 放补充材料。

- crossing、未 bracket crossing 和严格单边界限使用不同符号/箭头；bound 不画成普通点。
- 只有通过证据审计的数据可进入正式版本。当前 `q>0` 面板没有正式作图权限。
- 旧 `q>0` 数据如用于内部预览，图面和文件名都必须明确标注 `legacy diagnostic`。
- 旧 summary PNG/PDF 不复用；最终图必须由本目录中的受控脚本从 canonical source 重建。

## 7. 当前缺口与下一步权限

1. `q=0` 的物理缺口仍是 `L=9,11`、`p={0.1125,0.1150,0.1175,0.1200,0.1225,0.1250}`，但原 A/B 路径已被 pilot gate 否决。在重新设计并论证 L11 mixing 方法前，不得直接启动原正式网格，也不得只继续加长同一路径。
2. `q>0` 的修复后单点哨兵已经证明当前非 PT、cluster-off 路径在目标点仍缺乏 transport。不得自动扩展其余 `q`；后续若探索其他 transport 方法（例如新的受控 proposal/PT 设计），必须另写最小问题和新 gate 并重新取得用户授权。
3. 若未来任一受控点与旧曲线不一致，现有 `q>0` 曲线全部重算；不得靠少量补点或更换误差条升级旧数据权限。

本轮授权已在两个预注册停止条件处耗尽；没有尚在运行的本地或远端 2D 任务。
