# 2D Toric Code Finalization

本目录是 `2D` branch/worktree 的唯一接手入口。目标是审计和重分析历史 2D toric code 数据，并最终在这里整理可复现的发表级脚本、数据、图表与说明。

## 当前状态

- 已完成上下文去污染和证据取舍；2026-08-11 经用户授权在 nd-2 执行一次分阶段 pilot，但两条轨道均按预注册 gate 停止，没有启动正式补点，也没有生成最终论文图。详见 [`ND2_STAGED_PILOT_AUDIT.md`](ND2_STAGED_PILOT_AUDIT.md)。
- 两组 `q=0` 数据进入 `RETAIN_FOR_FORMAL_REANALYSIS`，仍需正式重分析后才能获得论文权限。
- 全部候选 `q>0` 主数据为 `LEGACY_VALIDATION_REQUIRED`：可用于选窗和受控 A/B，不得直接支撑 crossing、no-threshold 或其他正式物理结论。
- nd-2 `q=0` A/B pilot 的 B schedule 在两个测试点的平均四起点 spread 仍为 `0.1107/0.1177 > 0.10`，所以计划中的 L9/L11 正式 extension 未启动；`q=0` 缺口保持开放。
- 修复后 `q=0.001, L=11, p=0.0875` transport 哨兵在 A/B 均出现冻结链、Rhat/ESS/spread 失败，结论为 `STOP_Q_POSITIVE`；没有自动扩展其他 `q>0`。
- 方法原型、被覆盖的 overnight 扫描及旧派生图为 `ARCHIVE_ONLY`。
- 两组 retained `q=0` 数据已生成带 provenance 核验、disorder 置信区间和四起点 spread 的[审计预览](Q0_RETAINED_AUDIT_PREVIEW.md)；预览明确不是最终论文图。

完整依据、逐实验网格、provenance、验证结果、目标图和最小缺口见 [`EVIDENCE.md`](EVIDENCE.md)。

## 接手顺序

1. 读 [`EVIDENCE.md`](EVIDENCE.md)，确认数据权限和未解 blocker。
2. 需要追溯历史输入时，读 [`data/2d_toric_code/README.md`](../2d_toric_code/README.md)，再进入对应 run README/manifest。
3. 正式重分析只从 `EVIDENCE.md` 列出的 canonical source path 读取，不从旧 summary 目录的副本接手。
4. 新的正式脚本与资产只写入本目录；历史 source tree 保持不变。

## 权限边界

- 默认只读审计和重分析，不默认补算。任何新计算必须先写清最小缺口并取得用户授权。
- 旧 `q>0` 数据最多生成显著标注 `legacy diagnostic` 的内部预览。
- 不把 `completed` manifest、低标准误、曲线平滑或旧目录名中的 `final` 当作 mixing/transport 认证。
- 不触碰 expander、3D 数据或其他 worktree；不把本机或 sibling worktree 的绝对路径写入持久文件。
