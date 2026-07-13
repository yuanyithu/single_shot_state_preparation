# exp101 local instructions

本目录所有代码、测试、扫描和文档必须遵守 `PHYSICS_CONTRACT.md`
（`exp101.physics.v2`）；它是唯一物理权威。根目录的 3D toric legacy delta-only 公式不适用于
exp101 论文生产模式。

- 生产固定 `sector=x_error`、`ensemble=true_posterior`；持久化前先归一化弃用 alias。
- `epsilon_data_true` 不得直接进入 true-posterior 能量、Metropolis 比值、TI 或 PT swap ratio。
- 不得把一般 `effective_syndrome` 传给 `logical_sector_section`。
- 不得把 `posterior_mass_on_planted_class`、`posterior_purity` 与
  `map_success_probability` 混称；不得恢复公共字段 `w0`。
- `k>10` 禁止用 TI 产生 `q_top`；pairwise 结果只允许作为 free-energy-gap diagnostics。
- 任一 convergence gate 失败的 chunk 必须为 `INVALID`，不得进入 disorder aggregate 或 crossing。
- `exp101.scan.v1` chunk 不得复用于 v2；当前认证只认
  `validation/014_paper_alignment_20260713/`。
- 用户提供的 `../exp101修改说明/` 与 `文章.tex` 是只读来源，不得修改或纳入代码提交。

本文件只规定 exp101 的物理与生产约束；不要在这里写任何单机 conda、主机名或设备配置。
