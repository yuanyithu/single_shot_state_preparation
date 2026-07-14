# exp101 local instructions

本目录所有代码、测试、扫描和文档必须遵守 `PHYSICS_CONTRACT.md`
（`exp101.physics.v2`）；它是唯一物理权威。根目录的 3D toric legacy delta-only 公式不适用于
exp101 论文生产模式。

- 生产固定 `sector=x_error`、`ensemble=true_posterior`；持久化前先归一化弃用 alias。
- `epsilon_data_true` 不得直接进入 true-posterior 能量、Metropolis 比值、TI 或 PT swap ratio。
- 不得把一般 `effective_syndrome` 传给 `logical_sector_section`。
- 不得把 `posterior_mass_on_planted_class`、`posterior_purity` 与
  `map_success_probability` 混称；不得恢复公共字段 `w0`。exact/解析端点的 algebraic MAP bounds
  与普通 TI/sampled 的 plug-in estimated bounds 必须使用不同字段和 `map_success_bound_kind`；
  plug-in 固定声明无置信覆盖。
- `k>10` 禁止用 TI 产生 `q_top`；pairwise 结果只允许作为 free-energy-gap diagnostics。
- 任一 convergence gate 失败的 disorder 必须为 `INVALID`，保留未裁剪 raw estimator。无偏
  U-statistic 在有限样本下可为负或越出物理区间，这不是允许裁剪的理由。
- scan v3 参数点只有在所有 planned disorders 均存在且有效时才是 `REPORTABLE`。invalid 或 missing
  必须关闭整点正式 mean/SEM/crossing；valid-only 条件统计仅供诊断，因选择偏差不得用于
  publication/FSS。fraction 一律以 planned disorders 为分母，禁止恢复 `pass_fraction`。
- `exp101.scan.v1/v2` chunk 与 v2 NPZ 不得复用于 v3。publication/FSS 必须调用
  `src.scan_results.load_publication_q_top`，不得绕过 loader 或从 v2 条件均值推断资格。
- `validation/014_paper_alignment_20260713/` 只继续认证 `exp101.physics.v2`；scan v3 认证只认
  `validation/015_aggregation_safety_20260714/`。
- 用户提供的 `../exp101修改说明/` 与 `文章.tex` 是只读来源，不得修改或纳入代码提交。

本文件只规定 exp101 的物理与生产约束；不要在这里写任何单机 conda、主机名或设备配置。
