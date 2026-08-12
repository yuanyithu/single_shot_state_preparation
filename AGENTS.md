# Project D：2D 发表结果整理契约

本 worktree/branch 只承担 2D toric code 的证据审计、重分析与发表级交付整理。`CLAUDE.md` 只导入本文件；设备和 Python 环境由工作区上层规则管理，不在项目文件中重复。

## 接手顺序

1. 先读 `data/2D_final/README.md`，确认当前阶段、权限和交付目录。
2. 再读 `data/2D_final/EVIDENCE.md`，按其中的证据分类、provenance 和缺口接手。
3. 需要追溯历史输入时，读 `data/2d_toric_code/README.md`，然后只打开任务相关 run 的 README、manifest 与原始结果。
4. 旧 run README 只记录当时说法；若与 `data/2D_final/EVIDENCE.md` 冲突，以后者为准。

## 唯一目标与范围

- 唯一目标是把可审计的 2D toric code 结果整理到 `data/2D_final/`，形成可复现的正式重分析、图表和说明。
- `data/2d_toric_code/` 是 legacy source tree：保留原始证据和 provenance，不就地改写历史结论或覆盖原始资产。
- expander code 与 3D toric code 在本 worktree 均为 out-of-scope；不得从其 status、run、validation 或节点进度接手，也不得修改其数据。
- 共享 `src/` 和 `tests/` 默认只读。只有正式重分析明确需要、且能证明不改变其他工作线语义时，才做任务范围内的代码变更。

## 证据权限

- 正式结果必须能追溯到仓库相对源路径、参数网格、disorder 数、NPZ 字段/shape、manifest 完成状态及记录的 source SHA；缺项必须显式披露。
- `RETAIN_FOR_FORMAL_REANALYSIS` 只表示值得进入正式审计，不等于已经获准发表。
- `LEGACY_VALIDATION_REQUIRED` 只允许窗口选择、A/B 和内部诊断；通过修复后 sampler 的 transport 审计前，不得进入正式拟合、crossing 或论文图。
- `ARCHIVE_ONLY` 只保留方法史或否定旧说法，不得作为最终物理证据。
- 不以低接受率、低标准误、曲线平滑、manifest `completed` 或文件名中的 `final` 代替 mixing/transport 认证。
- crossing、未 bracket crossing 与严格单边界限必须分开表达；bound 不得画成普通测量点。

## 工作与数据纪律

- 默认只做只读审计和重分析；不得自行启动本地或远端补实验。补算必须先依据 `EVIDENCE.md` 写清最小缺口并取得用户授权。
- 当前阶段不生成最终论文图；旧 `q>0` 数据最多生成清楚标注 `legacy diagnostic` 的内部预览。
- 新的正式脚本、NPZ、JSON、CSV、PNG、PDF、SVG 和说明统一进入 `data/2D_final/`；临时 cache 不得混入交付目录。
- 不复制历史 raw 来伪造新 provenance，不把 sibling worktree 的绝对路径写入提交文件；所有持久指针使用仓库相对路径。
- 不删除历史 2D 数据，不修改其他 worktree，不触碰 expander/3D 资产。
- 不用 `git add .`；只暂存任务范围文件。除非用户明确要求，本阶段不 commit、不 push。
