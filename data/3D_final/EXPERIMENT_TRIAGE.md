# 3D toric code 实验取舍索引（exp01–41）

## 1. 判定口径

本索引只回答“已有 3D 证据现在能画什么、能声称什么”。它不重写冻结证据，也不把后来的模型定义倒灌成旧实验当时已经测过的量。

两条结论优先于所有旧 README：

1. exp40/41 属于 `legacy_delta_only`。exp41/003–006 可用于该历史模型的正式展示，但没有当前 `true_posterior` / reduced-MLD 的论文阈值资格。
2. decoding threshold 只由 sign-aware `w0` 的有限尺寸 crossing 定义。`q_top`（exp41 中也记为 `q_W`）只作伴随 purity/finite-size 曲线；旧 `Delta f` gap headline 撤销。

类别含义：

- `FINAL_MODEL_SPECIFIC`：可进入 legacy 模型主文，但必须醒目标模型名和 caveat。
- `INTERNAL_Q0_ANCHOR`：只作 q=0 有限尺寸内部定位，不是精确/渐近阈值。
- `METHOD_VALIDATION_ONLY`：只支持实现、估计量或 gate；不支持物理边界。
- `SUPPLEMENTARY_PHYSICAL_ONLY`：可展示有限范围内的物理分离，但不定阈值。
- `REUSABLE_RAW_HEADLINE_REVOKED`：raw 可按现口径重算；旧标题、旧阈值和旧图注失效。
- `RETRACTED_DO_NOT_PLOT`：物理量、网格或 gate 有根本问题；正式物理图禁用。
- `PROVENANCE_ONLY`：smoke、scout、性能或运维记录，只用于说明实验演进。

早期 exp05–33 的大部分活动目录已在历史清理中移除。表中的 `git show <rev>:<path>` 是可复核的 Git 对象定位，不是虚构的当前路径。

## 2. exp01–41 逐项取舍

| 实验 | 当前类别 | legacy 展示资格 / 用途 | reduced-MLD 资格 | 判定、理由与替代 | 权威证据定位 |
|---|---|---|---|---|---|
| exp01 | `PROVENANCE_ONLY` | 否；管线 smoke | 否 | 只证明 q=0 路径可运行。替代：q=0 定位看 exp10。 | [exp01 README](../3d_toric_code/without_measurement_noise/exp01_q0_pipeline_smoke/README.md) |
| exp02 | `PROVENANCE_ONLY` | 否；窗口 scout | 否 | 低 p 饱和，未包围 crossing。替代：exp04/10。 | [exp02 README](../3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout/README.md) |
| exp03 | `PROVENANCE_ONLY` | 否；窗口 scout | 否 | 右移后仍未覆盖可信 crossing。替代：exp04/10。 | [exp03 README](../3d_toric_code/without_measurement_noise/exp03_q0_right_shift_scout/README.md) |
| exp04 | `INTERNAL_Q0_ANCHOR`（历史佐证） | 仅内部/方法背景 | 否 | 首次看到 interior q=0 window，但统计与终态不如 exp10；不得单独报精确阈值。替代：exp10 anchor。 | [exp04 README](../3d_toric_code/without_measurement_noise/exp04_q0_crossing_window_scout/README.md) |
| exp05 | `RETRACTED_DO_NOT_PLOT` | 仅 provenance | 否 | q>0 本地预检，旧 observable 且非物理统计。替代：exp37 方法链、exp41 legacy 物理。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp06 | `RETRACTED_DO_NOT_PLOT` | 仅 provenance | 否 | all-zero disorder 单样本不能代表 disorder average。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp07 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | 仅 4 disorders，且属于旧 q>0 observable；诱人形状不授予权限。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp08 | `RETRACTED_DO_NOT_PLOT` | 仅失败诊断 | 否 | convergence 仅 3/21，通过率不足且未命中主窗口。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp09 | `PROVENANCE_ONLY` | 否；one-day 对照 | 否 | q=0 快速 deep 对照，不优先于 exp10。替代：exp10。 | [exp09 README](../3d_toric_code/without_measurement_noise/exp09_q0_oneday_deep_fixed/README.md) |
| exp10 | `INTERNAL_Q0_ANCHOR` | 仅内部 anchor/图中标记 | 否 | `384/384` chunks；`p=0.226843...` 只称有限尺寸内部 anchor，不称 exact/asymptotic threshold。 | [exp10 README](../3d_toric_code/without_measurement_noise/exp10_q0_oneday_deep_relaunch/README.md) |
| exp11 | `RETRACTED_DO_NOT_PLOT` | 仅 provenance | 否 | L=5 缺失，只有 partial L3–L4；不能做三尺寸 crossing。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp12 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | 旧 q>0 fine scan/旧 observable；后续 sector 与 estimator 审计已改变权限。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp13 | `RETRACTED_DO_NOT_PLOT` | 仅窗口 provenance | 否 | coarse 右侧侦察，不是正式 crossing。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp14 | `RETRACTED_DO_NOT_PLOT` | 仅窗口 provenance | 否 | fine window 仍使用旧 observable。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp15 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | 左侧 dense A 的旧 q>0 observable。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp16 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | 左侧 dense B 的旧 q>0 observable。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp17 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | 左侧 fine grid 的旧 q>0 observable。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp18 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | exp15–17 的旧综合 crossing，不具现行模型/observable 权限。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp19 | `RETRACTED_DO_NOT_PLOT` / `PROVENANCE_ONLY` | 否；方向侦察 | 否 | 快速摸底且使用旧 q>0 observable，只保留方向修正的演进记录。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp20 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | q=0.05 高力度复本仍基于旧 q>0 observable/mixing gate。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp21 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | exp20 池化 headline 已被后续统计和观测量审计撤销。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp22 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | L6 extension 的 mixing 诊断差，且仍是旧 observable。替代：exp41 的 L7 证据。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp23 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | 四尺寸综合图受 L6 mixing 污染。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp24 | `RETRACTED_DO_NOT_PLOT` | raw 仅历史 provenance | 否 | 三节点 dense raw 属旧 observable；不能复用旧 threshold。替代：exp41。 | [exp24a/b/c 归档](../3d_toric_code/with_measurement_noise/exp25_q001_q050_q100_p018_022_dense_combined_summary/) |
| exp25 | `RETRACTED_DO_NOT_PLOT` | 否；诱人形状也禁画 | 否 | pooling 完整不等于 mixing/observable 正确；旧 p crossing headline 撤销。替代：exp41。 | [exp25 README](../3d_toric_code/with_measurement_noise/exp25_q001_q050_q100_p018_022_dense_combined_summary/README.md) |
| exp26 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | 固定 p 的旧 q scan；相邻 L crossing 不同步且模型已过时。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp27 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | exp26 池化仍无共同 crossing，旧 headline 不保留。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp28 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | L6 q>0 convergence 全失败、PT swap 近零。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp29 | `RETRACTED_DO_NOT_PLOT` | 仅 mixing 失败 provenance | 否 | 四尺寸综合受 L6 冻结污染。替代：exp37 gate 与 exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp30 | `PROVENANCE_ONLY` | 否；性能基准 | 否 | cluster runtime 比较不产生物理统计。替代：不需要物理替代；阈值看 exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp31 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | q>0 gate 仅 5/180，L 对 drift 不同步。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp32 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | q>0 gate 仅 2/75；大统计不能挽救错误 mixing/observable。替代：exp41。 | `git show 9a16ab1:data/3d_toric_code/README.md` |
| exp33 | `RETRACTED_DO_NOT_PLOT` | 仅 corrected-observable 过渡 provenance | 否 | corrected scan 是 exp34 前驱，未形成现行合格网格。替代：exp37 方法链、exp41 物理。 | `git show 73ec8f5:data/3d_toric_code/with_measurement_noise/exp34_fast3d_p050_q000_075_L345_corrected_observable_20260518_nd123/README.md` |
| exp34 | `RETRACTED_DO_NOT_PLOT` | 否 | 否 | 固定 `p=0.0500`；34/45 点完成，L7 未启动，网格不完整且无可靠共同 crossing。替代：exp41。 | [exp34 README](../3d_toric_code/with_measurement_noise/exp34_fixed_p050_q000_080_L34567_corrected_observable_20260524_final_stopped_after_L6q060_nd12/README.md) |
| exp35 | `RETRACTED_DO_NOT_PLOT` | 仅失败审计 | 否 | frozen logical sectors、acceptance 近零且 convergence gate 漏判；高统计曲线无物理权限。替代：exp37 sector-resolved 方法。 | [exp35 audit](../3d_toric_code/with_measurement_noise/exp35/exp35_joint_pq_adaptive_pt_fixed_p050_q080_230_L3456_20260525_nd12/audit_exp35_20260527.md) |
| exp36 | `RETRACTED_DO_NOT_PLOT` | 仅 sampler/mixing provenance | 否 | 后续发现 sector-handling/observable 语义错误；初态 TV gate 不能认证正确逻辑标签。替代：exp37/033–039。 | [exp36 acceptance audit](../3d_toric_code/with_measurement_noise/exp36/acceptance_audit_20260602.md) |
| exp37 | 混合：`METHOD_VALIDATION_ONLY`（033–039）；`RETRACTED_DO_NOT_PLOT`（020–032） | 方法附录 only | 否 | 033–039 的 exact、sector-TI、BAR 与 estimator gate 可保留；001–012 仅早期验证；020–032 含错误 sector label 或下溢 `flip_reweight/FEP`，禁作物理曲线。替代物理结果：exp41。 | [033 model anchor](../3d_toric_code/with_measurement_noise/exp37/033_stageA_model_anchor_20260603/summary.md), [039 production audit](../3d_toric_code/with_measurement_noise/exp37/039_stageG_production_curve_20260604/summary.md), [旧 032](../3d_toric_code/with_measurement_noise/exp37/032_final_corrected_qgrid_20260603/) |
| exp38 | 混合：`METHOD_VALIDATION_ONLY`（P0/P1/P3/P4）；`SUPPLEMENTARY_PHYSICAL_ONLY`（P2/P5） | 方法附录；P2/P5 可作 legacy 高 q 分离补图 | 否 | exact/paired/BAR/acceptance audit 有效；P2/P5 只证明高 q 有限尺寸分离，未解析共同三尺寸 crossing。替代阈值：exp41。 | [P0](../3d_toric_code/with_measurement_noise/exp38/001_p0_regression_anchor_20260604/summary.md), [P4](../3d_toric_code/with_measurement_noise/exp38/005_p4_acceptance_20260605/summary.md), [P5](../3d_toric_code/with_measurement_noise/exp38/006_p5_production_curve_20260605/summary.md) |
| exp39 | 混合：`REUSABLE_RAW_HEADLINE_REVOKED`（004,006–008）；`METHOD_VALIDATION_ONLY`（008 exact/FSS）；其余 provenance | raw 可重算 `w0/q_top`；方法附录 | 否 | `Delta f` threshold headline 系统性偏高并撤销；004/006–008 raw 仅能按现口径重算，008 的 exact-vs-MCMC 与 estimator 排序可保留。替代阈值：exp41。 | [007 撤销说明](../3d_toric_code/with_measurement_noise/exp39_q_threshold_scout_20260605/007_phase_boundary_deltaf_20260608/summary.md), [008 exact audit](../3d_toric_code/with_measurement_noise/exp39_q_threshold_scout_20260605/008_near_pc_pcrossing_20260608/exact_vs_mcmc_L2.md), [008 FSS](../3d_toric_code/with_measurement_noise/exp39_q_threshold_scout_20260605/008_near_pc_pcrossing_20260608/qnear0_wide/summary.md) |
| exp40 | 混合：`RETRACTED_DO_NOT_PLOT`（002–003）；`REUSABLE_RAW_HEADLINE_REVOKED` + `SUPPLEMENTARY_PHYSICAL_ONLY`（004–005） | 004–005 仅作 exp41 数据 provenance，不独立出 headline | 否 | 002–003 的 48-disorder 相边界受首变号低偏与小 L 影响；004–005 高统计 raw 已被 exp41 统一重算并吸收，其补充权限不超出 provenance。替代：exp41/003–006。 | [003 旧边界](../3d_toric_code/with_measurement_noise/exp40_qtop_phase_boundary_20260610/003_boundary_analysis_20260610/summary.md), [004](../3d_toric_code/with_measurement_noise/exp40_qtop_phase_boundary_20260610/004_p011_highstats_20260611/summary.md), [005](../3d_toric_code/with_measurement_noise/exp40_qtop_phase_boundary_20260610/005_boundary_highstats_20260612/summary.md) |
| exp41 | 混合：`METHOD_VALIDATION_ONLY`（001–002）；`FINAL_MODEL_SPECIFIC`（003–006） | legacy 主文（003–006）；001–002 方法 provenance | 否 | 统一从 `delta_f` 重算 sign-aware `w0`，共同 crossing 核心 gate 合格。只保留 p=0.05–0.22 的 legacy 平台/曲线；排除饱和假 crossing、低 pass 深尾、p=0.05 L7 非单调尾与未测 sharp-knee。reduced-MLD 的替代只能是全新 true-posterior 实验。 | [003](../3d_toric_code/with_measurement_noise/exp41/003_p011_L7_prod_384dis_20260621/summary.md), [004](../3d_toric_code/with_measurement_noise/exp41/004_p021_L7_prod_384dis_20260624/summary.md), [005](../3d_toric_code/with_measurement_noise/exp41/005_p022_L7_knee_384dis_20260627/summary.md), [006](../3d_toric_code/with_measurement_noise/exp41/006_plateau_fill_p005_p017_L7_20260630/summary.md) |

## 3. exp41 正式使用边界

可用内容仅限 `legacy_delta_only`：

- exp41/003–006 在 p=`0.05,0.11,0.17,0.21,0.22` 上的统一 `w0` 与伴随 `q_top/q_W` 曲线。
- exp40/004–005 中被 exp41 明确复用的 L=3/4/5（p=0.11 另含 L=6）raw provenance。
- crossing 核心区通过率与 bootstrap 证据；不得把深尾通过率外推到整条曲线。

正式图必须排除或降级：

- 有序饱和段中差值恰为零造成的“首点 crossing”，尤其简并的 L3–L4、L6–L7 对。
- crossing 核心以外的低 pass 深尾。p=0.05,L=7 在 q>=0.048 的 `w0` 非单调回升是 TI-grid 伪影，不作物理趋势。
- exp39 的 `Delta f` threshold、exp40/002–003 的 48-disorder 相边界及所有 valid-only 补图。
- “sharp knee 已测得”这一旧表述。实测只说明 p=0.22 尚未下弯；q=0 anchor 在 `p=0.226843...`，中间没有采样，因而 knee 位置、宽度和形状均未测。

## 4. 已交付的正式汇总图

本目录现已交付 [PNG](legacy_delta_only_phase_boundary.png)、[PDF](legacy_delta_only_phase_boundary.pdf)、[CSV](legacy_delta_only_phase_boundary.csv)、[JSON](legacy_delta_only_phase_boundary.json) 和 [绘图/校验脚本](plot_legacy_delta_only_phase_boundary.py)。权限固定为：

- 图是 exp41/003–006 冻结 crossing 的 `summary-derived` 汇总，不冒充需要 per-disorder raw 才能生成的固定-q `w0/q_top` 主曲线。
- `w0` L3-L7 crossing 是唯一 threshold observable；`q_top=q_W` 使用空心标记并明确为 companion only。全部点都是有限尺寸结果，没有渐近拟合。
- 历史 95% CI 原样继承；本机没有重新 bootstrap，也不把它们描述为 paired-disorder CI。
- q=0 的 `p=0.226843...` 只标为 finite-size internal anchor；`p in (0.22,0.226843...)` 只加“not sampled”阴影，不连线、不画 schematic knee、不外推。
- JSON 固定模型权限、未测区、源文件 SHA256 与 exp41/005 per-disorder NPZ 缺失状态；脚本在渲染前逐项 fail-closed 校验 CSV/JSON 和冻结源。

## 5. 尚缺的完整曲线与条件式补点

完整固定-q 主曲线仍需 exp40/41 共同 legacy 网格的 per-disorder raw：p=`0.05,0.11,0.17,0.21,0.22`，q=`0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070`，共同 L=`3,4,5,7`。exp41/005 的 `p=0.22` 两份 NPZ 已从共享盘恢复到新的活动目录并严格验证，但其他共同网格 raw 缺口仍未因此补齐；完整曲线仍不得用 valid-only 子集代替。

[`3D待补实验.md`](3D待补实验.md) 的自适应 legacy pilot 已完成：`.230` 未翻号后按规则追加 `.240`，总计 12 cells；三个 q 在 `.240` 的 `D_q` 与 95% CI 仍全部严格为负。状态机因此对三个 q 均以 `no_flip_by_0240` 停止，没有运行 `.225/.235`，没有形成 production bracket，也没有升档至 384。

该 pilot 只排除了原候选池内的严格翻号，不外推 `.240` 以外的边界，也不改变上面的正式 summary-derived 图。若继续寻找 legacy bracket，需要先重新设计 p/q 扫描范围并获得新授权；不能把本次停止解释为不存在相边界。

若目标改为当前 reduced-MLD：停止使用该 legacy 补点计划，先独立实现并验证 `true_posterior`，再重新设计 disorder、网格、gate 与 production；exp41 的 legacy 论文权限不可继承。
