# 3D Final：上下文与交付入口

本目录是 Project D 当前 3D 分支的唯一活动整理区。历史实验保持冻结；这里保存证据权限、summary-derived 正式汇总图和后续最小补点规格。

## 当前终态

- exp41/003–006 已汇总为 `legacy_delta_only` 的正式有限尺寸 phase-boundary 展示，但不能作为当前 `true_posterior` / reduced-MLD 的论文阈值。
- legacy 模型的 threshold observable 固定为 sign-aware `w0` crossing。`q_top=q_W` 仅作 companion，不单独给 threshold headline。
- q=0 的 `p=0.226843...` 只作有限尺寸内部 anchor。现有数据实测到 `p=0.22`；两者之间的 sharp-knee 形状没有被测量。
- 汇总图只从冻结 summary 转录 crossing 与历史 CI，没有重新 bootstrap、拟合或外推。exp41/005 的两份 per-disorder NPZ 已从共享盘只读恢复到新的活动目录并通过严格校验；冻结 exp41 未回写。
- nd-3 自适应 pilot 已完成 `.230` 与 `.240` 共 12 cells。三个 q 的 `D_q` 到 `.240` 都保持严格负号，因而无 bracket，并按预定 gate 停止；`.225/.235` 与 384-disorder production 均未运行。

## 正式汇总产物

- [高分辨率 PNG](legacy_delta_only_phase_boundary.png) / [单页矢量 PDF](legacy_delta_only_phase_boundary.pdf)：五个 p 上的 L3-L7 `w0` crossing、空心 `q_top=q_W` companion、q=0 internal anchor 与未测区。
- [CSV](legacy_delta_only_phase_boundary.csv) / [JSON](legacy_delta_only_phase_boundary.json)：同源数值、角色、权限、历史源 SHA256、未测区与 raw 缺失状态。
- [绘图脚本](plot_legacy_delta_only_phase_boundary.py)：渲染前校验唯一键、CI、模型/角色、冻结源值、CSV/JSON 一致性和源哈希；不读取 raw，也不执行拟合。

## 阅读顺序

1. 仓库根 [`AGENTS.md`](../../AGENTS.md)：范围、模型边界和永久权限。
2. [`EXPERIMENT_TRIAGE.md`](EXPERIMENT_TRIAGE.md)：exp01–41 逐项判定、正式图权限与完整曲线缺口。
3. [`3D待补实验.md`](3D待补实验.md)：只针对 legacy 模型的自适应最小补点计划。
4. [`legacy_delta_only_nd3_pilot_20260811/`](legacy_delta_only_nd3_pilot_20260811/)：已完成 nd-3 pilot 的规格、严格校验、checkpoint runner、运行 provenance 与机器可读终态。
5. [有测量噪声历史档案](../3d_toric_code/with_measurement_noise/README.md) / [q=0 历史档案](../3d_toric_code/without_measurement_noise/README.md)：冻结证据入口。

## 写入规则

- 后续若明确授权复图或补点，新的图、表和机器可读汇总仍只放在本目录；不改写历史证据。
- 48-disorder 自适应 pilot 已在无 bracket 时结束；当前没有活动远端计算，384-disorder production 未授权。
- `.gitignore` 只精确放行本次正式 PNG/PDF/CSV/JSON，不对白名单以外的 data 产物放宽规则。
- 若研究目标改为 reduced-MLD，必须另立 true-posterior 实现、验证与实验链，不能在本目录给 legacy 结果改名升级。
