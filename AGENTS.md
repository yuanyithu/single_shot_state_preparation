# Project D：3D 最终整理契约

本文件是本分支唯一常驻上下文正本；`CLAUDE.md` 只导入本文件。设备环境由工作区上层规则管理，不写入本文件。

## 唯一目标与权威链

- 当前唯一目标是整理 `data/3D_final/`：审计历史 3D 证据、固定使用权限，并记录已完成的 legacy 自适应最小补点 pilot。
- 权威阅读顺序固定为：`AGENTS.md` → `data/3D_final/README.md` → `data/3D_final/EXPERIMENT_TRIAGE.md` → 被引用的历史证据。
- `data/3d_toric_code/**` 是冻结、只读的历史证据源；仅其两个分支入口 README 可维护权限指针，实验内容与 raw 不改写。
- 已交付 `legacy_delta_only` summary-derived 汇总图；yuany/nd-3 的 48-disorder 自适应 pilot 已在 `.240` 无翻号时停止，当前无活动远端计算，不自动升 384。提交与推送只在用户明确授权时执行。

## 模型与科学边界

- exp40/41 使用的模型统一标为 `legacy_delta_only`。exp41/003–006 足以制作该历史模型的正式展示，但不能升级为当前 `true_posterior` / reduced-MLD 的论文阈值。
- 阈值只由 sign-aware `w0` 的有限尺寸 crossing 定义；`q_top` 只作同图伴随量，用于展示 purity 与有限尺寸行为，不单独产生 decoding-threshold headline。
- 旧 `Delta f` gap headline 已撤销；饱和区零差值假 crossing、低 pass 深尾和 `p=0.05` 的 L7 非单调尾部均不得进入正式物理图。
- exp10 的 `p=0.226843...` 只称 `q=0` 有限尺寸内部 anchor，不称精确或渐近阈值；exp04 只作历史佐证。
- exp41 实测只到 `p=0.22`；`p=0.22` 与 q=0 anchor 之间的 sharp-knee 形状没有被测量。可标未测区或条件式推断，不得画成实测边界。
- 任何 reduced-MLD 结论都要求重新实现并验证 `true_posterior` 后开展独立实验；不得继承 legacy 数据的论文权限。

## 证据与作图权限

- exp01–41 的逐项权限、替代方案和证据定位只以 `data/3D_final/EXPERIMENT_TRIAGE.md` 为准。
- 正式 legacy 图优先使用 exp41/003–006 统一从 `delta_f_per_disorder` 重算的 `w0`；exp40/004–005 仅作为 exp41 已复用数据的 provenance。
- 旧目录中的 `PASS` 只认证其当时声明的数值/方法 gate，不自动授予当前模型或当前图的论文权限。
- invalid 或 missing 必须 fail-closed；不得用 valid-only 子集补正式 mean、crossing 或 FSS。
- 正式汇总图、机器可读数据、绘图脚本和 `3D待补实验.md` 只放在 `data/3D_final/`；`.gitignore` 只精确放行已交付的 PNG/PDF/CSV/JSON。
- exp41/005 两份 per-disorder NPZ 已从共享盘只读恢复到活动目录并严格校验；冻结 exp41 不回写。补点只按 `data/3D_final/3D待补实验.md` 的自适应 gate 执行。

## 工作区边界

- 本次远端权限仅覆盖已经结束的 `legacy_delta_only` yuany/nd-3 pilot；未经新授权不启动后续计算。expander code（包括 exp101–105 与 exp102 blocker）、2D、其他 remote/production 计算和其他 worktree 均明确越界。
- 不从 deployment、source snapshot、raw 或历史 validation 目录接手开发。
- 只暂存任务范围文件；不用 `git add .`。保留用户已有改动，不清理或重写无关文件。
- Markdown 链接、`git diff --check`、`CLAUDE.md` 单行导入、AGENTS 行数和 diff 范围是每次上下文整理的最低验收项。
