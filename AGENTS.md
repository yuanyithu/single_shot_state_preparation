# Project D 操作契约

本文件是项目唯一常驻上下文正本；`CLAUDE.md` 只导入本文件。设备专属环境由工作区上层 `AGENTS.md` 管理，不在此重复。

## 接手与权限

- 当前主线是 quantum expander code。先读 `data/expander_code/exp102/status.md`，再读 `data/expander_code/exp102/validation/INDEX.md`；只按其中指针打开所需契约和证据。
- exp102 当前为 `BLOCKED_BEFORE_REMOTE`；在 `status.md` 明确解除 blocker 且用户授权前，不启动 remote、formal、held-out 或 production，也不把诊断值写成物理结果。
- exp101 的物理与聚合权威分别是 `data/expander_code/exp101/PHYSICS_CONTRACT.md` 和该目录的 `AGENTS.md`、`status.md`、`validation/README.md`。
- 旧 3D toric 工作与 expander code 不混用；仅在明确处理 legacy 3D 时，按任务读取 `data/3d_toric_code/with_measurement_noise/README.md` 或 `data/3d_toric_code/without_measurement_noise/README.md`。
- `deployment/`、validation 内的 source snapshot 与 raw 是冻结证据，不是活动工作区；不得从其中接手开发或批量改写上下文文件。

## 文档与工作区

- `AGENTS.md` 只写长期规则和权威指针，保持不超过 120 行；不得追加 run 编年史、SHA、具体门值或单次禁令。
- `status.md` 只保留当前终态、未解 blocker、最近 2--3 个条目和指针，保持不超过 150 行。
- `validation/INDEX.md` 是 001 起的证据总账；每个 validation 一行，原终态与受控类别分列，不用类别改写证据权限。
- `HISTORY.md` 保存旧上下文与完整编年史，只追加、默认不加载；单次细节仍以对应 validation README/报告为准。
- `笔记/实验报告.md` 面向人类，每次实验最多 10 行摘要并链接权威证据；完整实验后更新。
- 默认只保留仓库根 worktree。临时 worktree 不得嵌入 `data/`；移除前审计 dirty/untracked 与提交可达性，合并后立即清理。
- 不用 `git add .`；只暂存任务范围文件。版本大改验证后使用清晰 commit 并 push，不混入缓存、临时产物或无关修改。

## 运行与数据

- Python 环境服从当前设备的上层指令；使用 `conda run` 时加 `--no-capture-output`，依赖安装使用设备约定的包管理器。
- 本地实验、smoke、profile 和 sanity check 的最终 JSON/NPZ/PNG/MD/CSV 必须在项目 `data/` 下；临时 cache 可例外。
- 多进程显式传 `--num-workers N`，不要依赖 screen 中的 `$(nproc)`。
- 服务器入口为 `ssh yuany` 到 nd-0；除非用户另行授权，exp102 计算只用 nd-2/nd-3。远端 Python 使用 conda `11`，长任务必须在 `screen` 中运行。
- 服务器根目录仅用 `~/.single_shot/`：长期保留 `runs/` 和 `logs/`；代码副本、cache、smoke/profile scratch 在产物取回校验后清理。
- clean-source tree 只能经 `run_verified_source.sh` 或等价的禁止 bytecode 包装运行；出现 `__pycache__` 会触发 exit 67，失败 run 不原地重跑。
- 每个实验使用编号目录；多节点结果放在同一实验目录的独立子目录。冻结产物与本地 `data/` + Git 是事实源，服务器只作异地运行与备份。

## 永久科学纪律

1. 非零 syndrome 下物理全零态不在支持集；平移坐标的零就是 P。不得用全链从零或 P 开始制造表面收敛。
2. 共冻不等于收敛。保留 P、低能且 logical/B 不同的 S/MAP、以及 U 等合法对抗初态；删除或统一初态即无效。
3. state change、label change、acceptance 或局部 ESS 不等于 logical transport；混合证据必须直接测真实慢变量。
4. 当前 q=0 HGP 的已知慢变量是 collapsed-B basin；新 kernel 开跑前先书面回答它如何跨 B 势垒。
5. 候选先声明 primary、正交确认方或严格 bound；三者皆非只允许小时级本地结构筛。同一 collapsed-B 机制族不能互作独立确认。
6. 冻结进 estimator mixture 的每个成员都有否决权；不得看结果后剔除成员或重配权重。
7. full support、跨 proposal 一致、低 jackknife SE 或高 collision 不能认证未观测 tail；须有严格 tail/normalizer bound 或机制独立确认。
8. bound 算得快不等于对交付目标够紧；先验证 tightness 的量级，再讨论资源。
9. exact/解析端点与 plug-in estimate 字段不混用；plug-in 标注 `no confidence coverage`，无偏 U-statistic 不裁剪。
10. 节点 worker 的 `SUCCESS` 不等于 preflight PASS；下游只认 aggregate status。
11. 资源门按 `2 x (生成 + replay + 分析 + 固定开销)` 从获批 stage 总账倒推；单轨迹 cap 只作 discovery 淘汰线。
12. 冻结复杂实验前做科学 red-team：目标分布与支持集、坐标和初态、慢变量与自环、估计量与交付量、门的假阳/假阴和共同失效、exact/独立确认、权限边界；并先回答“完全成功会解锁哪个 blocker”。
13. 同族失败不追加救援；validation 编号不得为探索性 kernel 无限增长。
14. 每完成 5 个 validation 或每 48 小时，向用户提交一次不超过一页的进展与方向 checkpoint。

## Expander 生产约束

- convention 固定：`sector=x_error`、`H_check=H_Z`、stabilizer move=`H_X` 行、logical move=`logical_X`、observable=`logical_Z`、制备态=`|+>_L`；对偶 `z_error/H_X` 才对应 `|0>_L`。
- posterior 固定为 `pi(e|y_eff) proportional exp[-K_p|e|-K_q|H_check e xor y_eff|]`，其中 `y_eff=H_check epsilon_data_true xor measurement_error`；真实错误不得进入能量或采样比值。
- meta-check decoder、preparation representative、logical-sector section 的 domain 不同；一般 q>0 `effective_syndrome` 不得传给 logical-sector section。
- 区分 absolute/relative characters、planted-class mass、posterior purity 与 MAP success；公共 `w0` 废弃。普通 TI/sampled 只填 plug-in estimated bounds。
- full-sector TI 仅用于 `k<=10`；large-k pairwise 只给 free-energy-gap diagnostics。任一 disorder gate 失败都保存 raw 并标 `INVALID`。
- 参数点 fail-closed：有 invalid 为 `SAMPLING_INSUFFICIENT`，有 missing 为 `INCOMPLETE`，正式 mean/SEM/crossing/FSS 输入为 NaN；valid-only 统计只作诊断。
- publication/FSS 只经规定 loader 读取；exp101 使用 `src.scan_results.load_publication_q_top`，exp102 不得交给 exp101 loader。
- 跨环境 proposal artifact 只由指定节点构造一次，其余节点审计同一冻结字节、代数和离散 replay；不得要求不同求解器重建结果逐字相同或混用 artifact。
