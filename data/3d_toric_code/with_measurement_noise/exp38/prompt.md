在 exp37 已验证的 sector-resolved TI 主线之上，把生产网格放大到统计可分辨并出图。本指令会被反复执行；每次只推进一步，靠状态文件接力，不要重头再来。

## 每次执行的固定流程

1. **读规范**：先读仓库根 `AGENTS.md`（物理图像与 L·T·S 分解、运行规则、避坑清单——本任务以 AGENTS.md 为准），再读
   `data/3d_toric_code/with_measurement_noise/exp38/detail_plan.md`（唯一权威规范，含全局铁律与 P0–P5 各阶段的目的/做什么/成功闸门/交付物）。
2. **定位进度**：读 `data/3d_toric_code/with_measurement_noise/exp38/STATUS.md`。
   - 若不存在，按 detail_plan 第 3 节初始化为 P0–P5 全 `TODO`，然后从 P0 开始。
   - 「当前阶段」= STATUS 表里第一个状态不是 `PASS` 的阶段。
3. **只做当前这一个阶段**：按 detail_plan 里该阶段的定义，产出它的交付物，并运行它的**成功闸门**。
   - 阶段较大时，本次可以只推进一个有意义的增量（例如先把闸门脚本/launcher 写好、先跑出标定或一批分片），但**本次必须有可检查的进展并落盘**。
4. **判定并落盘**：
   - 闸门**通过** → 在 STATUS.md 把该阶段标 `PASS`，并在「闸门达成数字」列贴关键对比值（如 `grid TV=0.011, paired SEM=0.018 vs unpaired 0.12, δq_top=0.008`）、交付目录、日期；增量更新 `笔记/实验报告.md`；**然后停止本次执行**（下一次 /goal 自动接下一阶段）。
   - 闸门**未过** → 标 `FAIL` 或 `DOING`，写清楚**未过的是哪条判据、差多少、下一次准备怎么修**；**停止本次执行**，不要硬闯下一阶段。
5. 全部 P0–P5 都 `PASS` 后，按 detail_plan 第 6 节核对 Definition of Done，在 STATUS.md 顶部写 `ALL DONE` 并停止，不再产生新动作。

## 不可违反的铁律（细节以 detail_plan 第 2 节为准）

- **闸门只认外部参照的数字**：精确枚举 / 第二种独立估计量（退火 + 双向 BAR）/ 解析锚点。「内部自洽、多初态一致、曲线平滑、全 PASS、看着收敛」都不算成功。
- **主估计量是 sector-resolved TI，直接复用 exp37 已验证代码路径**（`src/exp37_sector_ti.py` + exp37 的 runner/验收/绘图脚本），不改估计量语义、不另起炉灶。**禁止**把单步 FEP / `flip_reweight` 当生产估计量；第二方法只能是退火 + 双向（BAR / 双向 AIS），仅作交叉验证。
- **红线**：若某点除主导扇区外所有 `w_g` 都低于统计分辨率，必须标 `UNRESOLVED/FAIL` 并输出 `q_top` 下界，**绝不允许报成 `q_top≈1`**。「`q_top≈1` 且 `w_sub` 未分辨」永远是失败。
- **方法核心 = 跨 L 公共 disorder + 配对差分**：三个 L 必须用同一组公共 disorder 种子（`disorder_seed_scope` 不含 `lattice_size`）；crossing 结论必须由配对差分 `Δ=q_top(L_i)−q_top(L_j)` + 配对 disorder bootstrap CI 给出，单看独立均值重叠不算否证、也不算证实 crossing。
- **效率**：先本地（env `12`）做 P0 回归锚点和 P1 配对 de-risk + wall-time 标定，再上服务器；q 网格分两批（crossing 区密+多 disorder、深有序区稀）；按 L 分节点并行；用 `sleep-until` 阻塞等哨兵，不轮询。
- 一次只推进一个阶段；上一阶段未 PASS、未贴对比数字，不进下一阶段。
- 数据只写 `data/` 内；遵守 AGENTS.md 的 conda 环境（本地 `12`，远端 `11` + `screen` + `conda run --no-capture-output`，复杂脚本写 `.py` 文件不用 heredoc，不临时装包）；**L=5 显式设 `--max-effective-num-burn-in-sweeps` 防 burn 自动放大**；每个阶段放 exp38 下新的带序号子目录（从 `001` 起递增）。
- 在服务器部署程序后，如果需要等待程序结束，务必用 `sleep-until` 预期时间、设监控、等待运行结束。

## 本次输出（简短）

结束时用 3–6 行汇报：当前阶段、做了什么、闸门对比数字、PASS/FAIL、下一步。不要长篇大论，把证据写进 summary.md 和 STATUS.md。
