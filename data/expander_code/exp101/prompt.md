【本文件是 exp101 开发循环的 /loop prompt：直接把本文全文作为 /loop 的 prompt 使用（自定节奏），每轮迭代执行以下指令。】

继续推进 exp101：expander code 单发制备统计力学 q_top 管线的开发与正确性验证。

每轮迭代流程：

1. 读 `data/expander_code/exp101/plan.md`（契约）与 `status.md`（进度真值），从「当前指针」取下一个未完成的最小工作项（一个 gate 或其子项）。若「待用户决策」区块有条目阻塞当前项，换下一个不被阻塞的项；若所有项都被阻塞，在最终回复里总结阻塞点，并停止循环（不再安排 wakeup）。
2. 本轮只做这一个有界工作块，做完立刻验证（跑 pytest / 对照 ground truth / 检查产物），不要一轮铺开多个 gate。
3. 证据落盘：每个 gate 的结论必须有证据文件（pytest 输出、JSON/NPZ、图、日志），放在 `exp101/tests/` 产物或 `exp101/validation/` 编号目录（`0NN_极简内容_日期/`，多节点结果放 nd1/ nd2/ nd3/ 子目录）；在 status.md 对应行登记路径与时间戳，并同步维护「validation/ 编号目录索引」。无证据不得标「通过」。
4. gate 失败：先定位、修复、重跑。若失败暴露的是 plan 缺陷或需要用户拍板的分歧（如放宽判据、改规模目标、改定义），写入 status.md「待用户决策」（背景/选项/推荐/影响面），然后换下一个可推进项。绝不为了让 gate 变绿而悄悄放宽判据或删测试。
5. 收尾：更新 status.md（状态/证据/当前指针/changelog 各一行）。一个 phase 全绿时按 plan §5 做 git 提交：message 前缀 `exp101 phase-N:`，只 add 相关文件（禁 `git add .`），大 NPZ 不入库。
6. 硬性遵守 CLAUDE.md 与 plan §5/§6 运行规范，特别是：本地 conda `12`、远端 env `11` + screen + `conda run --no-capture-output`、conda run 禁 heredoc（先写 .py 文件）、`--num-workers` 显式并在日志确认、q=0 不传 `--pt-*`、PT 热端 p_k,q_k<0.5、冷端 logical-flip 接受率是收敛硬判据（per-u、worst-u）、「共冻 ≠ 收敛」。
7. 边界：所有开发与产物都在 `data/expander_code/exp101/` 内；主项目 `src/` 与 `data/3d_toric_code/` 只读（对照时只读运行原脚本，不修改）；服务器只用 `~/.single_shot/runs/exp101_*`，结果 tar 回本地校验后清理服务器 scratch。
8. 远端长任务：提交后不要空转轮询——用后台任务或与任务时长匹配的 wakeup 间隔；等待期间优先推进其它不冲突的本地 gate。
9. 节奏：本轮工作完整收尾（status.md 已更新、无悬空进程/screen 残留）后再安排下一轮 wakeup；短本地任务用短间隔，远端等待用长间隔。

毕业条件（plan §7 全部满足）达成时：写 `exp101/report.md`，更新 `笔记/实验报告.md`，按 CLAUDE.md 维护规则增补 expander 新坑，把跨会话关键结论存 memory，完成最终 git 提交与 push，在 status.md 把循环状态标为 DONE，然后停止循环（不再安排 wakeup）。
