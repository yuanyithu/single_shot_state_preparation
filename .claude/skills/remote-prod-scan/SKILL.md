---
name: remote-prod-scan
description: 远端 nd-1/2/3 多节点生产扫描全流程(launch→健康检查→轮询→回收→校验)。当需要在计算节点上启动/监控/回收 MCMC 生产实验(threshold 扫描、相边界、高统计单点)时使用。包含本项目所有已知运维坑的硬性 checklist。
---

# 远端多节点生产扫描流程

模板 launcher:`data/.../exp40_qtop_phase_boundary_20260610/002_production_boundary_20260610/launch_exp40_boundary.sh`(node↔cell、CELLS 可注入、screen 外探测 nproc、`_CELL_SUCCESS.json` 标记)。新实验**复制改造**它,不要从零写。

## 0. 硬性约束(违反任意一条 = 返工)

- **必须由主会话直接执行所有 ssh**:子 agent(Agent tool)没有 ssh 权限,每条含 ssh 的命令都会被拒。编排(launch/轮询/回收)全部在主会话做,不要 spawn agent 去干。
- **前台 `sleep` 被禁**:等待用 `run_in_background` 的 Bash 循环(`while ...; do ssh 检查; sleep 600; done`,加总时长上限),命令退出时会自动唤醒会话。
- **workers 必须在 screen 外探测**:in-screen `$(nproc)` 在 cgroup quota 下会误报 1,导致退化为串行。launcher 已在启动时(screen 外)探测并烘焙;launch 后必须在日志确认 `workers=N`(nd-1/2≈76, nd-3≈92)且 `uptime` load≈N 才算启动成功。
- 远端 python 一律 `conda run --no-capture-output -n 11`;复杂脚本写成 `.py` 再跑,不要 heredoc 进 `conda run`。
- 计算在 nd-1/2/3,文件读取/轮询走 nd-0 共享存储(`ssh yuany 'ls/tail ...'`),只有 screen/load 检查才 `ssh yuany "ssh nd-K ..."`。
- seed_base 规划:新实验的 seed 块不得与历史实验重叠(历史用到 800000–862xxx 段,新实验继续往上排,并写进 README)。
- q=0 与 q>0 参数差异、PT 限制等物理参数坑见 CLAUDE.md「实验参数陷阱」。

## 1. Launch

1. 复制改造 launcher:改 STAGE_DIR、DEFAULT_CELLS、screen 前缀、MASTER_RUN_ID 前缀;机制零改动。`bash -n` 语法检查 + `DRY_RUN=1` 验证 cell 矩阵。
2. **先本地烟测**(conda env `12`,分钟级,小 L/小 disorder/小 m),校验 NPZ 键齐全、物理量方向正确,再 launch 远端。
3. launch 前检查远端无同名 screen、无残留 run dir:`ssh yuany 'ssh nd-K "screen -ls | grep <prefix>"'`。launcher 的 exit 24 = screen 重名,停下排查,不要强行重启。
4. 固定 `RUN_TIMESTAMP`/`MASTER_RUN_ID` 并立刻写 `README.md`(恢复手册:run id、远端路径、log 路径、screen 名、cell 矩阵、seed、回收命令),保证会话中断后任何新会话可恢复。
5. tar 的 `LIBARCHIVE.xattr` 警告是 macOS xattr 噪声,无害。

## 2. 健康检查(launch 后 ~25 分钟,后台 sleep 然后查)

- 日志出现 `workers=N` 行 + `BEGIN cell`;
- `uptime` load 爬到 ≈workers(nd-3 常有其他用户,load 会超出,属正常);
- 出现 task 进度行 `[k/total] L=.. q=.. d=.. flags=PASS`。
- 若 load≈1:串行退化,收集证据(log tail + uptime)停下报告,不要盲目重启。

## 3. 轮询与回收

- 后台 watcher 循环:每 600s `ssh yuany 'ls .../runs/<RUN_ID>/*/collected/*/_CELL_SUCCESS.json ... _CELL_FAILED.json'`,发现新标记即退出(总时长 ≤7.5h,超时就重启 watcher;单 cell 典型 5–6h)。
- 每个 cell 完成**立即**拉回:`ssh yuany 'tar -C <run_root>/collected -cf - <ptag>' | tar -xf - -C <local>/collected/`,并马上用 conda env `12` 校验:
  - shape:`q_top_per_disorder == (nL, nq, ndis)`、`delta_f_per_disorder == (nL, nq, ndis, 8)`;
  - `pass_fraction`(crossing 核心区应 ≈1.0,高 q 无序侧 0.7–0.98 属预期);
  - q_top 随 q 基本单调不增;wall time 记录。
- cell 失败:拉 log 尾部 ~80 行留证,远端 runner 会自动继续后续 cell;同 seed 重发单 cell(CELLS 只含该 cell + 新 MASTER_RUN_ID)。不要盲目整体重启。

## 4. 收尾

- 全部 cell 成功后确认节点级 `_SUCCESS.json` + 日志 `ALL CELLS OK`;写 `ndK_status.md`。
- 提交遵守 .gitignore:data/ 下只有 `.py`/`.md` 可正常提交,launcher `.sh` 按惯例 `git add -f`;NPZ/png/json/笔记 不入库。更新 `笔记/实验报告.md`(中文、最新条目在顶部)。
