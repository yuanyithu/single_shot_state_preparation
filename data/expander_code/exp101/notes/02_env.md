# notes/02 — 环境记录（G0.3）

## 本地（开发与验证主环境）

- 主机：darwin（Mac），conda env **`12`**（项目规定，勿切换）。
- 2026-07-07 实测：python 3.12.12, numpy 2.4.1, numba 0.65.1, ldpc 2.4.1, scipy 1.17.0, matplotlib 3.10.8, pytest 9.0.3。
- 已复现坑：本地 `conda run -n 12 python - <<'PY'` heredoc 吞输出（与 CLAUDE.md 远端坑同源）——一律先写 .py 再运行。
- **已复现坑（G2.4 排障代价惨痛）**：`conda run` 不加 `--no-capture-output` 会**整体捕获子进程输出直到退出**——长任务/被 kill 的任务看起来"零输出"，无法判断进度，faulthandler 转储也被吞。**本地所有 conda run 一律加 `--no-capture-output`**（远端 launcher 已有此规则，本地同样强制）。
- 测试运行约定：`cd "project D" && conda run --no-capture-output -n 12 python -m pytest data/expander_code/exp101/tests -q`。

## 远端（nd-1 / nd-2 / nd-3，经 `ssh yuany` → `ssh nd-{1,2,3}`）

- **状态：已验（2026-07-09, G4.1）**：
  - nd-0 (storage) 可达；`~/.single_shot/{runs,logs,repos,mpl-cache,SERVER_README.md}` 完整。
  - env 11 三节点齐全：numba 0.65.1（与本地同）、ldpc 2.3.7（本地 2.4.1）、numpy 2.3.4（本地 2.4.1）、scipy 1.16.3。**版本小差无碍**：可移植 PRNG（splitmix64/xorshift128+）位级复现与版本无关；代码只用基础 numpy+numba njit+ldpc BpLsdDecoder。
  - 核数（screen 外 nproc/affinity 直测）：**nd-1=80, nd-2=80, nd-3=96**。
  - ⚠ **exp101 run_scan.py 目前单进程串行**，不会自动吃满多核——生产（exp102/G4.2）前需加 ProcessPoolExecutor + 显式 `--num-workers`（disorder/task 级并行），并遵守 remote-prod-scan 的 workers 探测坑。G4.1 smoke 用串行足够。
- 运行规范（CLAUDE.md 全文适用）：env `11`、screen 后台、`conda run --no-capture-output`、复杂脚本先落 .py、结果 tar 回本地校验后清 scratch。

## 版本记录规则

- 每个 run 的 manifest 必记：repo commit SHA、numpy/numba/ldpc 版本、hostname、ensemble、frame 指纹、RNG 协议字符串。
- 远端与本地 numba 版本不一致时：以统计等价 gate 为准（bit 级复现只在同机同版本内要求）。
