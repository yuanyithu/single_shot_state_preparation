# notes/02 — 环境与执行记录（exp101.scan.v2）

本文只记录可复现实验所需的执行环境与调度事实。物理语义以
`../PHYSICS_CONTRACT.md`（`exp101.physics.v2`）为唯一权威；本页不构成第二份物理契约。

## 本地（开发与验证主环境）

- 主机：darwin（Mac），conda env **`12`**（项目规定，勿切换）。
- 2026-07-07 实测：python 3.12.12, numpy 2.4.1, numba 0.65.1, ldpc 2.4.1, scipy 1.17.0, matplotlib 3.10.8, pytest 9.0.3。
- 已复现坑：本地 `conda run -n 12 python - <<'PY'` heredoc 吞输出（与 CLAUDE.md 远端坑同源）——一律先写 .py 再运行。
- **已复现坑（G2.4 排障代价惨痛）**：`conda run` 不加 `--no-capture-output` 会**整体捕获子进程输出直到退出**——长任务/被 kill 的任务看起来"零输出"，无法判断进度，faulthandler 转储也被吞。**本地所有 conda run 一律加 `--no-capture-output`**（远端 launcher 已有此规则，本地同样强制）。
- 测试运行约定：`cd "project D" && conda run --no-capture-output -n 12 python -m pytest data/expander_code/exp101/tests -q`。

## 远端（nd-1 / nd-2 / nd-3，经 `ssh yuany` → `ssh nd-{1,2,3}`）

- **基础环境状态：已验（2026-07-09, G4.1；属于 PRE_ALIGNMENT 环境证据）**：
  - nd-0 (storage) 可达；`~/.single_shot/{runs,logs,repos,mpl-cache,SERVER_README.md}` 完整。
  - env 11 三节点齐全：numba 0.65.1（与本地同）、ldpc 2.3.7（本地 2.4.1）、numpy 2.3.4（本地 2.4.1）、scipy 1.16.3。**版本小差无碍**：可移植 PRNG（splitmix64/xorshift128+）位级复现与版本无关；代码只用基础 numpy+numba njit+ldpc BpLsdDecoder。
  - 核数（screen 外 nproc/affinity 直测）：**nd-1=80, nd-2=80, nd-3=96**。
  - `run_scan.py` 已支持 task/disorder 级 `ProcessPoolExecutor`，并使用 `spawn` context 避免
    numba 与 `fork` 的死锁风险。CLI 通过显式 `--num-workers N` 启用；默认 `1` 仍是串行，
    不会自动采用 `nproc`。
- 运行规范（CLAUDE.md 全文适用）：env `11`、screen 后台、`conda run --no-capture-output`、复杂脚本先落 .py、结果 tar 回本地校验后清 scratch。

## scan v2 执行约束

- 默认 `engine=auto`：`k<=10` 解析为 full-sector TI；`k>10,q>0` 解析为四实例 PT
  observable sampling；`k>10,q=0` 解析为 validated 8-start sampling。manifest 必须同时记录
  requested 与 resolved engine。
- chunk identity 必须包含 `exp101.physics.v2`、`exp101.scan.v2`、canonical ensemble、sector、
  family rule/seed、code/implementation fingerprint 及完整 sampler/estimator 配置；v1 chunk 永不复用。
- 远端生产必须在 screen 外探测并显式烘焙 worker 数，在日志核对 `workers=N` 与实际负载；
  `screen`/cgroup 内的 `nproc` 可能误报 `1`。
- 当前 v2 状态为 `DONE`；多进程、路由、fingerprint 与 validity 语义由
  `validation/014_paper_alignment_20260713/` 认证。exp102 复用时仍须保留相同契约版本和 gate。

## 版本记录规则

- 每个 run 的 manifest 必记：repo commit SHA、worktree dirty 状态、implementation fingerprint、
  numpy/numba/ldpc 版本、hostname、canonical ensemble、requested/resolved engine、code/section/
  observable 指纹、family rule/seed、完整 resolved config 与 RNG 协议字符串。
- 远端与本地 numba 版本不一致时：以统计等价 gate 为准（bit 级复现只在同机同版本内要求）。
