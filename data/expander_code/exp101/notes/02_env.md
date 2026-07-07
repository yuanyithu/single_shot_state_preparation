# notes/02 — 环境记录（G0.3）

## 本地（开发与验证主环境）

- 主机：darwin（Mac），conda env **`12`**（项目规定，勿切换）。
- 2026-07-07 实测：python 3.12.12, numpy 2.4.1, numba 0.65.1, ldpc 2.4.1, scipy 1.17.0, matplotlib 3.10.8, pytest 9.0.3。
- 已复现坑：本地 `conda run -n 12 python - <<'PY'` heredoc 吞输出（与 CLAUDE.md 远端坑同源）——一律先写 .py 再运行。
- 测试运行约定：`cd "project D" && conda run -n 12 python -m pytest data/expander_code/exp101/tests -q`。

## 远端（nd-1 / nd-2 / nd-3，经 `ssh yuany` → `ssh nd-{1,2,3}`）

- **状态：未验（挂账）**。G4.1 前必须完成并回填本节：
  - [ ] `conda run -n 11 python -c "import numpy, numba, ldpc; ..."` 三节点各验一次，记版本。
  - [ ] screen 外探测核数（nproc/affinity；nd 节点既往实测 80/80/96 核），烘焙进 runner 显式 `--num-workers`。
  - [ ] `~/.single_shot/runs/` 可写、`SERVER_README.md` 索引更新方式确认。
- 运行规范（CLAUDE.md 全文适用）：env `11`、screen 后台、`conda run --no-capture-output`、复杂脚本先落 .py、结果 tar 回本地校验后清 scratch。

## 版本记录规则

- 每个 run 的 manifest 必记：repo commit SHA、numpy/numba/ldpc 版本、hostname、ensemble、frame 指纹、RNG 协议字符串。
- 远端与本地 numba 版本不一致时：以统计等价 gate 为准（bit 级复现只在同机同版本内要求）。
