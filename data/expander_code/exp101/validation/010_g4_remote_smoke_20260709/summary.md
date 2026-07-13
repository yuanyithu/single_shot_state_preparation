# G4.1 远端 env 确认 + 单节点 smoke 全往返

> **PRE_ALIGNMENT（scan/physics v1）：** 本页及 `sector_ti_results.npz` 只作历史运维证据；旧 chunk/schema 永不复用于 `exp101.scan.v2`，不是当前通过证据。见 `../README.md`。

2026-07-09。手工最小 launcher（tar src → ssh 运行 → 拉回校验 → 清 scratch）。

## 远端 env（screen 外直测）
| 节点 | 核数 | env 11 numba / ldpc |
|---|---|---|
| nd-0 (storage) | — | `~/.single_shot/{runs,logs,repos,mpl-cache}` 完整 |
| nd-1 | 80 | numba 0.65.1 / ldpc 2.3.7（numpy 2.3.4, scipy 1.16.3）|
| nd-2 | 80 | numba 0.65.1 / ldpc 2.3.7 |
| nd-3 | 96 | numba 0.65.1 / ldpc 2.3.7 |

本地对照 numba 0.65.1（同）/ ldpc 2.4.1 / numpy 2.4.1。版本小差无碍（可移植 PRNG 位级复现与版本无关）。

## smoke 全往返
- 传输：`tar -cf - src | ssh yuany 'tar -x'` → `~/.single_shot/runs/exp101_smoke_20260709/src`
- 运行（nd-1, env 11）：`python -m src.run_scan --family toric --size-list 2 --p-value 0.12 --q-values 0.1 --num-disorders 2 --engine ti`（tiny config）→ computed 2/2
- 回收+校验和：远端/本地 sha256 **一致**（bf8b64bc…）
- schema 校验（本地 env 12）：18 字段齐全，q_top shape (1,1,2) 有限=[0.825, 0.820]，manifest host=**nd-1**、ensemble=true_posterior、numpy/numba 版本已记录、flags 一 PASS 一 TI_GRID_WARN（tiny 9-grid 预期）
- 清理：远端 smoke dir 已删

**判定：G4.1 ✅**（env 确认 + 全往返 + 校验和一致 + 清理）

## 生产（exp102/G4.2）前置 TODO
1. **run_scan.py 目前单进程串行**——加 ProcessPoolExecutor + 显式 `--num-workers`（disorder/task 级），遵守 remote-prod-scan 的 workers 探测坑（screen 外探测、日志确认 workers=N、load≈N）。
2. **commit SHA 记录**：远端只传 src/（无 .git）→ manifest commit=unknown；launcher 需显式传 SHA。
3. 复制改造模板 launcher `.sh`（多节点 cell 矩阵、_CELL_SUCCESS 标记、README 恢复手册）。
