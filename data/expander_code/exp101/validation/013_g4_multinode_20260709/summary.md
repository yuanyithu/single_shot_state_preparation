# G4.3 多节点一致性 + G4.5 服务器清理

> **PRE_ALIGNMENT（scan/physics v1）：** 本页只记录 scan v1 的历史可复现/运维证据；未覆盖 v2 fingerprint、alias normalization 或 cache isolation，不是当前通过证据。见 `../README.md`。

2026-07-09。

## G4.3 多节点一致性
同一 scan（toric m=2,3, q=0.1, 6 disorder, TI, --num-workers 8）分别在 nd-1、nd-2 跑：
- 两节点各 computed 12/12，failed=[]，num_workers=8（并行生效）。
- 回收对比（本地 env 12）：
  - `disorder_seed_per_disorder` **bit-identical**（内容哈希 seed scope 与节点无关）
  - `q_top_per_disorder` **max abs diff = 0.0，bit-identical**
  - manifest host 分别记录 nd-1 / nd-2
- 结论：可移植 PRNG（splitmix64/xorshift128+，整型精确）+ 内容哈希 seed scope ⇒
  **跨物理节点逐位复现**（强于统计一致）。同 seed 复现=确定性（另见 test_run_scan
  跨目录确定性 + 续采一致单测）。

⚠ 生产提醒（remote-prod-scan checklist）：本次任务过短未观测 load；生产长任务须在
日志确认 `workers=N`、`uptime` load≈N（in-screen $(nproc) 在 cgroup 下误报 1）。

## G4.5 服务器清理
- 每次远端 run（G4.1 smoke、G4.3）回收校验后即删 run dir；当前 `~/.single_shot/runs/`
  无 exp101 scratch 残留。
- 本地 data/ + git 为 single source of truth；服务器仅作生产 NPZ 异地备份（exp102 起）。

**判定：G4.3 ✅ / G4.5 ✅ → Phase 4 全绿（G4.1-G4.5）**
