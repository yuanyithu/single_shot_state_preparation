# exp40/002 production run — recovery notes

- MASTER_RUN_ID: `exp40_boundary_20260610_031758` (RUN_TIMESTAMP=`20260610_031758`)
- Remote root: `/home/DATA1/users/yuany/.single_shot/runs/exp40_boundary_20260610_031758/{nd1,nd2,nd3}`
- Logs: `/home/DATA1/users/yuany/.single_shot/logs/exp40_boundary_20260610_031758_nd{1,2,3}.log`
- Screens: `exp40B_20260610_031758_nd{1,2,3}` (one per node, cells run sequentially inside)
- Manifests (local): `remote_runs_manifest_nd{1,2,3}.json` in this directory.

Cell matrix (q grid for every cell: 0.008,0.012,0.016,0.020,0.024,0.028,0.032,0.038,0.046,0.058;
L=3,4,5; 48 disorder; seeds below):

| node | p / seed_base |
|---|---|
| nd-1 | 0.01/850000, 0.08/853000, 0.17/856000 |
| nd-2 | 0.03/851000, 0.11/854000, 0.20/857000 |
| nd-3 | 0.05/852000, 0.14/855000, 0.22/858000 |

## How to check / resume collection (any later session)

Remote completion markers (visible from nd-0, shared storage — no need to ssh into nd-K):

```bash
ssh yuany 'ls /home/DATA1/users/yuany/.single_shot/runs/exp40_boundary_20260610_031758/*/collected/*/_CELL_SUCCESS.json 2>/dev/null'
ssh yuany 'tail -5 /home/DATA1/users/yuany/.single_shot/logs/exp40_boundary_20260610_031758_nd1.log'
```

Pull one completed cell back (example nd1 / p0p01):

```bash
ssh yuany 'tar -C /home/DATA1/users/yuany/.single_shot/runs/exp40_boundary_20260610_031758/nd1/collected -cf - p0p01' \
  | tar -xf - -C "<this dir>/nd1/collected/"
```

Re-launch a single failed cell with the same seed (example):

```bash
CELLS="nd-1|0.08|853000|0.008,0.012,0.016,0.020,0.024,0.028,0.032,0.038,0.046,0.058" \
MASTER_RUN_ID=exp40_boundary_retry_$(date +%Y%m%d_%H%M%S) \
MANIFEST_PATH=<this dir>/remote_runs_manifest_retry.json ./launch_exp40_boundary.sh
```

Expected per-cell wall time ≈ 6–7 h (76/92 workers); 3 cells/node sequential ⇒ nd-1/2 ≈ 20 h, nd-3 ≈ 17 h.
Analysis afterwards: `../003_boundary_analysis_20260610/analyze_exp40_boundary.py` (reads `nd*/collected/p*`).
