# exp40/005 高统计边界重扫 — 恢复手册

- MASTER_RUN_ID: `exp40_boundhs_20260612_225345`(RUN_TIMESTAMP=`20260612_225345`)
- 远端根:`/home/DATA1/users/yuany/.single_shot/runs/exp40_boundhs_20260612_225345/{nd1,nd2,nd3}`
- 日志:`/home/DATA1/users/yuany/.single_shot/logs/exp40_boundhs_20260612_225345_nd{1,2,3}.log`
- screen:`exp40B_20260612_225345_nd{1,2,3}`
- 规格(= exp40/004 p=0.11 规格):L=3,4,5;每 (p,节点) 128 disorder(合并 384);q 网格
  `0.012,0.018,0.022,0.026,0.030,0.034,0.040,0.048,0.058,0.070`;grid129/m8192/burn512/TI linear。

每节点 4 个 cell 串行(顺序 0.05 → 0.17 → 0.02 → 0.21),每 cell ~16h,总 ~2.7–3 天:

| p | nd-1 seed | nd-2 seed | nd-3 seed |
|---|---|---|---|
| 0.05 | 873000 | 874000 | 875000 |
| 0.17 | 876000 | 877000 | 878000 |
| 0.02 | 870000 | 871000 | 872000 |
| 0.21 | 879000 | 880000 | 881000 |

p=0.11 直接复用 `../004_p011_highstats_20260611/`(L=3,4,5,6 × 384)。

## 检查 / 回收(任意会话)

```bash
ssh yuany 'ls /home/DATA1/users/yuany/.single_shot/runs/exp40_boundhs_20260612_225345/*/collected/*/_CELL_SUCCESS.json 2>/dev/null'
ssh yuany 'tar -C /home/DATA1/users/yuany/.single_shot/runs/exp40_boundhs_20260612_225345/ndK/collected -cf - p0pXX' \
  | tar -xf - -C "<此目录>/ndK/collected/"
```

单 cell 失败:同 seed 重发(CELLS 只含该 cell + 新 MASTER_RUN_ID)。
分析:每 p 三块按 disorder 轴合并(`../004_*/analyze_p011_highstats.py` 的 load 逻辑),
headline = q_top L3-L5 crossing + w0 交叉验证,终图 = 5 个高统计点 + (0.227,0) 锚点 + 旧 48-dis 点灰色背景。
